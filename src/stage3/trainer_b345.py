"""
Training utility for Stage-3 B3/B4/B5 baselines.

This trainer keeps stage-2 evaluation style (mapped_exact) and uses
label-rerank prediction for stable local smoke validation.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.b345_model import B345Model  # noqa: E402
from src.data.graph_collator import BioREGraphCollator  # noqa: E402
from src.data.graph_dataset import BioREGraphDataset  # noqa: E402


SPECIAL_TOKENS = ["[E1S]", "[E1E]", "[E2S]", "[E2E]", "[DEP]", "[/DEP]"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _label_key(text: str) -> str:
    return "".join(str(text).strip().upper().split())


def _build_label_key_map(valid_labels: List[str]) -> Dict[str, str]:
    return {_label_key(label): label for label in valid_labels}


def project_prediction_to_label(
    prediction: str,
    valid_labels: List[str],
    label_key_map: Dict[str, str],
) -> str | None:
    pred = prediction.strip()
    if pred in label_key_map.values():
        return pred

    pred_key = _label_key(pred)
    if pred_key in label_key_map:
        return label_key_map[pred_key]

    cpr_match = re.search(r"CPR\s*:\s*([0-9]+)", pred, flags=re.IGNORECASE)
    if cpr_match:
        candidate = f"CPR:{cpr_match.group(1)}"
        if candidate in label_key_map.values():
            return candidate

    for label in sorted(valid_labels, key=len, reverse=True):
        if _label_key(label) in pred_key:
            return label
    return None


def get_generator_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(
        f"[Tokenizer] {model_path}  vocab_size={len(tokenizer)}  "
        f"added={added} special tokens"
    )
    return tokenizer


def get_pubmedbert_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path)


def build_stage3_dataloader(
    file_path: str,
    generator_tokenizer,
    pubmedbert_tokenizer,
    dep_type_vocab_path: str,
    split: str,
    target_mode: str,
    dep_view: str,
    dep_form: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_src_len: int,
    max_input_len: int,
    max_target_len: int,
    max_semantic_len: int,
    seed: int,
) -> DataLoader:
    dataset = BioREGraphDataset(
        file_path=file_path,
        use_dep=(dep_form != "none"),
        dep_view=dep_view,
        dep_form=dep_form,
        target_mode=target_mode,
        max_src_len=max_src_len,
        seed=seed,
        dep_type_vocab_path=dep_type_vocab_path,
    )
    if split in {"dev", "test"}:
        shuffle = False

    collator = BioREGraphCollator(
        generator_tokenizer=generator_tokenizer,
        pubmedbert_tokenizer=pubmedbert_tokenizer,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
        max_semantic_len=max_semantic_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    print(
        f"[Stage3Loader] {Path(file_path).stem} split={split} "
        f"n={len(dataset):,} batch={batch_size}"
    )
    return loader


def build_candidate_label_ids(
    tokenizer,
    candidate_labels: List[str],
    max_target_len: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        dec = tokenizer(
            text_target=candidate_labels,
            max_length=max_target_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    except TypeError:
        dec = tokenizer(
            candidate_labels,
            max_length=max_target_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    label_ids = dec["input_ids"].to(device)
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    return label_ids


def _to_device_batch(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _expand_batch_for_candidates(
    batch: Dict[str, torch.Tensor],
    num_candidates: int,
) -> Dict[str, torch.Tensor]:
    """
    Expand a batch from B to B*C for vectorized label rerank.
    """
    out = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        out[key] = value.repeat_interleave(num_candidates, dim=0)
    return out


def predict_labels_by_rerank(
    model: B345Model,
    batch: Dict[str, torch.Tensor],
    candidate_labels: List[str],
    candidate_label_ids: torch.Tensor,
) -> List[str]:
    batch_size = int(batch["input_ids"].size(0))
    num_candidates = len(candidate_labels)
    if num_candidates == 0:
        return []

    # Expand model inputs from B to B*C.
    expanded_batch = _expand_batch_for_candidates(batch, num_candidates=num_candidates)

    # Expand candidate labels to match B*C.
    # candidate_label_ids: [C, T] -> [B, C, T] -> [B*C, T]
    expanded_labels = (
        candidate_label_ids.unsqueeze(0)
        .expand(batch_size, -1, -1)
        .reshape(batch_size * num_candidates, -1)
    )

    outputs = model(expanded_batch, labels=expanded_labels)
    logits = outputs.logits

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        expanded_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(batch_size * num_candidates, -1)

    valid_mask = (expanded_labels != -100).float()
    seq_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
    seq_loss = seq_loss.view(batch_size, num_candidates)

    best_indices = torch.argmin(seq_loss, dim=1).tolist()
    return [candidate_labels[int(i)] for i in best_indices]


def _is_better(
    current: float,
    best: Optional[float],
    greater_is_better: bool,
    min_delta: float,
) -> bool:
    """Return whether current metric improves over the best seen value."""
    if best is None:
        return True
    if greater_is_better:
        return current > best + min_delta
    return current < best - min_delta


def _save_json(payload: Dict | List[Dict], path: Path) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_stage3_checkpoint(
    model: B345Model,
    generator_tokenizer,
    output_dir: Path,
    ckpt_name: str,
    metrics: Dict,
    run_meta: Dict,
) -> None:
    """
    Save a complete Stage-3 checkpoint.

    `generator.save_pretrained` keeps HuggingFace compatibility for the
    BioBART submodule; `stage3_model_state.pt` preserves the added semantic,
    syntax, and projection parameters.
    """
    ckpt_dir = output_dir / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.generator.save_pretrained(ckpt_dir)
    generator_tokenizer.save_pretrained(ckpt_dir)
    torch.save(model.state_dict(), ckpt_dir / "stage3_model_state.pt")
    _save_json(metrics, ckpt_dir / "metrics.json")
    _save_json(run_meta, ckpt_dir / "run_meta.json")


def run_experiment(cfg: Dict) -> Dict:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    use_cuda = bool(cfg.get("use_cuda", True))
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"[Run] device={device}")

    dtype_name = str(cfg.get("model_dtype", "float32")).lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported model_dtype={dtype_name!r}")
    model_dtype = dtype_map[dtype_name]

    generator_tokenizer = get_generator_tokenizer(cfg["biobart_path"])
    pubmedbert_tokenizer = get_pubmedbert_tokenizer(cfg["pubmedbert_path"])

    train_loader = build_stage3_dataloader(
        file_path=cfg["train_file"],
        generator_tokenizer=generator_tokenizer,
        pubmedbert_tokenizer=pubmedbert_tokenizer,
        dep_type_vocab_path=cfg["dep_type_vocab_path"],
        split="train",
        target_mode=cfg.get("target_mode", "relation_only"),
        dep_view=str(cfg.get("dep_view", "coarse")),
        dep_form=str(cfg.get("dep_form", "none")),
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        max_src_len=int(cfg.get("max_src_len", 256)),
        max_input_len=int(cfg.get("max_input_len", 512)),
        max_target_len=int(cfg.get("max_target_len", 16)),
        max_semantic_len=int(cfg.get("max_semantic_len", 256)),
        seed=seed,
    )
    dev_loader = build_stage3_dataloader(
        file_path=cfg["dev_file"],
        generator_tokenizer=generator_tokenizer,
        pubmedbert_tokenizer=pubmedbert_tokenizer,
        dep_type_vocab_path=cfg["dep_type_vocab_path"],
        split="dev",
        target_mode=cfg.get("target_mode", "relation_only"),
        dep_view=str(cfg.get("dep_view", "coarse")),
        dep_form=str(cfg.get("dep_form", "none")),
        batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 1))),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        max_src_len=int(cfg.get("max_src_len", 256)),
        max_input_len=int(cfg.get("max_input_len", 512)),
        max_target_len=int(cfg.get("max_target_len", 16)),
        max_semantic_len=int(cfg.get("max_semantic_len", 256)),
        seed=seed,
    )

    dep_vocab = json.loads(Path(cfg["dep_type_vocab_path"]).read_text(encoding="utf-8"))
    model = B345Model(
        model_type=str(cfg["model_type"]),
        biobart_path=cfg["biobart_path"],
        pubmedbert_path=cfg["pubmedbert_path"],
        dep_type_vocab_size=len(dep_vocab),
        gcn_hidden_dim=int(cfg.get("gcn_hidden_dim", 256)),
        gcn_layers=int(cfg.get("gcn_layers", 2)),
        gcn_attention_heads=int(cfg.get("gcn_attention_heads", 8)),
        dep_type_dim=int(cfg.get("dep_type_dim", 32)),
        dropout=float(cfg.get("dropout", 0.1)),
        freeze_pubmedbert=bool(cfg.get("freeze_pubmedbert", False)),
        model_dtype=model_dtype,
    )
    model.generator.resize_token_embeddings(len(generator_tokenizer))
    model.to(device)
    print(f"[Run] model_dtype={next(model.parameters()).dtype}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-5)))

    epochs = int(cfg.get("epochs", 1))
    max_train_batches = cfg.get("max_train_batches")
    max_eval_batches = cfg.get("max_eval_batches")
    prediction_mode = str(cfg.get("prediction_mode", "label_rerank")).lower()
    if prediction_mode != "label_rerank":
        raise ValueError("Stage-3 B345 trainer currently supports prediction_mode=label_rerank only.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "model_type": cfg["model_type"],
        "dataset_name": cfg.get("dataset_name", ""),
        "seed": seed,
        "biobart_path": cfg["biobart_path"],
        "pubmedbert_path": cfg["pubmedbert_path"],
        "prediction_mode": prediction_mode,
        "metric_for_best": str(cfg.get("metric_for_best", "mapped_exact")),
        "greater_is_better": bool(cfg.get("greater_is_better", True)),
    }

    metric_for_best = str(cfg.get("metric_for_best", "mapped_exact"))
    greater_is_better = bool(cfg.get("greater_is_better", True))
    min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    patience_value = cfg.get("early_stopping_patience")
    early_stopping_patience = (
        int(patience_value) if patience_value is not None else None
    )
    save_best_model = bool(cfg.get("save_best_model", True))
    save_final_model = bool(cfg.get("save_final_model", True))

    last_metrics: Dict = {}
    best_metrics: Dict = {}
    best_metric_value: Optional[float] = None
    epochs_without_improvement = 0
    metrics_history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for step, raw_batch in enumerate(train_loader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break
            batch = _to_device_batch(raw_batch, device)

            outputs = model(batch, labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss_sum += float(loss.item())
            train_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} steps={train_steps}")

        model.eval()
        eval_loss_sum = 0.0
        eval_steps = 0
        clean_preds: List[str] = []
        clean_golds: List[str] = []

        candidate_labels = sorted(dev_loader.dataset.get_labels())
        candidate_label_ids = build_candidate_label_ids(
            tokenizer=generator_tokenizer,
            candidate_labels=candidate_labels,
            max_target_len=int(cfg.get("max_target_len", 16)),
            device=device,
        )

        with torch.no_grad():
            for step, raw_batch in enumerate(dev_loader):
                if max_eval_batches is not None and step >= int(max_eval_batches):
                    break
                batch = _to_device_batch(raw_batch, device)

                outputs = model(batch, labels=batch["labels"])
                eval_loss_sum += float(outputs.loss.item())
                eval_steps += 1

                preds = predict_labels_by_rerank(
                    model=model,
                    batch=batch,
                    candidate_labels=candidate_labels,
                    candidate_label_ids=candidate_label_ids,
                )
                clean_preds.extend([p.strip() for p in preds])

                labels = batch["labels"].clone()
                labels[labels == -100] = generator_tokenizer.pad_token_id
                clean_golds.extend(
                    [
                        text.strip()
                        for text in generator_tokenizer.batch_decode(
                            labels,
                            skip_special_tokens=True,
                        )
                    ]
                )

        avg_eval_loss = eval_loss_sum / max(eval_steps, 1)
        print(f"[Epoch {epoch}] eval_loss={avg_eval_loss:.6f} steps={eval_steps}")

        label_key_map = _build_label_key_map(candidate_labels)
        mapped_preds = [
            project_prediction_to_label(p, candidate_labels, label_key_map)
            for p in clean_preds
        ]
        n = max(len(clean_preds), 1)
        valid_count = sum(1 for p in clean_preds if p in label_key_map.values())
        invalid_rate = 1.0 - (valid_count / n)
        mapped_valid_count = sum(1 for p in mapped_preds if p is not None)
        mapped_invalid_rate = 1.0 - (mapped_valid_count / n)
        raw_exact = sum(int(p == g) for p, g in zip(clean_preds, clean_golds)) / n
        mapped_exact = sum(int((p or "") == g) for p, g in zip(mapped_preds, clean_golds)) / n

        print(
            f"[Epoch {epoch}] generated={len(clean_preds)} "
            f"valid_label_rate={1.0 - invalid_rate:.4f} invalid_rate={invalid_rate:.4f}"
        )
        print(f"[Epoch {epoch}] prediction_mode={prediction_mode}")
        print(
            f"[Epoch {epoch}] raw_exact={raw_exact:.4f} "
            f"mapped_exact={mapped_exact:.4f} "
            f"mapped_invalid_rate={mapped_invalid_rate:.4f}"
        )
        for i in range(min(3, len(clean_preds))):
            print(f"[Sample {i}] pred={clean_preds[i]!r} mapped={mapped_preds[i]!r} gold={clean_golds[i]!r}")

        last_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "eval_loss": avg_eval_loss,
            "num_predictions": len(clean_preds),
            "valid_label_rate": 1.0 - invalid_rate,
            "invalid_rate": invalid_rate,
            "raw_exact": raw_exact,
            "mapped_exact": mapped_exact,
            "mapped_invalid_rate": mapped_invalid_rate,
        }

        if metric_for_best not in last_metrics:
            raise ValueError(
                f"metric_for_best={metric_for_best!r} is not available in metrics."
            )

        current_metric = float(last_metrics[metric_for_best])
        improved = _is_better(
            current=current_metric,
            best=best_metric_value,
            greater_is_better=greater_is_better,
            min_delta=min_delta,
        )
        history_row = dict(last_metrics)
        history_row["is_best"] = bool(improved)
        metrics_history.append(history_row)
        _save_json(metrics_history, output_dir / "metrics_history.json")

        if improved:
            best_metric_value = current_metric
            best_metrics = dict(last_metrics)
            epochs_without_improvement = 0
            _save_json(best_metrics, output_dir / "best_metrics.json")
            if save_best_model:
                save_stage3_checkpoint(
                    model=model,
                    generator_tokenizer=generator_tokenizer,
                    output_dir=output_dir,
                    ckpt_name="best_model",
                    metrics=best_metrics,
                    run_meta=run_meta,
                )
                checkpoint_note = " checkpoint=best_model"
            else:
                checkpoint_note = ""
            print(
                f"[Epoch {epoch}] best_{metric_for_best}="
                f"{current_metric:.6f}{checkpoint_note}"
            )
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"[EarlyStop] no {metric_for_best} improvement for "
                f"{epochs_without_improvement} epochs"
            )
            break

    model_dir = output_dir / "model"
    if save_final_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.generator.save_pretrained(model_dir)
        generator_tokenizer.save_pretrained(model_dir)
        torch.save(model.state_dict(), model_dir / "stage3_model_state.pt")
    else:
        print("[Run] final model save skipped by config")
    _save_json(run_meta, output_dir / "run_meta.json")
    _save_json(last_metrics, output_dir / "metrics.json")
    if best_metrics:
        _save_json(best_metrics, output_dir / "best_metrics.json")
    print(f"[Run] checkpoint saved to: {output_dir}")
    return last_metrics
