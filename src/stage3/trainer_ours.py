"""
Training utility for Stage-3 Ours (EA-GMIB).

This trainer reuses the verified Stage-3 graph data path and label-rerank
evaluation, while adding GM-IB specific losses and compression statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.ea_gmib import EAGMIBModel  # noqa: E402
from src.stage3.trainer_b345 import (  # noqa: E402
    _build_label_key_map,
    _expand_batch_for_candidates,
    _is_better,
    _save_json,
    _to_device_batch,
    build_candidate_label_ids,
    build_stage3_dataloader,
    get_generator_tokenizer,
    get_pubmedbert_tokenizer,
    load_config,
    project_prediction_to_label,
    set_seed,
)


def predict_labels_by_rerank(
    model: EAGMIBModel,
    batch: Dict[str, torch.Tensor],
    candidate_labels: List[str],
    candidate_label_ids: torch.Tensor,
) -> List[str]:
    batch_size = int(batch["input_ids"].size(0))
    num_candidates = len(candidate_labels)
    if num_candidates == 0:
        return []

    expanded_batch = _expand_batch_for_candidates(batch, num_candidates=num_candidates)
    expanded_labels = (
        candidate_label_ids.unsqueeze(0)
        .expand(batch_size, -1, -1)
        .reshape(batch_size * num_candidates, -1)
    )

    outputs = model(expanded_batch, labels=expanded_labels)
    logits = outputs["logits"]
    if logits is None:
        raise RuntimeError("EA-GMIB forward did not return logits for label rerank.")

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


def save_ours_checkpoint(
    model: EAGMIBModel,
    generator_tokenizer,
    output_dir: Path,
    ckpt_name: str,
    metrics: Dict,
    run_meta: Dict,
) -> None:
    ckpt_dir = output_dir / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.generator.save_pretrained(ckpt_dir)
    generator_tokenizer.save_pretrained(ckpt_dir)
    torch.save(model.state_dict(), ckpt_dir / "ea_gmib_model_state.pt")
    _save_json(metrics, ckpt_dir / "metrics.json")
    _save_json(run_meta, ckpt_dir / "run_meta.json")


def _tensor_item(value: Optional[torch.Tensor]) -> float:
    if value is None:
        return 0.0
    return float(value.detach().cpu().item())


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
    model = EAGMIBModel(
        biobart_path=cfg["biobart_path"],
        pubmedbert_path=cfg["pubmedbert_path"],
        dep_type_vocab_size=len(dep_vocab),
        gcn_hidden_dim=int(cfg.get("gcn_hidden_dim", 256)),
        gcn_layers=int(cfg.get("gcn_layers", 2)),
        gcn_attention_heads=int(cfg.get("gcn_attention_heads", 8)),
        dep_type_dim=int(cfg.get("dep_type_dim", 32)),
        gmib_hidden_dim=int(cfg.get("gmib_hidden_dim", 256)),
        beta=float(cfg.get("beta", 1e-6)),
        tau_init=float(cfg.get("tau_init", 1.0)),
        tau_min=float(cfg.get("tau_min", 0.1)),
        tau_anneal_rate=float(cfg.get("tau_anneal_rate", 0.95)),
        selection_threshold=float(cfg.get("selection_threshold", 0.5)),
        compression_loss_type=str(cfg.get("compression_loss_type", "l1")),
        dropout=float(cfg.get("dropout", 0.1)),
        freeze_pubmedbert=bool(cfg.get("freeze_pubmedbert", False)),
        model_dtype=model_dtype,
    )
    model.generator.resize_token_embeddings(len(generator_tokenizer))
    model.to(device)
    print(f"[Run] model_dtype={next(model.parameters()).dtype}")
    print(
        f"[Run] beta={model.beta:g} tau={model.current_tau:.4f} "
        f"threshold={model.gmib.selection_threshold:.3f}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-5)))

    epochs = int(cfg.get("epochs", 1))
    max_train_batches = cfg.get("max_train_batches")
    max_eval_batches = cfg.get("max_eval_batches")
    prediction_mode = str(cfg.get("prediction_mode", "label_rerank")).lower()
    if prediction_mode != "label_rerank":
        raise ValueError("EA-GMIB trainer currently supports prediction_mode=label_rerank only.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "model_type": "ours",
        "dataset_name": cfg.get("dataset_name", ""),
        "seed": seed,
        "biobart_path": cfg["biobart_path"],
        "pubmedbert_path": cfg["pubmedbert_path"],
        "prediction_mode": prediction_mode,
        "metric_for_best": str(cfg.get("metric_for_best", "mapped_exact")),
        "greater_is_better": bool(cfg.get("greater_is_better", True)),
        "beta": float(cfg.get("beta", 1e-6)),
        "tau_init": float(cfg.get("tau_init", 1.0)),
        "tau_min": float(cfg.get("tau_min", 0.1)),
        "tau_anneal_rate": float(cfg.get("tau_anneal_rate", 0.95)),
        "selection_threshold": float(cfg.get("selection_threshold", 0.5)),
        "compression_loss_type": str(cfg.get("compression_loss_type", "l1")),
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
        tau_used = model.current_tau
        model.train()
        train_loss_sum = 0.0
        train_gen_loss_sum = 0.0
        train_compress_loss_sum = 0.0
        train_retained_sum = 0.0
        train_compression_ratio_sum = 0.0
        train_steps = 0

        for step, raw_batch in enumerate(train_loader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break
            batch = _to_device_batch(raw_batch, device)

            outputs = model(batch, labels=batch["labels"])
            loss = outputs["loss"]
            if loss is None:
                raise RuntimeError("EA-GMIB did not return loss during training.")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss_sum += _tensor_item(outputs["loss"])
            train_gen_loss_sum += _tensor_item(outputs["gen_loss"])
            train_compress_loss_sum += _tensor_item(outputs["compress_loss"])
            train_retained_sum += _tensor_item(outputs["avg_retained_arcs"])
            train_compression_ratio_sum += _tensor_item(outputs["compression_ratio"])
            train_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)
        avg_train_gen_loss = train_gen_loss_sum / max(train_steps, 1)
        avg_train_compress_loss = train_compress_loss_sum / max(train_steps, 1)
        avg_train_retained = train_retained_sum / max(train_steps, 1)
        avg_train_compression_ratio = train_compression_ratio_sum / max(train_steps, 1)
        print(
            f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} "
            f"gen_loss={avg_train_gen_loss:.6f} "
            f"compress_loss={avg_train_compress_loss:.6f} "
            f"avg_retained_arcs={avg_train_retained:.4f} "
            f"compression_ratio={avg_train_compression_ratio:.4f} "
            f"tau={tau_used:.4f} steps={train_steps}"
        )

        model.eval()
        eval_loss_sum = 0.0
        eval_gen_loss_sum = 0.0
        eval_compress_loss_sum = 0.0
        eval_retained_sum = 0.0
        eval_compression_ratio_sum = 0.0
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
                eval_loss_sum += _tensor_item(outputs["loss"])
                eval_gen_loss_sum += _tensor_item(outputs["gen_loss"])
                eval_compress_loss_sum += _tensor_item(outputs["compress_loss"])
                eval_retained_sum += _tensor_item(outputs["avg_retained_arcs"])
                eval_compression_ratio_sum += _tensor_item(outputs["compression_ratio"])
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
        avg_eval_gen_loss = eval_gen_loss_sum / max(eval_steps, 1)
        avg_eval_compress_loss = eval_compress_loss_sum / max(eval_steps, 1)
        avg_eval_retained = eval_retained_sum / max(eval_steps, 1)
        avg_eval_compression_ratio = eval_compression_ratio_sum / max(eval_steps, 1)
        print(
            f"[Epoch {epoch}] eval_loss={avg_eval_loss:.6f} "
            f"gen_loss={avg_eval_gen_loss:.6f} "
            f"compress_loss={avg_eval_compress_loss:.6f} "
            f"avg_retained_arcs={avg_eval_retained:.4f} "
            f"compression_ratio={avg_eval_compression_ratio:.4f} steps={eval_steps}"
        )

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
            "train_gen_loss": avg_train_gen_loss,
            "train_compress_loss": avg_train_compress_loss,
            "train_avg_retained_arcs": avg_train_retained,
            "train_compression_ratio": avg_train_compression_ratio,
            "eval_loss": avg_eval_loss,
            "eval_gen_loss": avg_eval_gen_loss,
            "eval_compress_loss": avg_eval_compress_loss,
            "eval_avg_retained_arcs": avg_eval_retained,
            "eval_compression_ratio": avg_eval_compression_ratio,
            "num_predictions": len(clean_preds),
            "valid_label_rate": 1.0 - invalid_rate,
            "invalid_rate": invalid_rate,
            "raw_exact": raw_exact,
            "mapped_exact": mapped_exact,
            "mapped_invalid_rate": mapped_invalid_rate,
            "tau": tau_used,
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
                save_ours_checkpoint(
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

        next_tau = model.anneal_temperature()
        print(f"[Epoch {epoch}] tau_annealed_to={next_tau:.4f}")

    model_dir = output_dir / "model"
    if save_final_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.generator.save_pretrained(model_dir)
        generator_tokenizer.save_pretrained(model_dir)
        torch.save(model.state_dict(), model_dir / "ea_gmib_model_state.pt")
    else:
        print("[Run] final model save skipped by config")
    _save_json(run_meta, output_dir / "run_meta.json")
    _save_json(last_metrics, output_dir / "metrics.json")
    if best_metrics:
        _save_json(best_metrics, output_dir / "best_metrics.json")
    print(f"[Run] checkpoint saved to: {output_dir}")
    return last_metrics
