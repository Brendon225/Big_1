"""
Minimal local smoke-run entry for Stage-2 B0 (text-only) baseline.

This script is intentionally small and conservative:
- one backbone only (config-driven, no comparison);
- relation-only target mode;
- few batches / few epochs for local pipeline validation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataloader import build_dataloader, get_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_labels(tokenizer, labels_tensor: torch.Tensor) -> List[str]:
    labels = labels_tensor.clone()
    labels[labels == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(labels, skip_special_tokens=True)


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

    # Handle spaced variants such as "CPR : 3"
    cpr_match = re.search(r"CPR\s*:\s*([0-9]+)", pred, flags=re.IGNORECASE)
    if cpr_match:
        candidate = f"CPR:{cpr_match.group(1)}"
        if candidate in label_key_map.values():
            return candidate

    # Substring fallback for smoke-stage robustness
    for label in sorted(valid_labels, key=len, reverse=True):
        if _label_key(label) in pred_key:
            return label

    return None


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


def predict_labels_by_rerank(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_labels: List[str],
    candidate_label_ids: torch.Tensor,
) -> List[str]:
    """
    Select a legal relation label by minimum token-level NLL.

    This is a smoke-stage fallback to ensure we can evaluate relation-only
    outputs even when free decoding is unstable early in training.
    """
    batch_size = input_ids.size(0)
    num_candidates = len(candidate_labels)
    preds: List[str] = []

    for i in range(batch_size):
        rep_input_ids = input_ids[i : i + 1].repeat(num_candidates, 1)
        rep_attn_mask = attention_mask[i : i + 1].repeat(num_candidates, 1)

        outputs = model(
            input_ids=rep_input_ids,
            attention_mask=rep_attn_mask,
            labels=candidate_label_ids,
        )
        logits = outputs.logits

        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            candidate_label_ids.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(num_candidates, -1)

        valid_mask = (candidate_label_ids != -100).float()
        seq_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

        best_idx = int(torch.argmin(seq_loss).item())
        preds.append(candidate_labels[best_idx])

    return preds


def run(cfg: Dict) -> Dict:
    set_seed(int(cfg.get("seed", 42)))

    use_cuda = bool(cfg.get("use_cuda", True))
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"[Run] device={device}")

    model_name = cfg["model_name"]
    target_mode = cfg["target_mode"]
    use_dep = bool(cfg.get("use_dep", False))
    dep_view = str(cfg.get("dep_view", "raw"))
    dep_form = str(cfg.get("dep_form", "none"))
    if not use_dep:
        dep_form = "none"
    model_dtype_name = str(cfg.get("model_dtype", "float32")).lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if model_dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported model_dtype={model_dtype_name!r}. "
            f"Expected one of {sorted(dtype_map.keys())}."
        )
    model_dtype = dtype_map[model_dtype_name]

    tokenizer = get_tokenizer(model_name=model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=model_dtype)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(f"[Run] model_dtype={next(model.parameters()).dtype}")

    train_loader = build_dataloader(
        file_path=cfg["train_file"],
        tokenizer=tokenizer,
        split="train",
        use_dep=use_dep,
        dep_view=dep_view,
        dep_form=dep_form,
        target_mode=target_mode,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        max_src_len=int(cfg.get("max_src_len", 256)),
        max_input_len=int(cfg.get("max_input_len", 512)),
        max_target_len=int(cfg.get("max_target_len", 16)),
        seed=int(cfg.get("seed", 42)),
    )

    dev_loader = build_dataloader(
        file_path=cfg["dev_file"],
        tokenizer=tokenizer,
        split="dev",
        use_dep=use_dep,
        dep_view=dep_view,
        dep_form=dep_form,
        target_mode=target_mode,
        batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 2))),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        max_src_len=int(cfg.get("max_src_len", 256)),
        max_input_len=int(cfg.get("max_input_len", 512)),
        max_target_len=int(cfg.get("max_target_len", 16)),
        seed=int(cfg.get("seed", 42)),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-5)))

    epochs = int(cfg.get("epochs", 1))
    max_train_batches = cfg.get("max_train_batches")
    max_eval_batches = cfg.get("max_eval_batches")
    max_new_tokens = int(cfg.get("max_new_tokens", 8))
    prediction_mode = str(cfg.get("prediction_mode", "generate")).lower()
    if prediction_mode not in {"generate", "label_rerank"}:
        raise ValueError(
            f"Unsupported prediction_mode={prediction_mode!r}. "
            "Expected one of ['generate', 'label_rerank']."
        )

    last_metrics: Dict = {}
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for step, batch in enumerate(train_loader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
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
        generated_texts: List[str] = []
        gold_texts: List[str] = []
        rerank_preds: List[str] = []
        candidate_labels = sorted(dev_loader.dataset.get_labels())
        candidate_label_ids = build_candidate_label_ids(
            tokenizer=tokenizer,
            candidate_labels=candidate_labels,
            max_target_len=int(cfg.get("max_target_len", 16)),
            device=device,
        )

        with torch.no_grad():
            for step, batch in enumerate(dev_loader):
                if max_eval_batches is not None and step >= int(max_eval_batches):
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                eval_loss_sum += float(outputs.loss.item())
                eval_steps += 1

                if prediction_mode == "generate":
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                    )
                    generated_texts.extend(
                        tokenizer.batch_decode(generated, skip_special_tokens=True)
                    )
                else:
                    batch_preds = predict_labels_by_rerank(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        candidate_labels=candidate_labels,
                        candidate_label_ids=candidate_label_ids,
                    )
                    rerank_preds.extend(batch_preds)
                gold_texts.extend(decode_labels(tokenizer, labels.cpu()))

        avg_eval_loss = eval_loss_sum / max(eval_steps, 1)
        print(f"[Epoch {epoch}] eval_loss={avg_eval_loss:.6f} steps={eval_steps}")

        valid_labels = candidate_labels
        label_key_map = _build_label_key_map(valid_labels)
        if prediction_mode == "generate":
            clean_preds = [text.strip() for text in generated_texts]
        else:
            clean_preds = [text.strip() for text in rerank_preds]
        clean_golds = [text.strip() for text in gold_texts]
        mapped_preds = [
            project_prediction_to_label(text, valid_labels, label_key_map)
            for text in clean_preds
        ]

        n = max(len(clean_preds), 1)
        valid_count = sum(1 for text in clean_preds if text in label_key_map.values())
        invalid_rate = 1.0 - (valid_count / n)
        mapped_valid_count = sum(1 for text in mapped_preds if text is not None)
        mapped_invalid_rate = 1.0 - (mapped_valid_count / n)
        raw_exact = sum(int(p == g) for p, g in zip(clean_preds, clean_golds)) / n
        mapped_exact = (
            sum(int((p or "") == g) for p, g in zip(mapped_preds, clean_golds)) / n
        )
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

        sample_n = min(3, len(clean_preds))
        for i in range(sample_n):
            print(
                f"[Sample {i}] pred={clean_preds[i]!r}  "
                f"mapped={mapped_preds[i]!r}  gold={clean_golds[i]!r}"
            )
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

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    run_meta = {
        "model_name": model_name,
        "target_mode": target_mode,
        "dataset_name": cfg.get("dataset_name", ""),
        "seed": int(cfg.get("seed", 42)),
        "use_dep": use_dep,
        "dep_view": dep_view,
        "dep_form": dep_form,
        "prediction_mode": prediction_mode,
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(last_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Run] checkpoint saved to: {output_dir}")
    return last_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 B0 local smoke run")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_local_smoke.json",
        help="Path to local smoke config JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for config seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for config output_dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    run(config)
