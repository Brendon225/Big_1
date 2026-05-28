"""
Stage-4 F1 evaluator for paper-ready test metrics.

The script loads an existing Stage-2 or Stage-3 checkpoint, runs label-rerank
evaluation on dev/test, saves per-sample predictions, and reports RE metrics
with NO_RELATION excluded from the main P/R/F1 scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.baseline.run_smoke import (  # noqa: E402
    build_candidate_label_ids as build_baseline_candidate_label_ids,
    predict_labels_by_rerank as predict_baseline_labels_by_rerank,
)
from src.data.dataloader import build_dataloader  # noqa: E402
from src.data.dataset import SPECIAL_TOKENS  # noqa: E402
from src.models.b345_model import B345Model  # noqa: E402
from src.stage3.trainer_b345 import (  # noqa: E402
    _to_device_batch,
    build_candidate_label_ids as build_stage3_candidate_label_ids,
    build_stage3_dataloader,
    predict_labels_by_rerank as predict_stage3_labels_by_rerank,
    project_prediction_to_label,
    set_seed,
)


NEGATIVE_LABELS = {"NO_RELATION", "DDI-FALSE", "DDI-false"}


def load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload: Dict | List[Dict], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def infer_eval_file(cfg: Dict, split: str) -> str:
    if split == "dev":
        return str(cfg["dev_file"])
    if split == "train":
        return str(cfg["train_file"])

    dev_file = str(cfg["dev_file"])
    candidate = re.sub(r"_(dev|train|test)\.json$", f"_{split}.json", dev_file)
    if candidate == dev_file:
        raise ValueError(
            f"Could not infer {split!r} file from dev_file={dev_file!r}; "
            "pass --eval_file explicitly."
        )
    return candidate


def default_checkpoint_subdir(model_family: str) -> str:
    return "best_model" if model_family == "stage3" else "model"


def resolve_model_dir(checkpoint_dir: str | Path, model_family: str) -> Path:
    checkpoint = Path(checkpoint_dir)
    if (checkpoint / "stage3_model_state.pt").exists() or (
        checkpoint / "model.safetensors"
    ).exists():
        return checkpoint
    subdir = checkpoint / default_checkpoint_subdir(model_family)
    if not subdir.exists():
        raise FileNotFoundError(f"Cannot find checkpoint model directory: {subdir}")
    return subdir


def load_tokenizer(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer


def dtype_from_config(cfg: Dict) -> torch.dtype:
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
    return dtype_map[dtype_name]


def build_output_prefix(
    output_dir: Optional[str],
    cfg: Dict,
    model_family: str,
    checkpoint_dir: str | Path,
    split: str,
    model_alias: Optional[str] = None,
) -> Path:
    dataset = str(cfg.get("dataset_name", "dataset")).lower()
    model_name = str(model_alias or cfg.get("model_type") or Path(checkpoint_dir).name)
    seed = str(cfg.get("seed", "seed"))
    base = Path(output_dir) if output_dir else Path("outputs/stage4_predictions") / dataset
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{model_family}_{model_name}_{dataset}_seed{seed}_{split}"


def is_positive_label(label: Optional[str]) -> bool:
    return bool(label) and str(label) not in NEGATIVE_LABELS


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_metrics(
    rows: List[Dict],
    valid_labels: List[str],
) -> Tuple[Dict, Dict[str, Dict[str, float]], Dict[str, Counter]]:
    positive_labels = [label for label in valid_labels if is_positive_label(label)]
    per_class: Dict[str, Dict[str, float]] = {}

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        gold = row["gold_label"]
        pred = row["mapped_pred"] or "__INVALID__"
        confusion[gold][pred] += 1

    for label in positive_labels:
        tp = sum(1 for row in rows if row["gold_label"] == label and row["mapped_pred"] == label)
        fp = sum(1 for row in rows if row["gold_label"] != label and row["mapped_pred"] == label)
        fn = sum(1 for row in rows if row["gold_label"] == label and row["mapped_pred"] != label)
        support = sum(1 for row in rows if row["gold_label"] == label)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    precision = safe_div(micro_tp, micro_tp + micro_fp)
    recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2 * precision * recall, precision + recall)
    supported_classes = [v for v in per_class.values() if v["support"] > 0]
    macro_f1 = safe_div(sum(v["f1"] for v in supported_classes), len(supported_classes))
    macro_f1_all_positive_labels = safe_div(
        sum(v["f1"] for v in per_class.values()),
        len(per_class),
    )

    n = max(len(rows), 1)
    raw_valid = sum(1 for row in rows if row["raw_pred"] in valid_labels)
    mapped_valid = sum(1 for row in rows if row["mapped_pred"] in valid_labels)
    raw_exact = sum(1 for row in rows if row["raw_pred"] == row["gold_label"])
    mapped_exact = sum(1 for row in rows if row["mapped_pred"] == row["gold_label"])

    metrics = {
        "num_predictions": len(rows),
        "num_positive_labels": len(positive_labels),
        "positive_labels": positive_labels,
        "precision_excluding_no_relation": precision,
        "recall_excluding_no_relation": recall,
        "micro_f1_excluding_no_relation": micro_f1,
        "macro_f1_excluding_no_relation": macro_f1,
        "macro_f1_all_positive_labels": macro_f1_all_positive_labels,
        "raw_exact": safe_div(raw_exact, n),
        "mapped_exact": safe_div(mapped_exact, n),
        "invalid_rate": 1.0 - safe_div(raw_valid, n),
        "mapped_invalid_rate": 1.0 - safe_div(mapped_valid, n),
        "micro_tp": micro_tp,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
    }
    return metrics, per_class, confusion


def write_jsonl(rows: Iterable[Dict], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_confusion_csv(confusion: Dict[str, Counter], labels: List[str], path: str | Path) -> None:
    all_preds = sorted({pred for counter in confusion.values() for pred in counter})
    columns = sorted(set(labels) | set(all_preds))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_label", *columns])
        for gold in columns:
            writer.writerow([gold, *[confusion.get(gold, Counter()).get(pred, 0) for pred in columns]])


def decode_gold_labels(tokenizer, labels: torch.Tensor) -> List[str]:
    clean = labels.detach().cpu().clone()
    clean[clean == -100] = tokenizer.pad_token_id
    return [text.strip() for text in tokenizer.batch_decode(clean, skip_special_tokens=True)]


def build_stage3_model(cfg: Dict, model_dir: Path, tokenizer, device: torch.device) -> B345Model:
    dep_vocab = load_json(cfg["dep_type_vocab_path"])
    model = B345Model(
        model_type=str(cfg["model_type"]),
        biobart_path=str(model_dir),
        pubmedbert_path=cfg["pubmedbert_path"],
        dep_type_vocab_size=len(dep_vocab),
        gcn_hidden_dim=int(cfg.get("gcn_hidden_dim", 256)),
        gcn_layers=int(cfg.get("gcn_layers", 2)),
        gcn_attention_heads=int(cfg.get("gcn_attention_heads", 8)),
        dep_type_dim=int(cfg.get("dep_type_dim", 32)),
        dropout=float(cfg.get("dropout", 0.1)),
        freeze_pubmedbert=bool(cfg.get("freeze_pubmedbert", False)),
        model_dtype=dtype_from_config(cfg),
    )
    model.generator.resize_token_embeddings(len(tokenizer))
    state_path = model_dir / "stage3_model_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        load_result = model.load_state_dict(state, strict=False)
        model_type = str(cfg["model_type"]).lower()
        allowed_missing_prefixes = {
            "b3": ("sem_proj.", "dual_proj.", "gate_proj."),
            "b4": ("syntax_view.", "syn_proj.", "dual_proj.", "gate_proj."),
            "b5": ("gate_proj.",),
            "b7": ("dual_proj.",),
        }.get(model_type, ())
        unexpected = list(load_result.unexpected_keys)
        missing = list(load_result.missing_keys)
        disallowed_missing = [
            key
            for key in missing
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "Incompatible Stage-3 checkpoint state. "
                f"unexpected_keys={unexpected}; "
                f"disallowed_missing_keys={disallowed_missing}; "
                f"allowed_missing_keys={missing}"
            )
        if missing:
            print(
                "[Stage4] tolerated missing unused checkpoint keys: "
                + ", ".join(missing)
            )
    model.to(device)
    model.eval()
    return model


def evaluate_stage3(
    cfg: Dict,
    model_dir: Path,
    eval_file: str,
    split: str,
    device: torch.device,
    max_eval_batches: Optional[int],
    eval_batch_size: Optional[int],
) -> Tuple[List[Dict], Dict]:
    tokenizer = load_tokenizer(model_dir)
    pubmedbert_tokenizer = AutoTokenizer.from_pretrained(cfg["pubmedbert_path"])
    model = build_stage3_model(cfg, model_dir, tokenizer, device)

    loader = build_stage3_dataloader(
        file_path=eval_file,
        generator_tokenizer=tokenizer,
        pubmedbert_tokenizer=pubmedbert_tokenizer,
        dep_type_vocab_path=cfg["dep_type_vocab_path"],
        split=split,
        target_mode=cfg.get("target_mode", "relation_only"),
        dep_view=str(cfg.get("dep_view", "coarse")),
        dep_form=str(cfg.get("dep_form", "none")),
        batch_size=int(eval_batch_size or cfg.get("eval_batch_size", cfg.get("batch_size", 1))),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        max_src_len=int(cfg.get("max_src_len", 256)),
        max_input_len=int(cfg.get("max_input_len", 512)),
        max_target_len=int(cfg.get("max_target_len", 16)),
        max_semantic_len=int(cfg.get("max_semantic_len", 256)),
        seed=int(cfg.get("seed", 42)),
    )

    valid_labels = sorted(loader.dataset.get_labels())
    label_key_map = {"".join(label.strip().upper().split()): label for label in valid_labels}
    candidate_label_ids = build_stage3_candidate_label_ids(
        tokenizer=tokenizer,
        candidate_labels=valid_labels,
        max_target_len=int(cfg.get("max_target_len", 16)),
        device=device,
    )

    rows: List[Dict] = []
    eval_loss_sum = 0.0
    eval_steps = 0
    gate_sem_sum = 0.0
    gate_syn_sum = 0.0
    gate_steps = 0

    with torch.no_grad():
        for step, raw_batch in enumerate(loader):
            if max_eval_batches is not None and step >= max_eval_batches:
                break
            batch = _to_device_batch(raw_batch, device)
            outputs = model(batch, labels=batch["labels"])
            eval_loss_sum += float(outputs.loss.item())
            eval_steps += 1

            aux = getattr(model, "last_aux_metrics", {})
            if "gate_semantic_mean" in aux and "gate_syntax_mean" in aux:
                gate_sem_sum += float(aux["gate_semantic_mean"])
                gate_syn_sum += float(aux["gate_syntax_mean"])
                gate_steps += 1

            preds = predict_stage3_labels_by_rerank(
                model=model,
                batch=batch,
                candidate_labels=valid_labels,
                candidate_label_ids=candidate_label_ids,
            )
            golds = decode_gold_labels(tokenizer, batch["labels"])
            for meta, raw_pred, gold in zip(raw_batch["meta"], preds, golds):
                mapped = project_prediction_to_label(raw_pred, valid_labels, label_key_map)
                rows.append(
                    {
                        "id": meta.get("id", ""),
                        "dataset": meta.get("dataset", cfg.get("dataset_name", "")),
                        "split": split,
                        "gold_label": gold,
                        "raw_pred": raw_pred.strip(),
                        "mapped_pred": mapped,
                        "correct": bool(mapped == gold),
                        "e1_text": meta.get("e1_text", ""),
                        "e2_text": meta.get("e2_text", ""),
                        "dep_view_used": meta.get("dep_view_used"),
                        "dep_form_used": meta.get("dep_form_used"),
                        "truncated": meta.get("truncated", False),
                    }
                )

    aux_metrics = {
        "eval_loss": eval_loss_sum / max(eval_steps, 1),
        "eval_steps": eval_steps,
    }
    if gate_steps:
        aux_metrics["gate_semantic_mean"] = gate_sem_sum / gate_steps
        aux_metrics["gate_syntax_mean"] = gate_syn_sum / gate_steps
    return rows, {"valid_labels": valid_labels, **aux_metrics}


def evaluate_baseline(
    cfg: Dict,
    model_dir: Path,
    eval_file: str,
    split: str,
    device: torch.device,
    max_eval_batches: Optional[int],
    eval_batch_size: Optional[int],
) -> Tuple[List[Dict], Dict]:
    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=dtype_from_config(cfg))
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    use_dep = bool(cfg.get("use_dep", False))
    dep_form = str(cfg.get("dep_form", "none")) if use_dep else "none"
    loader = build_dataloader(
        file_path=eval_file,
        tokenizer=tokenizer,
        split=split,
        use_dep=use_dep,
        dep_view=str(cfg.get("dep_view", "raw")),
        dep_form=dep_form,
        target_mode=cfg.get("target_mode", "relation_only"),
        batch_size=int(eval_batch_size or cfg.get("eval_batch_size", cfg.get("batch_size", 1))),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        max_src_len=int(cfg.get("max_src_len", 256)),
        max_input_len=int(cfg.get("max_input_len", 512)),
        max_target_len=int(cfg.get("max_target_len", 16)),
        max_dep_arcs=cfg.get("max_dep_arcs"),
        seed=int(cfg.get("seed", 42)),
    )

    valid_labels = sorted(loader.dataset.get_labels())
    label_key_map = {"".join(label.strip().upper().split()): label for label in valid_labels}
    candidate_label_ids = build_baseline_candidate_label_ids(
        tokenizer=tokenizer,
        candidate_labels=valid_labels,
        max_target_len=int(cfg.get("max_target_len", 16)),
        device=device,
    )

    rows: List[Dict] = []
    eval_loss_sum = 0.0
    eval_steps = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_eval_batches is not None and step >= max_eval_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss_sum += float(outputs.loss.item())
            eval_steps += 1

            preds = predict_baseline_labels_by_rerank(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                candidate_labels=valid_labels,
                candidate_label_ids=candidate_label_ids,
            )
            golds = decode_gold_labels(tokenizer, labels)
            for meta, raw_pred, gold in zip(batch["meta"], preds, golds):
                mapped = project_prediction_to_label(raw_pred, valid_labels, label_key_map)
                rows.append(
                    {
                        "id": meta.get("id", ""),
                        "dataset": meta.get("dataset", cfg.get("dataset_name", "")),
                        "split": split,
                        "gold_label": gold,
                        "raw_pred": raw_pred.strip(),
                        "mapped_pred": mapped,
                        "correct": bool(mapped == gold),
                        "e1_text": meta.get("e1_text", ""),
                        "e2_text": meta.get("e2_text", ""),
                        "dep_view_used": meta.get("dep_view_used"),
                        "dep_form_used": dep_form,
                        "truncated": meta.get("truncated", False),
                    }
                )

    return rows, {
        "valid_labels": valid_labels,
        "eval_loss": eval_loss_sum / max(eval_steps, 1),
        "eval_steps": eval_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-4 paper F1 evaluator")
    parser.add_argument("--config", required=True, help="Training config JSON.")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint root directory.")
    parser.add_argument(
        "--model_family",
        choices=["stage3", "baseline"],
        default="stage3",
        help="Use 'stage3' for B4/B5/B7 and 'baseline' for Stage2 B2-coarse.",
    )
    parser.add_argument("--split", choices=["dev", "test", "train"], default="test")
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_alias", default=None, help="Short model name for output files.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed stored in config.")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    set_seed(int(cfg.get("seed", 42)))

    use_cuda = bool(cfg.get("use_cuda", True)) and not args.cpu
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    eval_file = args.eval_file or infer_eval_file(cfg, args.split)
    model_dir = resolve_model_dir(args.checkpoint_dir, args.model_family)
    output_prefix = build_output_prefix(
        output_dir=args.output_dir,
        cfg=cfg,
        model_family=args.model_family,
        checkpoint_dir=args.checkpoint_dir,
        split=args.split,
        model_alias=args.model_alias,
    )

    print(f"[Stage4] device={device}")
    print(f"[Stage4] config={args.config}")
    print(f"[Stage4] checkpoint_model_dir={model_dir}")
    print(f"[Stage4] eval_file={eval_file}")
    print(f"[Stage4] output_prefix={output_prefix}")

    if args.model_family == "stage3":
        rows, aux = evaluate_stage3(
            cfg=cfg,
            model_dir=model_dir,
            eval_file=eval_file,
            split=args.split,
            device=device,
            max_eval_batches=args.max_eval_batches,
            eval_batch_size=args.eval_batch_size,
        )
    else:
        rows, aux = evaluate_baseline(
            cfg=cfg,
            model_dir=model_dir,
            eval_file=eval_file,
            split=args.split,
            device=device,
            max_eval_batches=args.max_eval_batches,
            eval_batch_size=args.eval_batch_size,
        )

    metrics, per_class, confusion = compute_metrics(rows, aux["valid_labels"])
    metrics.update(
        {
            "model_family": args.model_family,
            "model_type": args.model_alias or cfg.get("model_type", Path(args.checkpoint_dir).name),
            "dataset_name": cfg.get("dataset_name", ""),
            "seed": cfg.get("seed"),
            "split": args.split,
            "eval_file": eval_file,
            "checkpoint_dir": str(args.checkpoint_dir),
            **{k: v for k, v in aux.items() if k != "valid_labels"},
        }
    )

    write_jsonl(rows, f"{output_prefix}_predictions.jsonl")
    save_json(metrics, f"{output_prefix}_metrics.json")
    save_json(per_class, f"{output_prefix}_per_class.json")
    write_confusion_csv(confusion, aux["valid_labels"], f"{output_prefix}_confusion.csv")

    print(
        "[Stage4] "
        f"P={metrics['precision_excluding_no_relation']:.4f} "
        f"R={metrics['recall_excluding_no_relation']:.4f} "
        f"Micro-F1={metrics['micro_f1_excluding_no_relation']:.4f} "
        f"Macro-F1={metrics['macro_f1_excluding_no_relation']:.4f} "
        f"mapped_exact={metrics['mapped_exact']:.4f} "
        f"invalid_rate={metrics['invalid_rate']:.4f}"
    )
    print(f"[Stage4] saved predictions/metrics to {output_prefix}_*")


if __name__ == "__main__":
    main()
