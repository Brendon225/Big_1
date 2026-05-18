"""
Threshold sweep analysis for trained Stage-3 Ours (EA-GMIB) checkpoints.

This script does not retrain the model. It reloads an existing `best_model`,
evaluates dev label-rerank under multiple hard-selection thresholds, and writes
a CSV with accuracy, compression, and p_ij distribution diagnostics.

Example:
  venv/Scripts/python.exe src/stage3/analyze_ours_threshold_sweep.py \
    --config configs/stage3_full_ours_chemprotsent_seed42.json \
    --checkpoint_dir checkpoints/stage3_full_ours_chemprotsent_seed42/best_model \
    --output_csv checkpoints/stage3_ours_threshold_sweep_chemprotsent_seed42.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.ea_gmib import EAGMIBModel  # noqa: E402
from src.stage3.trainer_b345 import (  # noqa: E402
    _build_label_key_map,
    _to_device_batch,
    build_candidate_label_ids,
    build_stage3_dataloader,
    get_pubmedbert_tokenizer,
    load_config,
    project_prediction_to_label,
    set_seed,
)
from src.stage3.trainer_ours import predict_labels_by_rerank  # noqa: E402


DEFAULT_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.475"


def _parse_thresholds(raw: str) -> List[float]:
    thresholds: List[float] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def _dtype_from_config(cfg: Dict) -> torch.dtype:
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


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_model(
    cfg: Dict,
    generator_tokenizer,
    checkpoint_dir: Path,
    device: torch.device,
) -> EAGMIBModel:
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
        target_compression_ratio=cfg.get("target_compression_ratio"),
        target_ratio_loss_weight=float(cfg.get("target_ratio_loss_weight", 1.0)),
        readout_mode=str(cfg.get("readout_mode", "degree_pool")),
        dropout=float(cfg.get("dropout", 0.1)),
        freeze_pubmedbert=bool(cfg.get("freeze_pubmedbert", False)),
        model_dtype=_dtype_from_config(cfg),
    )
    model.generator.resize_token_embeddings(len(generator_tokenizer))

    state_path = checkpoint_dir / "ea_gmib_model_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing checkpoint state file: {state_path}")
    state_dict = _load_state_dict(state_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint state mismatch. "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    model.to(device)
    model.eval()
    return model


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float().cpu(), q).item())


def _collect_prob_stats(
    outputs: Dict[str, Optional[torch.Tensor]],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    p_ij = outputs["p_ij"]
    if p_ij is None:
        return {
            "p_mean": 0.0,
            "p_min": 0.0,
            "p_p10": 0.0,
            "p_p25": 0.0,
            "p_p50": 0.0,
            "p_p75": 0.0,
            "p_p90": 0.0,
            "p_max": 0.0,
        }
    edge_mask = (
        (batch["adj_matrix"] > 0)
        & batch["node_mask"].unsqueeze(1)
        & batch["node_mask"].unsqueeze(2)
    )
    valid_probs = p_ij[edge_mask]
    if valid_probs.numel() == 0:
        return {
            "p_mean": 0.0,
            "p_min": 0.0,
            "p_p10": 0.0,
            "p_p25": 0.0,
            "p_p50": 0.0,
            "p_p75": 0.0,
            "p_p90": 0.0,
            "p_max": 0.0,
        }
    probs = valid_probs.float().detach().cpu()
    return {
        "p_mean": float(probs.mean().item()),
        "p_min": float(probs.min().item()),
        "p_p10": _quantile(probs, 0.10),
        "p_p25": _quantile(probs, 0.25),
        "p_p50": _quantile(probs, 0.50),
        "p_p75": _quantile(probs, 0.75),
        "p_p90": _quantile(probs, 0.90),
        "p_max": float(probs.max().item()),
    }


def _mean_dict(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def evaluate_threshold(
    model: EAGMIBModel,
    dev_loader,
    generator_tokenizer,
    device: torch.device,
    threshold: float,
    max_eval_batches: Optional[int],
    max_target_len: int,
) -> Dict[str, float | int]:
    model.eval()
    model.gmib.selection_threshold = float(threshold)

    candidate_labels = sorted(dev_loader.dataset.get_labels())
    candidate_label_ids = build_candidate_label_ids(
        tokenizer=generator_tokenizer,
        candidate_labels=candidate_labels,
        max_target_len=max_target_len,
        device=device,
    )
    label_key_map = _build_label_key_map(candidate_labels)

    eval_loss_sum = 0.0
    eval_gen_loss_sum = 0.0
    eval_compress_loss_sum = 0.0
    eval_retained_sum = 0.0
    eval_compression_ratio_sum = 0.0
    valid_arc_sum = 0.0
    retained_arc_sum = 0.0
    eval_steps = 0
    clean_preds: List[str] = []
    clean_golds: List[str] = []
    prob_stats_rows: List[Dict[str, float]] = []

    with torch.no_grad():
        for step, raw_batch in enumerate(dev_loader):
            if max_eval_batches is not None and step >= int(max_eval_batches):
                break
            batch = _to_device_batch(raw_batch, device)

            outputs = model(batch, labels=batch["labels"])
            eval_loss_sum += float(outputs["loss"].detach().cpu().item())
            eval_gen_loss_sum += float(outputs["gen_loss"].detach().cpu().item())
            eval_compress_loss_sum += float(outputs["compress_loss"].detach().cpu().item())
            eval_retained_sum += float(outputs["avg_retained_arcs"].detach().cpu().item())
            eval_compression_ratio_sum += float(outputs["compression_ratio"].detach().cpu().item())
            valid_arc_sum += float(outputs["valid_arc_count"].detach().cpu().item())
            retained_arc_sum += float(outputs["retained_arc_count"].detach().cpu().item())
            prob_stats_rows.append(_collect_prob_stats(outputs=outputs, batch=batch))
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

    n = max(len(clean_preds), 1)
    mapped_preds = [
        project_prediction_to_label(p, candidate_labels, label_key_map)
        for p in clean_preds
    ]
    valid_count = sum(1 for p in clean_preds if p in label_key_map.values())
    mapped_valid_count = sum(1 for p in mapped_preds if p is not None)
    raw_exact = sum(int(p == g) for p, g in zip(clean_preds, clean_golds)) / n
    mapped_exact = sum(int((p or "") == g) for p, g in zip(mapped_preds, clean_golds)) / n
    prob_stats = _mean_dict(prob_stats_rows)

    row: Dict[str, float | int] = {
        "threshold": float(threshold),
        "eval_steps": int(eval_steps),
        "num_predictions": int(len(clean_preds)),
        "eval_loss": eval_loss_sum / max(eval_steps, 1),
        "eval_gen_loss": eval_gen_loss_sum / max(eval_steps, 1),
        "eval_compress_loss": eval_compress_loss_sum / max(eval_steps, 1),
        "eval_avg_retained_arcs": eval_retained_sum / max(eval_steps, 1),
        "eval_compression_ratio_batch_mean": eval_compression_ratio_sum / max(eval_steps, 1),
        "eval_compression_ratio_global": retained_arc_sum / max(valid_arc_sum, 1.0),
        "valid_arc_count": valid_arc_sum,
        "retained_arc_count": retained_arc_sum,
        "valid_label_rate": valid_count / n,
        "invalid_rate": 1.0 - (valid_count / n),
        "raw_exact": raw_exact,
        "mapped_exact": mapped_exact,
        "mapped_invalid_rate": 1.0 - (mapped_valid_count / n),
    }
    row.update(prob_stats)
    return row


def write_csv(rows: List[Dict[str, float | int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EA-GMIB threshold sweep analysis")
    parser.add_argument("--config", type=str, required=True, help="Full-run config JSON.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to trained best_model directory.",
    )
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    use_cuda = bool(args.use_cuda or cfg.get("use_cuda", True))
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    thresholds = _parse_thresholds(args.thresholds)
    print(f"[Sweep] device={device} thresholds={thresholds}")
    print(f"[Sweep] checkpoint_dir={checkpoint_dir}")

    # Use the saved generator tokenizer so label ids match the trained checkpoint.
    generator_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    pubmedbert_tokenizer = get_pubmedbert_tokenizer(cfg["pubmedbert_path"])

    eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else cfg.get("eval_batch_size", cfg.get("batch_size", 1))
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
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        max_src_len=int(cfg.get("max_src_len", 256)),
        max_input_len=int(cfg.get("max_input_len", 512)),
        max_target_len=int(cfg.get("max_target_len", 16)),
        max_semantic_len=int(cfg.get("max_semantic_len", 256)),
        seed=seed,
    )

    model = _build_model(
        cfg=cfg,
        generator_tokenizer=generator_tokenizer,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    print(f"[Sweep] model loaded dtype={next(model.parameters()).dtype}")

    rows: List[Dict[str, float | int]] = []
    for threshold in thresholds:
        row = evaluate_threshold(
            model=model,
            dev_loader=dev_loader,
            generator_tokenizer=generator_tokenizer,
            device=device,
            threshold=threshold,
            max_eval_batches=args.max_eval_batches,
            max_target_len=int(cfg.get("max_target_len", 16)),
        )
        rows.append(row)
        print(
            "[Sweep] threshold={threshold:.3f} mapped_exact={mapped_exact:.4f} "
            "compression_ratio={eval_compression_ratio_global:.4f} "
            "avg_retained={eval_avg_retained_arcs:.4f} p50={p_p50:.4f}".format(**row)
        )

    output_csv = Path(args.output_csv)
    write_csv(rows=rows, output_csv=output_csv)
    print(f"[Sweep] wrote {len(rows)} rows to: {output_csv}")


if __name__ == "__main__":
    main()
