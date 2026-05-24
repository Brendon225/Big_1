"""Summarize Stage3 full-run metrics across models, datasets, and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("b4", "b5", "b7")
DEFAULT_DATASETS = ("ChemProtSent", "CDRIntra")
DEFAULT_SEEDS = (42, 123, 456)
PRIMARY_METRICS = (
    "mapped_exact",
    "raw_exact",
    "eval_loss",
    "epoch",
    "eval_gate_semantic_mean",
    "eval_gate_syntax_mean",
)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dataset_slug(dataset: str) -> str:
    return dataset.lower()


def _checkpoint_dir(root: Path, model: str, dataset: str, seed: int) -> Path:
    return root / f"stage3_full_{model}_{_dataset_slug(dataset)}_seed{seed}"


def _read_metrics(path: Path) -> dict[str, Any] | None:
    metrics_path = path / "best_metrics.json"
    if not metrics_path.exists():
        metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows(
    checkpoint_root: Path,
    models: list[str],
    datasets: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                ckpt_dir = _checkpoint_dir(checkpoint_root, model, dataset, seed)
                metrics = _read_metrics(ckpt_dir)
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "checkpoint_dir": str(ckpt_dir),
                    "status": "ok" if metrics is not None else "missing",
                }
                for metric in PRIMARY_METRICS:
                    row[metric] = metrics.get(metric, "") if metrics else ""
                if metrics and row["epoch"] == "":
                    row["epoch"] = metrics.get("best_epoch", "")
                rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        groups.setdefault((row["dataset"], row["model"]), []).append(row)

    aggregate: list[dict[str, Any]] = []
    for (dataset, model), group in sorted(groups.items()):
        mapped_values = [float(row["mapped_exact"]) for row in group if row["mapped_exact"] != ""]
        raw_values = [float(row["raw_exact"]) for row in group if row["raw_exact"] != ""]
        loss_values = [float(row["eval_loss"]) for row in group if row["eval_loss"] != ""]
        aggregate.append(
            {
                "dataset": dataset,
                "model": model,
                "n": len(mapped_values),
                "mapped_exact_mean": statistics.fmean(mapped_values) if mapped_values else "",
                "mapped_exact_std": statistics.stdev(mapped_values) if len(mapped_values) > 1 else 0.0,
                "raw_exact_mean": statistics.fmean(raw_values) if raw_values else "",
                "eval_loss_mean": statistics.fmean(loss_values) if loss_values else "",
            }
        )
    return aggregate


def _format_float(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "seed",
        "status",
        *PRIMARY_METRICS,
        "checkpoint_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_float(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_rows_md = [
        [
            row["dataset"],
            row["model"],
            row["n"],
            row["mapped_exact_mean"],
            row["mapped_exact_std"],
            row["raw_exact_mean"],
            row["eval_loss_mean"],
        ]
        for row in aggregate
    ]
    per_seed_rows_md = [
        [
            row["dataset"],
            row["model"],
            row["seed"],
            row["status"],
            row["mapped_exact"],
            row["raw_exact"],
            row["eval_loss"],
            row["epoch"],
            row["eval_gate_semantic_mean"],
            row["eval_gate_syntax_mean"],
        ]
        for row in rows
    ]
    content = "\n".join(
        [
            "# Stage3 Multi-Seed Summary",
            "",
            "This file is generated by `src/stage3/summarize_stage3_multiseed.py`.",
            "Rows marked `missing` indicate that the corresponding 5090 checkpoint has not been copied back yet.",
            "",
            "## Aggregate",
            "",
            _markdown_table(
                [
                    "dataset",
                    "model",
                    "n",
                    "mapped_exact_mean",
                    "mapped_exact_std",
                    "raw_exact_mean",
                    "eval_loss_mean",
                ],
                aggregate_rows_md,
            ),
            "",
            "## Per-Seed",
            "",
            _markdown_table(
                [
                    "dataset",
                    "model",
                    "seed",
                    "status",
                    "mapped_exact",
                    "raw_exact",
                    "eval_loss",
                    "epoch",
                    "gate_sem",
                    "gate_syn",
                ],
                per_seed_rows_md,
            ),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--output_csv", default="checkpoints/stage3_multiseed_summary.csv")
    parser.add_argument("--output_md", default="docs/stage3_model_docs/stage3_multiseed_summary.md")
    args = parser.parse_args()

    models = _split_csv(args.models)
    datasets = _split_csv(args.datasets)
    seeds = [int(seed) for seed in _split_csv(args.seeds)]

    rows = collect_rows(Path(args.checkpoint_root), models, datasets, seeds)
    aggregate = aggregate_rows(rows)
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows, aggregate)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    missing_count = len(rows) - ok_count
    print(f"[Summary] wrote {args.output_csv}")
    print(f"[Summary] wrote {args.output_md}")
    print(f"[Summary] ok={ok_count} missing={missing_count}")


if __name__ == "__main__":
    main()
