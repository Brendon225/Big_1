"""
Summarize Stage-4 F1 evaluation outputs into CSV/Markdown paper tables.

This script is intentionally dependency-light: it reads the JSON files emitted
by `src/stage4/evaluate_f1.py` and does not import torch/transformers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Dict, Iterable, List


MODEL_ORDER = {
    "b0": 0,
    "b1_raw": 1,
    "b1_coarse": 2,
    "b2_raw": 3,
    "b2_coarse": 4,
    "b4": 5,
    "b5": 6,
    "b7": 7,
}
MODEL_DISPLAY = {
    "b0": "B0 text-only",
    "b1_raw": "B1 raw tree",
    "b1_coarse": "B1 coarse tree",
    "b2_raw": "B2 raw SDP",
    "b2_coarse": "Stage2 B2-coarse",
    "b4": "B4 semantics-only",
    "b5": "B5 dual-view concat",
    "b7": "B7 gated dual-view",
}


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return fmean(values) if values else 0.0


def std(values: Iterable[float]) -> float:
    values = list(values)
    return stdev(values) if len(values) > 1 else 0.0


def fmt(value: float, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_pm(avg: float, sd: float, digits: int = 4) -> str:
    return f"{avg:.{digits}f} ± {sd:.{digits}f}"


def collect_metrics(input_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(input_dir.glob("*_metrics.json")):
        payload = read_json(path)
        rows.append(
            {
                "file": path.name,
                "dataset": payload["dataset_name"],
                "model": payload["model_type"],
                "seed": int(payload["seed"]),
                "precision": float(payload["precision_excluding_no_relation"]),
                "recall": float(payload["recall_excluding_no_relation"]),
                "micro_f1": float(payload["micro_f1_excluding_no_relation"]),
                "macro_f1": float(payload["macro_f1_excluding_no_relation"]),
                "mapped_exact": float(payload["mapped_exact"]),
                "invalid_rate": float(payload["invalid_rate"]),
                "eval_loss": float(payload["eval_loss"]),
                "gate_semantic": (
                    float(payload["gate_semantic_mean"])
                    if "gate_semantic_mean" in payload
                    else None
                ),
                "gate_syntax": (
                    float(payload["gate_syntax_mean"])
                    if "gate_syntax_mean" in payload
                    else None
                ),
                "num_predictions": int(payload["num_predictions"]),
            }
        )
    return rows


def summarize(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)

    summary = []
    for (dataset, model), group in grouped.items():
        group = sorted(group, key=lambda row: row["seed"])
        gate_sem_values = [row["gate_semantic"] for row in group if row["gate_semantic"] is not None]
        gate_syn_values = [row["gate_syntax"] for row in group if row["gate_syntax"] is not None]
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n": len(group),
                "seeds": ",".join(str(row["seed"]) for row in group),
                "precision_mean": mean(row["precision"] for row in group),
                "precision_std": std(row["precision"] for row in group),
                "recall_mean": mean(row["recall"] for row in group),
                "recall_std": std(row["recall"] for row in group),
                "micro_f1_mean": mean(row["micro_f1"] for row in group),
                "micro_f1_std": std(row["micro_f1"] for row in group),
                "macro_f1_mean": mean(row["macro_f1"] for row in group),
                "macro_f1_std": std(row["macro_f1"] for row in group),
                "mapped_exact_mean": mean(row["mapped_exact"] for row in group),
                "mapped_exact_std": std(row["mapped_exact"] for row in group),
                "invalid_rate_mean": mean(row["invalid_rate"] for row in group),
                "eval_loss_mean": mean(row["eval_loss"] for row in group),
                "gate_semantic_mean": mean(gate_sem_values) if gate_sem_values else None,
                "gate_syntax_mean": mean(gate_syn_values) if gate_syn_values else None,
            }
        )
    return sorted(summary, key=lambda row: (row["dataset"], MODEL_ORDER.get(row["model"], 99)))


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "model_display",
        "n",
        "seeds",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "micro_f1_mean",
        "micro_f1_std",
        "macro_f1_mean",
        "macro_f1_std",
        "mapped_exact_mean",
        "mapped_exact_std",
        "invalid_rate_mean",
        "eval_loss_mean",
        "gate_semantic_mean",
        "gate_syntax_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_by_dataset(summary: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for row in summary:
        current = out.get(row["dataset"])
        if current is None or row["micro_f1_mean"] > current["micro_f1_mean"]:
            out[row["dataset"]] = row
    return out


def write_summary_md(summary: List[Dict], path: Path) -> None:
    lines = [
        "# Stage4 F1 Summary",
        "",
        "Metrics are computed on the test split. Precision, recall, Micro-F1, and Macro-F1 exclude `NO_RELATION`.",
        "",
        "| Dataset | Model | P | R | Micro-F1 | Macro-F1 | mapped exact | invalid rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    best = best_by_dataset(summary)
    for row in summary:
        model = row["model_display"]
        if best.get(row["dataset"]) is row:
            model = f"**{model}**"
        lines.append(
            "| {dataset} | {model} | {p} | {r} | {micro} | {macro} | {exact} | {invalid} |".format(
                dataset=row["dataset"],
                model=model,
                p=fmt_pm(row["precision_mean"], row["precision_std"]),
                r=fmt_pm(row["recall_mean"], row["recall_std"]),
                micro=fmt_pm(row["micro_f1_mean"], row["micro_f1_std"]),
                macro=fmt_pm(row["macro_f1_mean"], row["macro_f1_std"]),
                exact=fmt_pm(row["mapped_exact_mean"], row["mapped_exact_std"]),
                invalid=fmt(row["invalid_rate_mean"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_per_class(input_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(input_dir.glob("*_per_class.json")):
        metrics_path = path.with_name(path.name.replace("_per_class.json", "_metrics.json"))
        if not metrics_path.exists():
            continue
        meta = read_json(metrics_path)
        model = meta["model_type"]
        dataset = meta["dataset_name"]
        seed = int(meta["seed"])
        payload = read_json(path)
        for label, metrics in payload.items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "label": label,
                    "support": int(metrics.get("support", 0)),
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1": float(metrics.get("f1", 0.0)),
                }
            )
    return rows


def write_per_class_csv(rows: List[Dict], path: Path) -> None:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["label"])].append(row)
    out_rows = []
    for (dataset, model, label), group in grouped.items():
        out_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "label": label,
                "support_mean": mean(row["support"] for row in group),
                "precision_mean": mean(row["precision"] for row in group),
                "recall_mean": mean(row["recall"] for row in group),
                "f1_mean": mean(row["f1"] for row in group),
                "f1_std": std(row["f1"] for row in group),
            }
        )
    out_rows = sorted(
        out_rows,
        key=lambda row: (row["dataset"], MODEL_ORDER.get(row["model"], 99), row["label"]),
    )
    fields = [
        "dataset",
        "model",
        "label",
        "support_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "f1_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)


def write_report(summary: List[Dict], path: Path) -> None:
    by_key = {(row["dataset"], row["model"]): row for row in summary}
    best = best_by_dataset(summary)

    key_findings = []
    for dataset in sorted({row["dataset"] for row in summary}):
        dataset_best = best[dataset]
        key_findings.append(
            f"- On {dataset}, {dataset_best['model_display']} achieves the best Micro-F1 "
            f"({fmt_pm(dataset_best['micro_f1_mean'], dataset_best['micro_f1_std'])})."
        )
        if (dataset, "b0") in by_key:
            gain = dataset_best["micro_f1_mean"] - by_key[(dataset, "b0")]["micro_f1_mean"]
            key_findings.append(
                f"  Compared with B0 text-only, the best model improves Micro-F1 by {fmt(gain)}."
            )
        if (dataset, "b2_coarse") in by_key:
            gain = dataset_best["micro_f1_mean"] - by_key[(dataset, "b2_coarse")]["micro_f1_mean"]
            key_findings.append(
                f"  Compared with Stage2 B2-coarse, the best model improves Micro-F1 by {fmt(gain)}."
            )
        if (dataset, "b2_raw") in by_key and (dataset, "b2_coarse") in by_key:
            gain = by_key[(dataset, "b2_coarse")]["micro_f1_mean"] - by_key[(dataset, "b2_raw")]["micro_f1_mean"]
            key_findings.append(
                f"  B2 coarse-vs-raw SDP difference is {fmt(gain)} Micro-F1."
            )

    lines = [
        "# Stage4 Final Evaluation Report",
        "",
        "## Scope",
        "",
        "This report summarizes test-split F1 evaluation for Stage2 B2-coarse and Stage3 B4/B5/B7 on ChemProtSent and CDRIntra. Main F1 metrics exclude `NO_RELATION`.",
        "",
        "## Key Findings",
        "",
        *key_findings,
        "- All evaluated runs use label-rerank; invalid_rate should remain 0.0000 if every output maps to a legal relation label.",
        "",
        "## Main Table",
        "",
        "| Dataset | Model | P | R | Micro-F1 | Macro-F1 | mapped exact | invalid rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {model} | {p} | {r} | {micro} | {macro} | {exact} | {invalid} |".format(
                dataset=row["dataset"],
                model=row["model_display"],
                p=fmt_pm(row["precision_mean"], row["precision_std"]),
                r=fmt_pm(row["recall_mean"], row["recall_std"]),
                micro=fmt_pm(row["micro_f1_mean"], row["micro_f1_std"]),
                macro=fmt_pm(row["macro_f1_mean"], row["macro_f1_std"]),
                exact=fmt_pm(row["mapped_exact_mean"], row["mapped_exact_std"]),
                invalid=fmt(row["invalid_rate_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The test F1 results should be interpreted in three layers: B0 measures the pure text-only generation baseline, B1/B2 raw-vs-coarse variants measure dependency injection and entity coarsening effects, and B4/B5/B7 measure explicit semantic/syntactic representation learning. The strongest final variant may be dataset-dependent, so the paper narrative should emphasize the component-level evidence rather than relying on a single absolute ranking.",
            "",
            "A conservative paper narrative should report the full baseline ladder and then discuss where dual-view modeling improves over both text-only and dependency-linearization baselines.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage4 F1 metrics.")
    parser.add_argument("--input_dir", default="outputs/stage4_predictions/full")
    parser.add_argument("--summary_csv", default="checkpoints/stage4_f1_summary.csv")
    parser.add_argument("--summary_md", default="checkpoints/stage4_f1_summary.md")
    parser.add_argument("--per_class_csv", default="checkpoints/stage4_per_class_summary.csv")
    parser.add_argument(
        "--report_md",
        default="docs/stage4_experiments_docs/stage4_final_eval_report.md",
    )
    parser.add_argument(
        "--paper_tables_md",
        default="docs/stage4_experiments_docs/stage4_paper_tables.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = collect_metrics(input_dir)
    if not rows:
        raise FileNotFoundError(f"No *_metrics.json files found in {input_dir}")
    summary = summarize(rows)
    write_csv(summary, Path(args.summary_csv))
    write_summary_md(summary, Path(args.summary_md))
    write_summary_md(summary, Path(args.paper_tables_md))
    write_per_class_csv(collect_per_class(input_dir), Path(args.per_class_csv))
    write_report(summary, Path(args.report_md))
    print(f"[Stage4Summary] metrics={len(rows)} groups={len(summary)}")
    print(f"[Stage4Summary] wrote {args.summary_csv}")
    print(f"[Stage4Summary] wrote {args.summary_md}")
    print(f"[Stage4Summary] wrote {args.report_md}")


if __name__ == "__main__":
    main()
