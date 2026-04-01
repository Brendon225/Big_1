#!/usr/bin/env python3
"""
Stage 1 statistics for the BioRE data assets.

This script summarizes the original full-context views under data/coarsened and
writes the outputs into docs/data_docs.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "coarsened"
OUT_DIR = ROOT / "docs" / "data_docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "CDR_train.json",
    "CDR_dev.json",
    "CDR_test.json",
    "ChemProt_train.json",
    "ChemProt_dev.json",
    "ChemProt_test.json",
    "DDI_train.json",
    "DDI_dev.json",
    "DDI_test.json",
]

DDI_SAMPLE_SIZE = 8000


def normalize_node_text(text: str) -> str:
    return "_".join(str(text).split())


def load_full(filepath: Path) -> List[Dict]:
    records = []
    with filepath.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def step_sample(filepath: Path, target_size: int) -> List[Dict]:
    file_size_gb = os.path.getsize(filepath) / 1e9
    est_bytes_per_record = 40000 if file_size_gb > 1 else 3000
    est_lines = max(target_size, int(file_size_gb * 1e9 / est_bytes_per_record))
    step = max(1, est_lines // target_size)
    print(f"    estimated_lines={est_lines:,}  step={step}", flush=True)

    sampled = []
    with filepath.open(encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            if idx % step != 0:
                continue
            raw_line = raw_line.strip()
            if raw_line:
                sampled.append(json.loads(raw_line))
    return sampled


def percentile(values: List[int], p: int) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    idx = int(math.ceil(p / 100 * len(sorted_values))) - 1
    return sorted_values[max(0, idx)]


def describe(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": round(sum(values) / len(values), 1),
    }


def count_valid_arcs(heads: List[int]) -> int:
    count = 0
    for dep_idx, head_idx in enumerate(heads):
        if head_idx is None or head_idx < 0:
            continue
        if head_idx == dep_idx:
            continue
        count += 1
    return count


def linearize_arcs(tokens: List[str], heads: List[int], labels: List[str]) -> str:
    parts = []
    limit = min(len(tokens), len(heads), len(labels))
    for dep_idx in range(limit):
        head_idx = heads[dep_idx]
        if head_idx is None or head_idx < 0 or head_idx >= limit:
            continue
        if head_idx == dep_idx:
            continue
        src = normalize_node_text(tokens[head_idx])
        tgt = normalize_node_text(tokens[dep_idx])
        parts.append(f"{src}->{tgt}:{labels[dep_idx]}")
    return " ".join(parts)


def fmt_desc(summary: Dict[str, float]) -> str:
    if not summary:
        return "N/A"
    return (
        f"min={summary['min']} / p25={summary['p25']} / p50={summary['p50']} / "
        f"p75={summary['p75']} / p90={summary['p90']} / p95={summary['p95']} / "
        f"p99={summary['p99']} / max={summary['max']} / mean={summary['mean']}"
    )


stats: Dict[str, Dict] = {}

for filename in FILES:
    file_path = DATA_DIR / filename
    is_ddi = filename.startswith("DDI")

    print(f"[INFO] Loading {filename} ...", flush=True)
    records = step_sample(file_path, DDI_SAMPLE_SIZE) if is_ddi else load_full(file_path)
    if is_ddi:
        print(f"  sampled {len(records):,} records", flush=True)
    else:
        print(f"  n={len(records):,}", flush=True)

    token_lengths = []
    entity1_lengths = []
    entity2_lengths = []
    original_arc_counts = []
    coarse_arc_counts = []
    coarse_node_counts = []
    linearized_dep_words = []
    full_input_bpe = []
    label_counts = Counter()
    coarse_status_counts = Counter()
    class_counts = Counter()

    for record in records:
        tokens = record.get("tokens", [])
        entity1 = record.get("entity1", {})
        entity2 = record.get("entity2", {})
        dep_heads = record.get("dep_heads", [])
        coarse_tokens = record.get("coarse_tokens", [])
        coarse_heads = record.get("coarse_heads", [])
        coarse_labels = record.get("coarse_labels", [])
        relation = record.get("relation", "")
        coarse_status = record.get("coarse_status", "unknown")

        token_lengths.append(len(tokens))
        entity1_lengths.append(max(1, entity1.get("end_tok", 0) - entity1.get("start_tok", 0)))
        entity2_lengths.append(max(1, entity2.get("end_tok", 0) - entity2.get("start_tok", 0)))
        original_arc_counts.append(count_valid_arcs(dep_heads))
        coarse_arc_counts.append(count_valid_arcs(coarse_heads))
        coarse_node_counts.append(len(coarse_tokens))

        dep_string = linearize_arcs(coarse_tokens, coarse_heads, coarse_labels)
        dep_word_count = len(dep_string.split()) if dep_string else 0
        linearized_dep_words.append(dep_word_count)

        proxy_bpe = int((len(tokens) + 4) * 1.3)
        if dep_word_count:
            proxy_bpe += int((dep_word_count + 2) * 1.3)
        full_input_bpe.append(proxy_bpe)

        label_counts[relation] += 1
        coarse_status_counts[coarse_status] += 1
        class_counts["positive" if record.get("is_positive", True) else "negative"] += 1

    total_orig_arcs = sum(original_arc_counts)
    total_coarse_arcs = sum(coarse_arc_counts)
    arc_reduction_pct = ((total_orig_arcs - total_coarse_arcs) / total_orig_arcs * 100) if total_orig_arcs else 0.0

    stats[filename] = {
        "n_records": len(records),
        "sampled": is_ddi,
        "token_len": describe(token_lengths),
        "e1_len": describe(entity1_lengths),
        "e2_len": describe(entity2_lengths),
        "orig_arcs": describe(original_arc_counts),
        "coarse_arcs": describe(coarse_arc_counts),
        "coarse_nodes": describe(coarse_node_counts),
        "lin_dep_words": describe(linearized_dep_words),
        "full_input_bpe": describe(full_input_bpe),
        "arc_reduction_pct": round(arc_reduction_pct, 2),
        "label_dist": dict(label_counts.most_common()),
        "class_dist": dict(class_counts),
        "coarse_status_dist": dict(coarse_status_counts),
    }

    print(
        f"  arc_reduction={arc_reduction_pct:.2f}%  token_p95={percentile(token_lengths, 95)}  full_input_p95={percentile(full_input_bpe, 95)}",
        flush=True,
    )

print("\n[INFO] Writing results...", flush=True)

json_path = OUT_DIR / "stats_1_4.json"
json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  Saved {json_path}", flush=True)

report_lines = []
report_lines.append("# Stage 1 data statistics report\n")
report_lines.append("> DDI files use deterministic step sampling (target about 8,000 examples); all other files are processed exhaustively.\n")
report_lines.append("> Arc counts exclude ROOT/self-loop arcs; multi-word nodes are normalized with underscores in dependency linearization.\n")

for filename, summary in stats.items():
    sampled_tag = " *(sampled)*" if summary["sampled"] else ""
    report_lines.append(f"\n## {filename}{sampled_tag}\n")
    report_lines.append(f"- **records**: {summary['n_records']:,}\n")
    report_lines.append(f"- **token length** (p50/p95/max): {summary['token_len']['p50']} / {summary['token_len']['p95']} / {summary['token_len']['max']}\n")
    report_lines.append(f"  - full distribution: {fmt_desc(summary['token_len'])}\n")
    report_lines.append(f"- **entity1 length** (p50/p95/max): {summary['e1_len']['p50']} / {summary['e1_len']['p95']} / {summary['e1_len']['max']}\n")
    report_lines.append(f"- **entity2 length** (p50/p95/max): {summary['e2_len']['p50']} / {summary['e2_len']['p95']} / {summary['e2_len']['max']}\n")
    report_lines.append(f"- **original arc count** (p50/p95/max): {summary['orig_arcs']['p50']} / {summary['orig_arcs']['p95']} / {summary['orig_arcs']['max']}\n")
    report_lines.append(f"- **coarsened arc count** (p50/p95/max): {summary['coarse_arcs']['p50']} / {summary['coarse_arcs']['p95']} / {summary['coarse_arcs']['max']}\n")
    report_lines.append(f"- **arc reduction**: {summary['arc_reduction_pct']}%\n")
    report_lines.append(f"- **linearized dep words** (p50/p95/max): {summary['lin_dep_words']['p50']} / {summary['lin_dep_words']['p95']} / {summary['lin_dep_words']['max']}\n")
    report_lines.append(f"- **estimated full-input BPE** (p50/p95/p99/max): {summary['full_input_bpe']['p50']} / {summary['full_input_bpe']['p95']} / {summary['full_input_bpe']['p99']} / {summary['full_input_bpe']['max']}\n")
    report_lines.append(f"- **class distribution**: {summary['class_dist']}\n")
    report_lines.append(f"- **coarse_status distribution**: {summary['coarse_status_dist']}\n")
    report_lines.append("- **label distribution**:\n")
    for label, count in summary["label_dist"].items():
        report_lines.append(f"  - `{label}`: {count:,}\n")

max_p99 = max(summary["full_input_bpe"].get("p99", 0) for summary in stats.values())
report_lines.append("\n## Key conclusions and max_input_length suggestion\n")
report_lines.append(f"- Maximum p99 of estimated full-input BPE across datasets: {max_p99}\n")
report_lines.append("- For B0 text-only, start with `max_input_length = 512`.\n")
report_lines.append("- For B1 and later text+dep settings, use `1024` and allow tail truncation in the collator/tokenizer.\n")
report_lines.append("- DDI still requires an entity-centered text window; do not feed full document text plus full dependency text together.\n")

md_path = OUT_DIR / "stats_1_4_report.md"
md_path.write_text("".join(report_lines), encoding="utf-8")
print(f"  Saved {md_path}", flush=True)
print("[DONE]", flush=True)