"""
Run local seed sweeps for Stage-2 baseline smoke experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_CONFIGS = [
    "configs/baseline_local_smoke.json",
    "configs/baseline_local_smoke_b1_raw.json",
    "configs/baseline_local_smoke_b2_raw.json",
    "configs/baseline_local_smoke_b1_coarse.json",
    "configs/baseline_local_smoke_b2_coarse.json",
    "configs/baseline_local_smoke_cdrintra_b0.json",
    "configs/baseline_local_smoke_cdrintra_b1_raw.json",
    "configs/baseline_local_smoke_cdrintra_b2_raw.json",
    "configs/baseline_local_smoke_cdrintra_b1_coarse.json",
    "configs/baseline_local_smoke_cdrintra_b2_coarse.json",
]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seed sweep for local smoke baselines.")
    parser.add_argument(
        "--python",
        type=str,
        default=r".\venv\Scripts\python.exe",
        help="Python executable used for each run.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="List of config files to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42, 123, 456],
        help="Seed list.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="checkpoints/sweep_summary.csv",
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue subsequent runs if one run fails.",
    )
    return parser.parse_args()


def with_seed_suffix(output_dir: str, seed: int) -> str:
    base = re.sub(r"_seed\d+$", "", output_dir)
    return f"{base}_seed{seed}"


def run_one(py: str, config_path: str, seed: int) -> Dict:
    cfg = load_json(Path(config_path))
    base_output = cfg["output_dir"]
    run_output = with_seed_suffix(base_output, seed)

    cmd = [
        py,
        "src/baseline/run_smoke.py",
        "--config",
        config_path,
        "--seed",
        str(seed),
        "--output_dir",
        run_output,
    ]
    print(f"[Sweep] running: config={config_path} seed={seed}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed: config={config_path}, seed={seed}, returncode={completed.returncode}"
        )

    metrics_path = Path(run_output) / "metrics.json"
    run_meta_path = Path(run_output) / "run_meta.json"
    if not metrics_path.exists() or not run_meta_path.exists():
        raise FileNotFoundError(
            f"Missing metrics or meta file in {run_output}"
        )
    metrics = load_json(metrics_path)
    run_meta = load_json(run_meta_path)

    row = {
        "config": config_path,
        "dataset_name": run_meta.get("dataset_name", ""),
        "dep_view": run_meta.get("dep_view", ""),
        "dep_form": run_meta.get("dep_form", ""),
        "use_dep": run_meta.get("use_dep", False),
        "prediction_mode": run_meta.get("prediction_mode", ""),
        "seed": seed,
        "output_dir": run_output,
        "train_loss": metrics.get("train_loss"),
        "eval_loss": metrics.get("eval_loss"),
        "raw_exact": metrics.get("raw_exact"),
        "mapped_exact": metrics.get("mapped_exact"),
        "invalid_rate": metrics.get("invalid_rate"),
        "mapped_invalid_rate": metrics.get("mapped_invalid_rate"),
        "valid_label_rate": metrics.get("valid_label_rate"),
    }
    return row


def write_summary(rows: List[Dict], output_csv: str) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Sweep] summary saved: {path}")


def main() -> None:
    args = parse_args()
    rows: List[Dict] = []
    for config_path in args.configs:
        for seed in args.seeds:
            try:
                row = run_one(args.python, config_path, seed)
            except Exception as exc:
                print(f"[Sweep] failed: config={config_path} seed={seed} err={exc}")
                if not args.continue_on_error:
                    raise
            else:
                rows.append(row)

    write_summary(rows, args.summary_csv)
    print(f"[Sweep] completed runs: {len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Sweep] abort: {exc}")
        sys.exit(1)
