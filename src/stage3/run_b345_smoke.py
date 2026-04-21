"""
Local smoke runner for Stage-3 B3/B4/B5 baselines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.stage3.trainer_b345 import load_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 B345 local smoke run")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3_smoke_b3_chemprotsent.json",
        help="Path to smoke config JSON.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    run_experiment(cfg)
