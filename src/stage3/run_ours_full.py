"""
Server/full runner for Stage-3 Ours (EA-GMIB).

The JSON config remains the source of truth for official runs. CLI overrides
are provided only for local smoke checks and emergency server debugging.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.stage3.trainer_ours import load_config, run_experiment


def _optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 EA-GMIB full run")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3_full_ours_chemprotsent_seed42.json",
        help="Path to full-run config JSON.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # Smoke/debug overrides. Leave unset for official full runs.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--save_best_model", type=_optional_bool, default=None)
    parser.add_argument("--save_final_model", type=_optional_bool, default=None)
    parser.add_argument("--use_cuda", type=_optional_bool, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    override_names = [
        "seed",
        "output_dir",
        "epochs",
        "max_train_batches",
        "max_eval_batches",
        "batch_size",
        "eval_batch_size",
        "save_best_model",
        "save_final_model",
        "use_cuda",
    ]
    for name in override_names:
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value

    run_experiment(cfg)
