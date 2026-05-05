"""Stage-3 training and utility entry points."""

from src.stage3.trainer_b345 import run_experiment as run_b345_experiment
from src.stage3.trainer_ours import run_experiment as run_ours_experiment

# Backward-compatible default export keeps the baseline runner behavior.
run_experiment = run_b345_experiment

__all__ = ["run_experiment", "run_b345_experiment", "run_ours_experiment"]
