"""
Build a shared dependency-label vocabulary for Stage-3 graph modules.

Example:
python src/stage3/build_dep_type_vocab.py ^
  --input data/experiment_views/coarsened/ChemProtSent_train.json ^
          data/experiment_views/coarsened/ChemProtSent_dev.json ^
          data/experiment_views/coarsened/ChemProtSent_test.json ^
          data/experiment_views/coarsened/CDRIntra_train.json ^
          data/experiment_views/coarsened/CDRIntra_dev.json ^
          data/experiment_views/coarsened/CDRIntra_test.json ^
  --output data/dep_type_vocab.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.builders.graph_builder import build_dep_type_vocab, save_dep_type_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dependency type vocabulary.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more coarsened jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dep_type_vocab.json",
        help="Path to output dep-type vocab json.",
    )
    parser.add_argument(
        "--labels_key",
        type=str,
        default=None,
        help=(
            "Record field name that stores dependency labels. "
            "Defaults to dep_labels for --dep_view raw and coarse_labels for --dep_view coarse."
        ),
    )
    parser.add_argument(
        "--dep_view",
        type=str,
        choices=["raw", "coarse"],
        default="coarse",
        help="Dependency graph view used to choose the default labels field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(p) for p in args.input]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    labels_key = args.labels_key
    if labels_key is None:
        labels_key = "dep_labels" if args.dep_view == "raw" else "coarse_labels"

    vocab = build_dep_type_vocab(
        jsonl_paths=input_paths,
        labels_key=labels_key,
    )
    save_dep_type_vocab(vocab=vocab, path=args.output)

    print(f"[dep-vocab] saved: {args.output}")
    print(f"[dep-vocab] dep_view: {args.dep_view}")
    print(f"[dep-vocab] labels_key: {labels_key}")
    print(f"[dep-vocab] size: {len(vocab)}")
    print(f"[dep-vocab] sample: {list(vocab.items())[:10]}")


if __name__ == "__main__":
    main()
