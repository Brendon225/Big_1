#!/usr/bin/env python3
"""
Run dependency parsing and entity-aware coarsening on the new experiment views.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.dep_parser import load_nlp, process_file as parse_file
from src.preprocessing.entity_coarsen import process_file as coarsen_file

VIEW_DIR = ROOT / "data" / "experiment_views"
PROCESSED_DIR = VIEW_DIR / "processed"
PARSED_DIR = VIEW_DIR / "parsed"
COARSENED_DIR = VIEW_DIR / "coarsened"

PARSED_DIR.mkdir(parents=True, exist_ok=True)
COARSENED_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "ChemProtSent_train.json",
    "ChemProtSent_dev.json",
    "ChemProtSent_test.json",
    "CDRIntra_train.json",
    "CDRIntra_dev.json",
    "CDRIntra_test.json",
]


def main() -> None:
    print("=" * 60)
    print("  Postprocess experiment views")
    print("=" * 60)

    nlp = load_nlp()

    print("\n[Parsing]")
    for name in FILES:
        src = PROCESSED_DIR / name
        dst = PARSED_DIR / name
        if not src.exists():
            print(f"  [WARN] missing processed view: {src.name}")
            continue
        parse_file(nlp, src, dst)

    print("\n[Coarsening]")
    for name in FILES:
        src = PARSED_DIR / name
        dst = COARSENED_DIR / name
        if not src.exists():
            print(f"  [WARN] missing parsed view: {src.name}")
            continue
        coarsen_file(src, dst)

    print("\nOutput dirs:")
    print("  ", PARSED_DIR)
    print("  ", COARSENED_DIR)


if __name__ == "__main__":
    main()