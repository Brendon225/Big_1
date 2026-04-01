#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify the JSONL files under data/processed.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

EXPECTED_FILES = {
    "CDR_train.json": 100,
    "CDR_dev.json": 50,
    "CDR_test.json": 50,
    "ChemProt_train.json": 500,
    "ChemProt_dev.json": 200,
    "ChemProt_test.json": 200,
    "DDI_train.json": 300,
    "DDI_dev.json": 50,
    "DDI_test.json": 100,
}

REQUIRED_FIELDS = {
    "id", "dataset", "split", "sentence", "tokens",
    "entity1", "entity2", "relation", "is_positive",
}
REQUIRED_ENTITY_FIELDS = {"text", "type", "start_char", "end_char", "start_tok", "end_tok"}

PASS = "[PASS]"
FAIL = "[FAIL]"


def check_file(filename: str, min_records: int) -> bool:
    path = PROCESSED_DIR / filename
    issues = []

    if not path.exists():
        print(f"{FAIL} {filename:<30} missing")
        return False

    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    issues.append(f"line {lineno}: JSON decode error: {exc}")
    except Exception as exc:
        print(f"{FAIL} {filename:<30} read failed: {exc}")
        return False

    if len(records) < min_records:
        issues.append(f"record count {len(records)} below minimum {min_records}")

    field_errors = 0
    token_errors = 0
    for record in records[:20]:
        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            field_errors += 1
            continue

        for key in ("entity1", "entity2"):
            entity = record[key]
            missing_entity = REQUIRED_ENTITY_FIELDS - set(entity.keys())
            if missing_entity:
                field_errors += 1
                continue

            n_tokens = len(record["tokens"])
            start_tok = entity["start_tok"]
            end_tok = entity["end_tok"]
            if not (0 <= start_tok < end_tok <= n_tokens):
                token_errors += 1

    if field_errors:
        issues.append(f"{field_errors} sampled records missing required fields")
    if token_errors:
        issues.append(f"{token_errors} sampled entities have invalid token spans")

    if issues:
        print(f"{FAIL} {filename:<30} {len(records)} records")
        for issue in issues:
            print(f"       - {issue}")
        return False

    print(f"{PASS} {filename:<30} {len(records)} records")
    return True


def check_stats() -> bool:
    path = PROCESSED_DIR / "dataset_stats.json"
    if not path.exists():
        print(f"{FAIL} {'dataset_stats.json':<30} missing")
        return False
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
        datasets_found = list(stats.keys())
        print(f"{PASS} {'dataset_stats.json':<30} datasets={datasets_found}")
        return True
    except Exception as exc:
        print(f"{FAIL} {'dataset_stats.json':<30} read failed: {exc}")
        return False


def main():
    print("=" * 60)
    print("  Verify processed data")
    print("=" * 60)

    ok = True
    for filename, min_records in EXPECTED_FILES.items():
        ok &= check_file(filename, min_records)
    ok &= check_stats()

    print()
    if ok:
        print("Processed data verification completed successfully.")
        print("Next: run src/preprocessing/dep_parser.py")
    else:
        print("Processed data verification found issues.")


if __name__ == "__main__":
    main()