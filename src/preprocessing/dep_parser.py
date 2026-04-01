#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dependency parsing for JSONL files under data/processed.

For each record, this script adds:
- dep_heads
- dep_labels

Output files are written to data/parsed.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
PARSED_DIR = ROOT / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "CDR_train.json", "CDR_dev.json", "CDR_test.json",
    "ChemProt_train.json", "ChemProt_dev.json", "ChemProt_test.json",
    "DDI_train.json", "DDI_dev.json", "DDI_test.json",
]


def load_nlp():
    """Load SciSpaCy and keep only tok2vec+parser for speed."""
    try:
        import spacy
        nlp = spacy.load("en_core_sci_lg")
        disabled = [name for name in nlp.pipe_names if name not in ("tok2vec", "parser")]
        for name in disabled:
            nlp.disable_pipe(name)
        print(f"  loaded model: en_core_sci_lg  disabled={disabled}")
        return nlp
    except OSError:
        print("[ERROR] Missing en_core_sci_lg model.")
        print("Install with:")
        print("  pip install scispacy")
        print("  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz")
        sys.exit(1)


def process_file(nlp, src_path: Path, dst_path: Path, batch_size: int = 256):
    """Parse one JSONL file in batches and write the parsed version."""
    records = []
    with src_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if total == 0:
        print(f"  [SKIP] {src_path.name} is empty")
        return 0

    unique_sents = list({record["sentence"] for record in records})
    sent_to_dep = {}

    print(f"  parsing {src_path.name}: {total} records, {len(unique_sents)} unique sentences")
    t0 = time.time()

    for start in range(0, len(unique_sents), batch_size):
        batch = unique_sents[start : start + batch_size]
        for doc, sent in zip(nlp.pipe(batch, batch_size=batch_size), batch):
            spacy_tokens = [token.text for token in doc]
            dep_heads = [token.head.i for token in doc]
            dep_labels = [token.dep_ for token in doc]
            char_to_tok = {}
            for token in doc:
                for char_idx in range(token.idx, token.idx + len(token.text)):
                    char_to_tok[char_idx] = token.i
            sent_to_dep[sent] = (spacy_tokens, dep_heads, dep_labels, char_to_tok)

        done = min(start + batch_size, len(unique_sents))
        elapsed = time.time() - t0
        print(f"    {done}/{len(unique_sents)} sentences  ({elapsed:.1f}s)", end="\r")

    print()

    def remap_entity(entity, char_to_tok, n_tokens):
        start_char = entity["start_char"]
        end_char = entity["end_char"]

        start_tok = char_to_tok.get(start_char)
        if start_tok is None:
            for char_idx in range(start_char, min(start_char + 10, end_char)):
                if char_idx in char_to_tok:
                    start_tok = char_to_tok[char_idx]
                    break
        if start_tok is None:
            start_tok = 0

        end_tok = char_to_tok.get(end_char - 1)
        if end_tok is None:
            for char_idx in range(end_char - 1, max(end_char - 10, start_char) - 1, -1):
                if char_idx in char_to_tok:
                    end_tok = char_to_tok[char_idx]
                    break
        if end_tok is None:
            end_tok = start_tok

        end_tok = min(end_tok + 1, n_tokens)
        return start_tok, end_tok

    with dst_path.open("w", encoding="utf-8") as handle:
        for record in records:
            parsed = sent_to_dep.get(record["sentence"])
            if parsed is None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            spacy_tokens, dep_heads, dep_labels, char_to_tok = parsed
            record["tokens"] = spacy_tokens
            record["dep_heads"] = dep_heads
            record["dep_labels"] = dep_labels

            for entity_key in ("entity1", "entity2"):
                start_tok, end_tok = remap_entity(record[entity_key], char_to_tok, len(spacy_tokens))
                record[entity_key]["start_tok"] = start_tok
                record[entity_key]["end_tok"] = end_tok

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"  wrote {total} records -> {dst_path.name}  ({elapsed:.1f}s)")
    return total


def main():
    print("=" * 60)
    print("  Dependency parsing stage")
    print("=" * 60)

    nlp = load_nlp()
    total_records = 0
    t0 = time.time()

    for filename in FILES:
        src_path = PROCESSED_DIR / filename
        dst_path = PARSED_DIR / filename
        if not src_path.exists():
            print(f"  [WARN] missing file: {filename}")
            continue
        total_records += process_file(nlp, src_path, dst_path)

    elapsed = time.time() - t0
    print(f"\nDone. Parsed {total_records} records in {elapsed:.1f}s")
    print(f"Output dir: {PARSED_DIR}")
    print("Next: run src/preprocessing/entity_coarsen.py")


if __name__ == "__main__":
    main()