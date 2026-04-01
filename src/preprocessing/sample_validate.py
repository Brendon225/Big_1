#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sample_validate.py

Stream random samples from data/processed, data/parsed and data/coarsened,
then validate structural integrity, cross-stage consistency and optional
stage reproducibility checks.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.entity_coarsen import coarsen_record

try:
    from src.preprocessing.dep_parser import load_nlp as load_dep_nlp
except Exception:
    load_dep_nlp = None

STAGE_DIRS = {
    "processed": ROOT / "data" / "processed",
    "parsed": ROOT / "data" / "parsed",
    "coarsened": ROOT / "data" / "coarsened",
}

REQUIRED_FIELDS = {
    "id",
    "dataset",
    "split",
    "sentence",
    "tokens",
    "entity1",
    "entity2",
    "relation",
    "is_positive",
}
REQUIRED_ENTITY_FIELDS = {
    "text",
    "type",
    "start_char",
    "end_char",
    "start_tok",
    "end_tok",
}
REQUIRED_PARSED_FIELDS = {"dep_heads", "dep_labels"}
REQUIRED_COARSE_FIELDS = {
    "coarse_tokens",
    "coarse_heads",
    "coarse_labels",
    "coarse_e1_idx",
    "coarse_e2_idx",
    "coarse_node_types",
}
ALLOWED_NODE_TYPES = {"normal", "entity1", "entity2"}
DUAL_ENTITY_NODE = "entity1+entity2"
ALLOWED_NODE_TYPES.add(DUAL_ENTITY_NODE)
WORD_RE = re.compile(r"[A-Za-z0-9]+")
DATASET_ORDER = {"CDR": 0, "ChemProt": 1, "DDI": 2}
SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}


@dataclass
class Issue:
    severity: str
    stage: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
        }


class ReservoirSampler:
    def __init__(self, size: int, rng: random.Random):
        self.size = max(0, size)
        self.rng = rng
        self.items: list[dict[str, Any]] = []
        self.seen = 0

    def add(self, item: dict[str, Any]) -> None:
        if self.size <= 0:
            return
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        idx = self.rng.randrange(self.seen)
        if idx < self.size:
            self.items[idx] = item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streamed random-sample validator for processed/parsed/coarsened JSONL files."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset prefixes to validate, e.g. CDR ChemProt DDI",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional exact JSON file names, e.g. CDR_train.json DDI_dev.json",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Uniform random sample size per processed file.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=5,
        help="Extra sample size per positivity class (positive/negative/unknown).",
    )
    parser.add_argument(
        "--per-relation",
        type=int,
        default=3,
        help="Extra sample size per relation label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to data/sample_validation/run_<timestamp>.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N non-empty lines while scanning large files.",
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="Re-run SciSpaCy parsing for sampled records and compare with parsed outputs.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with code 1 if any sampled record has at least one error.",
    )
    return parser.parse_args()


def file_sort_key(filename: str) -> tuple[int, int, str]:
    stem = Path(filename).stem
    if "_" not in stem:
        return (99, 99, filename)
    dataset, split = stem.split("_", 1)
    return (
        DATASET_ORDER.get(dataset, 99),
        SPLIT_ORDER.get(split, 99),
        filename,
    )


def discover_files(args: argparse.Namespace) -> list[str]:
    processed_dir = STAGE_DIRS["processed"]
    files = [
        path.name
        for path in processed_dir.glob("*.json")
        if path.name != "dataset_stats.json"
    ]
    files.sort(key=file_sort_key)

    if args.datasets:
        dataset_set = {item.strip() for item in args.datasets if item.strip()}
        files = [name for name in files if name.split("_", 1)[0] in dataset_set]

    if args.files:
        requested = {item.strip() for item in args.files if item.strip()}
        missing = sorted(requested - set(files))
        if missing:
            print("[WARN] Requested files not found in data/processed:")
            for name in missing:
                print(f"  - {name}")
        files = [name for name in files if name in requested]

    return files


def build_output_dir(output_dir_arg: str) -> Path:
    if output_dir_arg:
        out_dir = Path(output_dir_arg)
    else:
        stamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out_dir = ROOT / "data" / "sample_validation" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def word_seq(text: Any) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def compact_text(text: Any) -> str:
    return "".join(word_seq(text))


def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return True
    idx = 0
    for tok in haystack:
        if tok == needle[idx]:
            idx += 1
            if idx == len(needle):
                return True
    return False


def loose_text_match(expected: Any, observed: Any) -> bool:
    exp_words = word_seq(expected)
    obs_words = word_seq(observed)
    if not exp_words or not obs_words:
        return compact_text(expected) == compact_text(observed)
    if exp_words == obs_words:
        return True
    if is_subsequence(exp_words, obs_words) or is_subsequence(obs_words, exp_words):
        return True
    exp_set = set(exp_words)
    obs_set = set(obs_words)
    overlap = len(exp_set & obs_set)
    min_size = min(len(exp_set), len(obs_set))
    return min_size > 0 and overlap >= max(1, min_size - 1)


def preview(text: Any, limit: int = 120) -> str:
    raw = str(text).replace("\n", " ").strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3] + "..."


def add_issue(
    issues: list[Issue],
    severity: str,
    stage: str,
    code: str,
    message: str,
) -> None:
    issues.append(Issue(severity=severity, stage=stage, code=code, message=message))


def classify_issues(issues: list[Issue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "fail"
    if issues:
        return "warn"
    return "pass"


def find_first_list_mismatch(left: list[Any], right: list[Any]) -> str:
    if len(left) != len(right):
        return f"length {len(left)} != {len(right)}"
    for idx, (l_item, r_item) in enumerate(zip(left, right)):
        if l_item != r_item:
            return f"index {idx}: {l_item!r} != {r_item!r}"
    return "identical"


def validate_entity(
    rec: dict[str, Any],
    ent_key: str,
    stage: str,
    issues: list[Issue],
) -> tuple[int | None, int | None, int | None, int | None]:
    entity = rec.get(ent_key)
    if not isinstance(entity, dict):
        add_issue(issues, "error", stage, f"{ent_key}_not_dict", f"{ent_key} is not a dict")
        return (None, None, None, None)

    missing = sorted(REQUIRED_ENTITY_FIELDS - set(entity.keys()))
    if missing:
        add_issue(
            issues,
            "error",
            stage,
            f"{ent_key}_missing_fields",
            f"{ent_key} is missing fields: {missing}",
        )
        return (None, None, None, None)

    text = entity.get("text")
    ent_type = entity.get("type")
    if not isinstance(text, str) or not text.strip():
        add_issue(issues, "error", stage, f"{ent_key}_bad_text", f"{ent_key}.text is empty or not a string")
    if not isinstance(ent_type, str) or not ent_type.strip():
        add_issue(issues, "error", stage, f"{ent_key}_bad_type", f"{ent_key}.type is empty or not a string")

    int_fields = {}
    for field_name in ("start_char", "end_char", "start_tok", "end_tok"):
        value = entity.get(field_name)
        if not isinstance(value, int):
            add_issue(
                issues,
                "error",
                stage,
                f"{ent_key}_{field_name}_not_int",
                f"{ent_key}.{field_name} is not an int: {value!r}",
            )
            int_fields[field_name] = None
        else:
            int_fields[field_name] = value

    sentence = rec.get("sentence")
    tokens = rec.get("tokens")
    sc = int_fields["start_char"]
    ec = int_fields["end_char"]
    st = int_fields["start_tok"]
    et = int_fields["end_tok"]

    if isinstance(sentence, str) and sc is not None and ec is not None:
        if not (0 <= sc < ec <= len(sentence)):
            add_issue(
                issues,
                "error",
                stage,
                f"{ent_key}_char_span_oob",
                f"{ent_key} char span [{sc}, {ec}) is outside sentence length {len(sentence)}",
            )

    if isinstance(tokens, list) and st is not None and et is not None:
        if not (0 <= st < et <= len(tokens)):
            add_issue(
                issues,
                "error",
                stage,
                f"{ent_key}_tok_span_oob",
                f"{ent_key} token span [{st}, {et}) is outside token length {len(tokens)}",
            )

    char_ok = None
    tok_ok = None
    if isinstance(sentence, str) and sc is not None and ec is not None and 0 <= sc < ec <= len(sentence):
        char_slice = sentence[sc:ec]
        char_ok = loose_text_match(text, char_slice)
    if isinstance(tokens, list) and st is not None and et is not None and 0 <= st < et <= len(tokens):
        token_slice = " ".join(str(tok) for tok in tokens[st:et])
        tok_ok = loose_text_match(text, token_slice)

    if char_ok is False and tok_ok is False:
        add_issue(
            issues,
            "error",
            stage,
            f"{ent_key}_text_alignment_fail",
            f"{ent_key} text does not align with either sentence span or token span",
        )
    elif char_ok is False or tok_ok is False:
        add_issue(
            issues,
            "warning",
            stage,
            f"{ent_key}_text_alignment_warn",
            f"{ent_key} text aligns weakly with sentence/token spans",
        )

    return (sc, ec, st, et)


def validate_processed_record(rec: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    stage = "processed"

    missing = sorted(REQUIRED_FIELDS - set(rec.keys()))
    if missing:
        add_issue(issues, "error", stage, "missing_fields", f"Record is missing fields: {missing}")
        return issues

    for field_name in ("id", "dataset", "split", "relation"):
        value = rec.get(field_name)
        if not isinstance(value, str) or not value.strip():
            add_issue(issues, "error", stage, f"{field_name}_bad_type", f"{field_name} is empty or not a string")

    rec_id = rec.get("id")
    dataset = rec.get("dataset")
    split = rec.get("split")
    if isinstance(rec_id, str) and isinstance(dataset, str) and isinstance(split, str):
        expected_prefix = f"{dataset}_{split}_"
        if not rec_id.startswith(expected_prefix):
            add_issue(
                issues,
                "error",
                stage,
                "id_prefix_mismatch",
                f"id {rec_id!r} does not start with expected prefix {expected_prefix!r}",
            )

    sentence = rec.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        add_issue(issues, "error", stage, "bad_sentence", "sentence is empty or not a string")

    tokens = rec.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        add_issue(issues, "error", stage, "bad_tokens", "tokens is empty or not a list")
    elif not all(isinstance(tok, str) for tok in tokens):
        add_issue(issues, "error", stage, "tokens_not_str", "tokens contains non-string items")

    is_positive = rec.get("is_positive")
    if not isinstance(is_positive, bool):
        add_issue(issues, "error", stage, "is_positive_not_bool", f"is_positive is not bool: {is_positive!r}")
    else:
        relation = rec.get("relation")
        dataset = rec.get("dataset")
        if dataset in {"CDR", "ChemProt"} and is_positive is False:
            add_issue(
                issues,
                "error",
                stage,
                "unexpected_negative",
                f"{dataset} should not contain negative relation instances in current pipeline",
            )
        if dataset == "DDI":
            if relation == "DDI-false" and is_positive is not False:
                add_issue(
                    issues,
                    "error",
                    stage,
                    "ddi_false_positive_mismatch",
                    "DDI-false must have is_positive == False",
                )
            if relation != "DDI-false" and is_positive is not True:
                add_issue(
                    issues,
                    "error",
                    stage,
                    "ddi_positive_label_mismatch",
                    "Positive DDI relations must have is_positive == True",
                )

    e1 = validate_entity(rec, "entity1", stage, issues)
    e2 = validate_entity(rec, "entity2", stage, issues)

    if e1[2] is not None and e1[3] is not None and e2[2] is not None and e2[3] is not None:
        overlap = not (e1[3] <= e2[2] or e2[3] <= e1[2])
        if overlap:
            add_issue(
                issues,
                "warning",
                stage,
                "entity_token_overlap",
                "entity1 and entity2 token spans overlap",
            )

    return issues


def validate_dependency_fields(rec: dict[str, Any], stage: str) -> list[Issue]:
    issues: list[Issue] = []
    missing = sorted(REQUIRED_PARSED_FIELDS - set(rec.keys()))
    if missing:
        add_issue(issues, "error", stage, "missing_dep_fields", f"Missing dependency fields: {missing}")
        return issues

    tokens = rec.get("tokens")
    dep_heads = rec.get("dep_heads")
    dep_labels = rec.get("dep_labels")

    if not isinstance(dep_heads, list) or not isinstance(dep_labels, list):
        add_issue(issues, "error", stage, "bad_dep_types", "dep_heads or dep_labels is not a list")
        return issues
    if not isinstance(tokens, list):
        add_issue(issues, "error", stage, "tokens_not_list", "tokens is not a list")
        return issues

    if not (len(tokens) == len(dep_heads) == len(dep_labels)):
        add_issue(
            issues,
            "error",
            stage,
            "dep_length_mismatch",
            f"len(tokens)={len(tokens)}, len(dep_heads)={len(dep_heads)}, len(dep_labels)={len(dep_labels)}",
        )

    root_count = 0
    for idx, head in enumerate(dep_heads):
        if not isinstance(head, int):
            add_issue(issues, "error", stage, "dep_head_not_int", f"dep_heads[{idx}] is not int: {head!r}")
            continue
        if not (0 <= head < len(tokens)):
            add_issue(
                issues,
                "error",
                stage,
                "dep_head_oob",
                f"dep_heads[{idx}]={head} is outside token length {len(tokens)}",
            )
        if head == idx:
            root_count += 1

    if not all(isinstance(label, str) and label for label in dep_labels):
        add_issue(issues, "error", stage, "dep_label_bad_type", "dep_labels contains empty or non-string values")

    if root_count == 0 and isinstance(tokens, list):
        add_issue(issues, "error", stage, "no_root", "No root token found (head[i] == i)")

    return issues


def remap_entity_from_char(
    entity: dict[str, Any],
    char_to_tok: dict[int, int],
    n_tokens: int,
) -> tuple[int, int]:
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


def load_reparse_nlp():
    if load_dep_nlp is None:
        raise RuntimeError("Could not import load_nlp from src.preprocessing.dep_parser")
    try:
        return load_dep_nlp()
    except SystemExit as exc:
        raise RuntimeError("SciSpaCy model loading failed") from exc


def build_reparse_cache(
    nlp: Any,
    sample_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    unique_sentences = []
    seen = set()
    for sample in sample_map.values():
        rec = sample.get("processed")
        if not isinstance(rec, dict):
            continue
        sentence = rec.get("sentence")
        if isinstance(sentence, str) and sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)

    cache: dict[str, dict[str, Any]] = {}
    if not unique_sentences:
        return cache

    for doc, sentence in zip(nlp.pipe(unique_sentences, batch_size=128), unique_sentences):
        tokens = [token.text for token in doc]
        heads = [token.head.i for token in doc]
        labels = [token.dep_ for token in doc]
        char_to_tok = {}
        for token in doc:
            for char_idx in range(token.idx, token.idx + len(token.text)):
                char_to_tok[char_idx] = token.i
        cache[sentence] = {
            "tokens": tokens,
            "dep_heads": heads,
            "dep_labels": labels,
            "char_to_tok": char_to_tok,
        }
    return cache


def validate_parsed_record(
    rec: dict[str, Any],
    reparse_cache: dict[str, dict[str, Any]] | None = None,
) -> list[Issue]:
    issues = validate_processed_record(rec)
    stage = "parsed"
    issues.extend(validate_dependency_fields(rec, stage))

    if reparse_cache:
        sentence = rec.get("sentence")
        parsed = reparse_cache.get(sentence)
        if parsed is None:
            add_issue(issues, "error", stage, "reparse_missing", "Sentence not found in reparse cache")
            return issues

        if rec.get("tokens") != parsed["tokens"]:
            add_issue(
                issues,
                "error",
                stage,
                "reparse_tokens_mismatch",
                f"Reparsed tokens mismatch: {find_first_list_mismatch(rec.get('tokens', []), parsed['tokens'])}",
            )
        if rec.get("dep_heads") != parsed["dep_heads"]:
            add_issue(
                issues,
                "error",
                stage,
                "reparse_heads_mismatch",
                f"Reparsed dep_heads mismatch: {find_first_list_mismatch(rec.get('dep_heads', []), parsed['dep_heads'])}",
            )
        if rec.get("dep_labels") != parsed["dep_labels"]:
            add_issue(
                issues,
                "error",
                stage,
                "reparse_labels_mismatch",
                f"Reparsed dep_labels mismatch: {find_first_list_mismatch(rec.get('dep_labels', []), parsed['dep_labels'])}",
            )

        for ent_key in ("entity1", "entity2"):
            entity = rec.get(ent_key)
            if isinstance(entity, dict) and {"start_char", "end_char"} <= set(entity.keys()):
                exp_start, exp_end = remap_entity_from_char(entity, parsed["char_to_tok"], len(parsed["tokens"]))
                if entity.get("start_tok") != exp_start or entity.get("end_tok") != exp_end:
                    add_issue(
                        issues,
                        "error",
                        stage,
                        f"{ent_key}_reparse_span_mismatch",
                        (
                            f"{ent_key} token span [{entity.get('start_tok')}, {entity.get('end_tok')}) "
                            f"!= reparsed [{exp_start}, {exp_end})"
                        ),
                    )

    return issues


def validate_coarse_fields(rec: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    stage = "coarsened"
    missing = sorted(REQUIRED_COARSE_FIELDS - set(rec.keys()))
    if missing:
        add_issue(issues, "error", stage, "missing_coarse_fields", f"Missing coarse fields: {missing}")
        return issues

    coarse_tokens = rec.get("coarse_tokens")
    coarse_heads = rec.get("coarse_heads")
    coarse_labels = rec.get("coarse_labels")
    coarse_node_types = rec.get("coarse_node_types")
    coarse_e1_idx = rec.get("coarse_e1_idx")
    coarse_e2_idx = rec.get("coarse_e2_idx")
    coarse_status = rec.get("coarse_status", "")

    def node_has_role(node_type: str, role: str) -> bool:
        if role == "entity1":
            return node_type in {"entity1", DUAL_ENTITY_NODE}
        if role == "entity2":
            return node_type in {"entity2", DUAL_ENTITY_NODE}
        return node_type == "normal"

    if not all(isinstance(field, list) for field in (coarse_tokens, coarse_heads, coarse_labels, coarse_node_types)):
        add_issue(issues, "error", stage, "bad_coarse_types", "One or more coarse fields is not a list")
        return issues

    coarse_n = len(coarse_tokens)
    if not (coarse_n == len(coarse_heads) == len(coarse_labels) == len(coarse_node_types)):
        add_issue(
            issues,
            "error",
            stage,
            "coarse_length_mismatch",
            (
                f"len(coarse_tokens)={len(coarse_tokens)}, len(coarse_heads)={len(coarse_heads)}, "
                f"len(coarse_labels)={len(coarse_labels)}, len(coarse_node_types)={len(coarse_node_types)}"
            ),
        )

    for idx, head in enumerate(coarse_heads):
        if not isinstance(head, int):
            add_issue(issues, "error", stage, "coarse_head_not_int", f"coarse_heads[{idx}] is not int: {head!r}")
            continue
        if not (0 <= head < coarse_n):
            add_issue(
                issues,
                "error",
                stage,
                "coarse_head_oob",
                f"coarse_heads[{idx}]={head} is outside coarse length {coarse_n}",
            )

    if not all(isinstance(label, str) and label for label in coarse_labels):
        add_issue(issues, "error", stage, "coarse_label_bad_type", "coarse_labels contains empty or non-string values")

    if not all(isinstance(tok, str) for tok in coarse_tokens):
        add_issue(issues, "error", stage, "coarse_token_bad_type", "coarse_tokens contains non-string values")

    if not all(isinstance(node_type, str) for node_type in coarse_node_types):
        add_issue(issues, "error", stage, "coarse_node_type_bad_type", "coarse_node_types contains non-string values")
    else:
        bad_types = sorted({node_type for node_type in coarse_node_types if node_type not in ALLOWED_NODE_TYPES})
        if bad_types:
            add_issue(
                issues,
                "error",
                stage,
                "coarse_node_type_invalid",
                f"Invalid node types: {bad_types}",
            )

    for idx_name, idx_value in (("coarse_e1_idx", coarse_e1_idx), ("coarse_e2_idx", coarse_e2_idx)):
        if not isinstance(idx_value, int):
            add_issue(issues, "error", stage, f"{idx_name}_not_int", f"{idx_name} is not int: {idx_value!r}")
        elif not (0 <= idx_value < coarse_n):
            add_issue(
                issues,
                "error",
                stage,
                f"{idx_name}_oob",
                f"{idx_name}={idx_value} is outside coarse length {coarse_n}",
            )

    if isinstance(coarse_e1_idx, int) and 0 <= coarse_e1_idx < coarse_n:
        if not node_has_role(coarse_node_types[coarse_e1_idx], "entity1"):
            add_issue(
                issues,
                "error",
                stage,
                "coarse_e1_node_type_mismatch",
                f"coarse_node_types[{coarse_e1_idx}] is {coarse_node_types[coarse_e1_idx]!r}, expected 'entity1'",
            )
    if isinstance(coarse_e2_idx, int) and 0 <= coarse_e2_idx < coarse_n:
        if not node_has_role(coarse_node_types[coarse_e2_idx], "entity2"):
            add_issue(
                issues,
                "error",
                stage,
                "coarse_e2_node_type_mismatch",
                f"coarse_node_types[{coarse_e2_idx}] is {coarse_node_types[coarse_e2_idx]!r}, expected 'entity2'",
            )

    e1_count = sum(1 for node_type in coarse_node_types if node_has_role(node_type, "entity1"))
    e2_count = sum(1 for node_type in coarse_node_types if node_has_role(node_type, "entity2"))
    if e1_count != 1:
        add_issue(
            issues,
            "error",
            stage,
            "entity1_count_invalid",
            f"coarse_node_types contains {e1_count} entity1 nodes",
        )
    if e2_count != 1:
        add_issue(
            issues,
            "error",
            stage,
            "entity2_count_invalid",
            f"coarse_node_types contains {e2_count} entity2 nodes",
        )

    root_count = sum(1 for idx, head in enumerate(coarse_heads) if isinstance(head, int) and head == idx)
    if coarse_n > 0 and root_count == 0:
        add_issue(issues, "error", stage, "coarse_no_root", "No coarse root token found (head[i] == i)")

    entity1 = rec.get("entity1", {})
    entity2 = rec.get("entity2", {})
    if isinstance(coarse_e1_idx, int) and isinstance(entity1, dict) and 0 <= coarse_e1_idx < coarse_n:
        if not loose_text_match(entity1.get("text"), coarse_tokens[coarse_e1_idx]):
            add_issue(
                issues,
                "error",
                stage,
                "coarse_e1_text_mismatch",
                "coarse entity1 node text does not match entity1.text",
            )
    if isinstance(coarse_e2_idx, int) and isinstance(entity2, dict) and 0 <= coarse_e2_idx < coarse_n:
        if not loose_text_match(entity2.get("text"), coarse_tokens[coarse_e2_idx]):
            add_issue(
                issues,
                "error",
                stage,
                "coarse_e2_text_mismatch",
                "coarse entity2 node text does not match entity2.text",
            )

    tokens = rec.get("tokens")
    if isinstance(tokens, list) and coarse_n > len(tokens):
        add_issue(
            issues,
            "error",
            stage,
            "coarse_size_increase",
            f"coarse node count {coarse_n} is larger than original token count {len(tokens)}",
        )

    return issues


def validate_coarsened_record(
    rec: dict[str, Any],
    parsed_rec: dict[str, Any] | None = None,
) -> list[Issue]:
    issues = validate_processed_record(rec)
    issues.extend(validate_dependency_fields(rec, "coarsened"))
    issues.extend(validate_coarse_fields(rec))

    if parsed_rec and isinstance(parsed_rec.get("tokens"), list) and isinstance(rec.get("coarse_tokens"), list):
        e1 = rec.get("entity1", {})
        e2 = rec.get("entity2", {})
        overlap_passthrough = rec.get("coarse_status") == "overlap_passthrough"
        if isinstance(e1, dict) and isinstance(e2, dict) and not overlap_passthrough:
            e1_span = e1.get("end_tok", 0) - e1.get("start_tok", 0)
            e2_span = e2.get("end_tok", 0) - e2.get("start_tok", 0)
            if isinstance(e1_span, int) and isinstance(e2_span, int):
                expected = len(parsed_rec["tokens"]) - max(0, e1_span - 1) - max(0, e2_span - 1)
                if len(rec["coarse_tokens"]) != expected:
                    add_issue(
                        issues,
                        "error",
                        "coarsened",
                        "coarse_size_unexpected",
                        f"Expected coarse length {expected}, got {len(rec['coarse_tokens'])}",
                    )

        expected_rec = coarsen_record(copy.deepcopy(parsed_rec))
        for field_name in sorted(REQUIRED_COARSE_FIELDS):
            if rec.get(field_name) != expected_rec.get(field_name):
                left = rec.get(field_name)
                right = expected_rec.get(field_name)
                if isinstance(left, list) and isinstance(right, list):
                    detail = find_first_list_mismatch(left, right)
                else:
                    detail = f"{left!r} != {right!r}"
                add_issue(
                    issues,
                    "error",
                    "coarsened",
                    f"recoarsen_{field_name}_mismatch",
                    f"{field_name} differs from recomputed result: {detail}",
                )

    return issues


def compare_entities(
    left: dict[str, Any],
    right: dict[str, Any],
    stage: str,
    issues: list[Issue],
    include_tok_spans: bool,
) -> None:
    compare_fields = ["text", "type", "start_char", "end_char"]
    if include_tok_spans:
        compare_fields.extend(["start_tok", "end_tok"])
    for ent_key in ("entity1", "entity2"):
        left_ent = left.get(ent_key, {})
        right_ent = right.get(ent_key, {})
        for field_name in compare_fields:
            if left_ent.get(field_name) != right_ent.get(field_name):
                add_issue(
                    issues,
                    "error",
                    stage,
                    f"{ent_key}_{field_name}_drift",
                    (
                        f"{ent_key}.{field_name} differs across stages: "
                        f"{left_ent.get(field_name)!r} != {right_ent.get(field_name)!r}"
                    ),
                )


def validate_cross_stage_alignment(sample: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    processed = sample.get("processed")
    parsed = sample.get("parsed")
    coarsened = sample.get("coarsened")

    if processed is None:
        add_issue(issues, "error", "alignment", "processed_missing", "Processed record is missing")
        return issues
    if parsed is None:
        add_issue(issues, "error", "alignment", "parsed_missing", "Parsed record is missing")
    if coarsened is None:
        add_issue(issues, "error", "alignment", "coarsened_missing", "Coarsened record is missing")

    if processed and parsed:
        for field_name in ("id", "dataset", "split", "sentence", "relation", "is_positive"):
            if processed.get(field_name) != parsed.get(field_name):
                add_issue(
                    issues,
                    "error",
                    "alignment",
                    f"processed_parsed_{field_name}_drift",
                    (
                        f"{field_name} differs between processed and parsed: "
                        f"{processed.get(field_name)!r} != {parsed.get(field_name)!r}"
                    ),
                )
        compare_entities(processed, parsed, "alignment", issues, include_tok_spans=False)

    if parsed and coarsened:
        for field_name in ("id", "dataset", "split", "sentence", "relation", "is_positive"):
            if parsed.get(field_name) != coarsened.get(field_name):
                add_issue(
                    issues,
                    "error",
                    "alignment",
                    f"parsed_coarsened_{field_name}_drift",
                    (
                        f"{field_name} differs between parsed and coarsened: "
                        f"{parsed.get(field_name)!r} != {coarsened.get(field_name)!r}"
                    ),
                )
        compare_entities(parsed, coarsened, "alignment", issues, include_tok_spans=True)

        for field_name in ("tokens", "dep_heads", "dep_labels"):
            if parsed.get(field_name) != coarsened.get(field_name):
                left = parsed.get(field_name, [])
                right = coarsened.get(field_name, [])
                if isinstance(left, list) and isinstance(right, list):
                    detail = find_first_list_mismatch(left, right)
                else:
                    detail = f"{left!r} != {right!r}"
                add_issue(
                    issues,
                    "error",
                    "alignment",
                    f"parsed_coarsened_{field_name}_drift",
                    f"{field_name} differs between parsed and coarsened: {detail}",
                )

    return issues


def sample_processed_file(
    path: Path,
    sample_size: int,
    per_class: int,
    per_relation: int,
    rng: random.Random,
    progress_every: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    global_sampler = ReservoirSampler(sample_size, rng)
    class_samplers: dict[str, ReservoirSampler] = {}
    relation_samplers: dict[str, ReservoirSampler] = {}
    relation_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    parse_errors: list[dict[str, Any]] = []
    total_records = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            total_records += 1
            if progress_every > 0 and total_records % progress_every == 0:
                print(f"    processed scan {path.name}: {total_records} records")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(
                    {
                        "line_no": line_no,
                        "error": str(exc),
                        "preview": preview(line),
                    }
                )
                continue

            relation = str(record.get("relation", "<missing>"))
            if record.get("is_positive") is True:
                class_key = "positive"
            elif record.get("is_positive") is False:
                class_key = "negative"
            else:
                class_key = "unknown"

            relation_counts[relation] += 1
            class_counts[class_key] += 1

            item = {
                "record": record,
                "line_no": line_no,
            }
            global_sampler.add(item)
            if per_class > 0:
                class_samplers.setdefault(class_key, ReservoirSampler(per_class, rng)).add(item)
            if per_relation > 0:
                relation_samplers.setdefault(relation, ReservoirSampler(per_relation, rng)).add(item)

    selected: dict[str, dict[str, Any]] = {}

    def merge_selected(reason: str, sampler: ReservoirSampler) -> None:
        for item in sampler.items:
            record = item["record"]
            rec_id = str(record.get("id", f"<line:{item['line_no']}>"))
            slot = selected.setdefault(
                rec_id,
                {
                    "id": rec_id,
                    "line_no": item["line_no"],
                    "selection_reasons": [],
                    "processed": record,
                },
            )
            slot["selection_reasons"].append(reason)

    merge_selected("global", global_sampler)
    for class_key, sampler in sorted(class_samplers.items()):
        merge_selected(f"class:{class_key}", sampler)
    for relation, sampler in sorted(relation_samplers.items()):
        merge_selected(f"relation:{relation}", sampler)

    for sample in selected.values():
        sample["selection_reasons"] = sorted(set(sample["selection_reasons"]))

    meta = {
        "total_records": total_records,
        "relation_counts": dict(sorted(relation_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "parse_errors": parse_errors,
        "selected_count": len(selected),
    }
    return selected, meta


def scan_stage_file(
    path: Path,
    target_ids: set[str],
    stage: str,
    progress_every: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    parse_errors: list[dict[str, Any]] = []
    duplicate_ids: list[dict[str, Any]] = []
    total_records = 0

    if not path.exists():
        return {}, {"missing_file": True, "total_records": 0, "parse_errors": [], "duplicate_ids": []}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            total_records += 1
            if progress_every > 0 and total_records % progress_every == 0:
                print(f"    {stage} scan {path.name}: {total_records} records")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(
                    {
                        "line_no": line_no,
                        "error": str(exc),
                        "preview": preview(line),
                    }
                )
                continue

            rec_id = str(record.get("id", f"<line:{line_no}>"))
            if rec_id in target_ids:
                if rec_id in found:
                    duplicate_ids.append({"id": rec_id, "line_no": line_no})
                found[rec_id] = record

    meta = {
        "missing_file": False,
        "total_records": total_records,
        "parse_errors": parse_errors,
        "duplicate_ids": duplicate_ids,
    }
    return found, meta


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_markdown_report(
    args: argparse.Namespace,
    run_meta: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> str:
    lines = []
    lines.append("# Sample Validation Report")
    lines.append("")
    lines.append(f"- Generated: {run_meta['generated_at']}")
    lines.append(f"- Files validated: {len(file_summaries)}")
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Uniform sample size per file: {args.sample_size}")
    lines.append(f"- Extra per-class sample size: {args.per_class}")
    lines.append(f"- Extra per-relation sample size: {args.per_relation}")
    lines.append(f"- Reparse enabled: {args.reparse}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| File | Processed | Sampled | Pass | Warn | Fail | Errors | Warnings |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    total_samples = 0
    total_pass = 0
    total_warn = 0
    total_fail = 0
    total_errors = 0
    total_warnings = 0
    issue_counter: Counter[str] = Counter()

    for summary in file_summaries:
        lines.append(
            "| {file} | {processed} | {sampled} | {pass_n} | {warn_n} | {fail_n} | {err_n} | {warn_issue_n} |".format(
                file=summary["file"],
                processed=summary["processed_total"],
                sampled=summary["sampled_count"],
                pass_n=summary["status_counts"]["pass"],
                warn_n=summary["status_counts"]["warn"],
                fail_n=summary["status_counts"]["fail"],
                err_n=summary["error_count"],
                warn_issue_n=summary["warning_count"],
            )
        )
        total_samples += summary["sampled_count"]
        total_pass += summary["status_counts"]["pass"]
        total_warn += summary["status_counts"]["warn"]
        total_fail += summary["status_counts"]["fail"]
        total_errors += summary["error_count"]
        total_warnings += summary["warning_count"]
        issue_counter.update(summary["issue_codes"])

    lines.append("")
    lines.append(f"- Total sampled records: {total_samples}")
    lines.append(f"- Pass/Warn/Fail: {total_pass}/{total_warn}/{total_fail}")
    lines.append(f"- Total errors: {total_errors}")
    lines.append(f"- Total warnings: {total_warnings}")
    lines.append("")

    if issue_counter:
        lines.append("## Top Issue Codes")
        lines.append("")
        lines.append("| Code | Count |")
        lines.append("| --- | ---: |")
        for code, count in issue_counter.most_common(20):
            lines.append(f"| {code} | {count} |")
        lines.append("")

    lines.append("## Per-file Notes")
    lines.append("")
    for summary in file_summaries:
        lines.append(f"### {summary['file']}")
        lines.append("")
        lines.append(
            f"- Processed/Parsed/Coarsened totals: "
            f"{summary['processed_total']}/{summary['parsed_total']}/{summary['coarsened_total']}"
        )
        lines.append(
            f"- Relation counts: {summary['relation_counts']}"
        )
        lines.append(
            f"- Class counts: {summary['class_counts']}"
        )
        lines.append(
            f"- Selected sample count: {summary['sampled_count']}"
        )
        lines.append(
            f"- Status counts: {summary['status_counts']}"
        )
        if summary["missing_stage_files"]:
            lines.append(f"- Missing stage files: {summary['missing_stage_files']}")
        if summary["top_failed_samples"]:
            lines.append("- Failed samples:")
            for item in summary["top_failed_samples"]:
                lines.append(
                    f"  - {item['id']} ({', '.join(item['selection_reasons'])}): {item['first_issue']} "
                    f"| {item['sentence_preview']}"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    files = discover_files(args)
    if not files:
        print("[ERROR] No data files selected.")
        return 1

    output_dir = build_output_dir(args.output_dir)
    rng = random.Random(args.seed)

    print("=" * 72)
    print("Sample Validation")
    print("=" * 72)
    print(f"Output directory: {output_dir}")
    print(f"Selected files: {len(files)}")
    print(f"Seed: {args.seed}")
    print(f"Sample sizes -> global: {args.sample_size}, per-class: {args.per_class}, per-relation: {args.per_relation}")
    print(f"Reparse enabled: {args.reparse}")

    nlp = None
    if args.reparse:
        print("\nLoading SciSpaCy model for reparsing...")
        nlp = load_reparse_nlp()

    aligned_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []

    for filename in files:
        print(f"\n[File] {filename}")
        processed_path = STAGE_DIRS["processed"] / filename
        parsed_path = STAGE_DIRS["parsed"] / filename
        coarsened_path = STAGE_DIRS["coarsened"] / filename

        sample_map, processed_meta = sample_processed_file(
            path=processed_path,
            sample_size=args.sample_size,
            per_class=args.per_class,
            per_relation=args.per_relation,
            rng=rng,
            progress_every=args.progress_every,
        )
        target_ids = set(sample_map.keys())

        parsed_records, parsed_meta = scan_stage_file(
            parsed_path, target_ids, "parsed", args.progress_every
        )
        coarsened_records, coarsened_meta = scan_stage_file(
            coarsened_path, target_ids, "coarsened", args.progress_every
        )

        for rec_id, record in parsed_records.items():
            sample_map.setdefault(rec_id, {"id": rec_id, "selection_reasons": ["external"], "processed": None})
            sample_map[rec_id]["parsed"] = record
        for rec_id, record in coarsened_records.items():
            sample_map.setdefault(rec_id, {"id": rec_id, "selection_reasons": ["external"], "processed": None})
            sample_map[rec_id]["coarsened"] = record

        reparse_cache = None
        if nlp is not None:
            reparse_cache = build_reparse_cache(nlp, sample_map)

        status_counter: Counter[str] = Counter()
        issue_codes: Counter[str] = Counter()
        error_count = 0
        warning_count = 0
        top_failed_samples = []

        for rec_id in sorted(sample_map.keys()):
            sample = sample_map[rec_id]
            processed_record = sample.get("processed")
            parsed_record = sample.get("parsed", parsed_records.get(rec_id))
            coarsened_record = sample.get("coarsened", coarsened_records.get(rec_id))
            sample["parsed"] = parsed_record
            sample["coarsened"] = coarsened_record

            issues: list[Issue] = []
            if processed_record is not None:
                issues.extend(validate_processed_record(processed_record))
            else:
                add_issue(issues, "error", "processed", "missing_record", "Processed record not found for sampled id")

            if parsed_record is not None:
                issues.extend(validate_parsed_record(parsed_record, reparse_cache=reparse_cache))
            else:
                add_issue(issues, "error", "parsed", "missing_record", "Parsed record not found for sampled id")

            if coarsened_record is not None:
                issues.extend(validate_coarsened_record(coarsened_record, parsed_rec=parsed_record))
            else:
                add_issue(issues, "error", "coarsened", "missing_record", "Coarsened record not found for sampled id")

            issues.extend(validate_cross_stage_alignment(sample))

            status = classify_issues(issues)
            status_counter[status] += 1
            issue_codes.update(issue.code for issue in issues)
            error_count += sum(1 for issue in issues if issue.severity == "error")
            warning_count += sum(1 for issue in issues if issue.severity == "warning")

            sentence_source = processed_record or parsed_record or coarsened_record or {}
            sentence_preview = preview(sentence_source.get("sentence", ""))
            issue_payload = [issue.to_dict() for issue in issues]

            row = {
                "file": filename,
                "id": rec_id,
                "selection_reasons": sample.get("selection_reasons", []),
                "status": status,
                "error_count": sum(1 for issue in issues if issue.severity == "error"),
                "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
                "sentence_preview": sentence_preview,
                "issues": issue_payload,
                "records": {
                    stage_name: sample.get(stage_name)
                    for stage_name in ("processed", "parsed", "coarsened")
                    if sample.get(stage_name) is not None
                },
            }
            aligned_rows.append(row)

            if status != "pass":
                first_issue = issue_payload[0]["code"] if issue_payload else "unknown_issue"
                issue_rows.append(
                    {
                        "file": filename,
                        "id": rec_id,
                        "selection_reasons": sample.get("selection_reasons", []),
                        "status": status,
                        "sentence_preview": sentence_preview,
                        "issues": issue_payload,
                    }
                )
                if status == "fail" and len(top_failed_samples) < 5:
                    top_failed_samples.append(
                        {
                            "id": rec_id,
                            "selection_reasons": sample.get("selection_reasons", []),
                            "first_issue": first_issue,
                            "sentence_preview": sentence_preview,
                        }
                    )

        missing_stage_files = []
        if parsed_meta.get("missing_file"):
            missing_stage_files.append(str(parsed_path))
        if coarsened_meta.get("missing_file"):
            missing_stage_files.append(str(coarsened_path))

        file_summary = {
            "file": filename,
            "processed_total": processed_meta["total_records"],
            "parsed_total": parsed_meta["total_records"],
            "coarsened_total": coarsened_meta["total_records"],
            "sampled_count": len(sample_map),
            "relation_counts": processed_meta["relation_counts"],
            "class_counts": processed_meta["class_counts"],
            "status_counts": {
                "pass": status_counter["pass"],
                "warn": status_counter["warn"],
                "fail": status_counter["fail"],
            },
            "error_count": error_count,
            "warning_count": warning_count,
            "issue_codes": dict(issue_codes),
            "missing_stage_files": missing_stage_files,
            "top_failed_samples": top_failed_samples,
            "processed_parse_errors": processed_meta["parse_errors"],
            "parsed_parse_errors": parsed_meta["parse_errors"],
            "coarsened_parse_errors": coarsened_meta["parse_errors"],
            "parsed_duplicate_ids": parsed_meta["duplicate_ids"],
            "coarsened_duplicate_ids": coarsened_meta["duplicate_ids"],
        }
        file_summaries.append(file_summary)

        print(
            "  Sampled {sampled} | pass {passed} | warn {warned} | fail {failed}".format(
                sampled=file_summary["sampled_count"],
                passed=file_summary["status_counts"]["pass"],
                warned=file_summary["status_counts"]["warn"],
                failed=file_summary["status_counts"]["fail"],
            )
        )

    run_meta = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "output_dir": str(output_dir),
        "files": files,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "per_class": args.per_class,
        "per_relation": args.per_relation,
        "reparse": args.reparse,
    }

    summary_payload = {
        "run": run_meta,
        "files": file_summaries,
    }
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    aligned_jsonl_path = output_dir / "aligned_samples.jsonl"
    issues_jsonl_path = output_dir / "issues.jsonl"

    summary_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text(build_markdown_report(args, run_meta, file_summaries), encoding="utf-8")
    write_jsonl(aligned_jsonl_path, aligned_rows)
    write_jsonl(issues_jsonl_path, issue_rows)

    total_failures = sum(item["status_counts"]["fail"] for item in file_summaries)
    total_warnings = sum(item["status_counts"]["warn"] for item in file_summaries)

    print("\nReports written:")
    print(f"  - {summary_md_path}")
    print(f"  - {summary_json_path}")
    print(f"  - {aligned_jsonl_path}")
    print(f"  - {issues_jsonl_path}")
    print(f"\nFinal status: fail={total_failures}, warn={total_warnings}")

    if args.fail_on_error and total_failures > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

