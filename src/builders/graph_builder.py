"""
Build graph-side tensors from coarsened JSONL records.

This module is intentionally model-agnostic. It only converts one record into:
- adjacency matrix;
- dependency-type id matrix;
- node-level metadata (entity node indices, node char spans).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAD_LABEL = "<PAD>"
UNK_LABEL = "<UNK>"


@dataclass(frozen=True)
class GraphFeatures:
    """Graph features for one sentence-level sample."""

    adj_matrix: List[List[int]]
    dep_type_ids: List[List[int]]
    node_count: int
    e1_node_idx: int
    e2_node_idx: int
    node_char_spans: List[Tuple[int, int]]


def _as_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_dep_type_vocab(path: str | Path) -> Dict[str, int]:
    """Load dependency label vocabulary from json."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if PAD_LABEL not in payload:
        raise ValueError(f"Missing {PAD_LABEL!r} in dep vocab: {path}")
    if UNK_LABEL not in payload:
        raise ValueError(f"Missing {UNK_LABEL!r} in dep vocab: {path}")
    return {str(k): int(v) for k, v in payload.items()}


def save_dep_type_vocab(vocab: Dict[str, int], path: str | Path) -> None:
    """Persist dependency label vocabulary to json."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_dep_type_vocab(
    jsonl_paths: Sequence[str | Path],
    labels_key: str = "coarse_labels",
) -> Dict[str, int]:
    """
    Build a global dependency label vocabulary by scanning jsonl files.

    Index convention:
    - 0 -> <PAD> for non-edge/padding positions
    - 1 -> <UNK> for unseen labels
    - 2... -> sorted dependency labels observed in data
    """
    labels = set()
    for file_path in jsonl_paths:
        with Path(file_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for dep_label in record.get(labels_key, []):
                    labels.add(str(dep_label))

    vocab = {PAD_LABEL: 0, UNK_LABEL: 1}
    for dep_label in sorted(labels):
        if dep_label in vocab:
            continue
        vocab[dep_label] = len(vocab)
    return vocab


def _find_span(sentence: str, token_text: str, cursor: int) -> Tuple[int, int]:
    """
    Find token span in sentence by greedy left-to-right matching.

    Returns (-1, -1) if no match is found.
    """
    if not token_text:
        return -1, -1

    start = sentence.find(token_text, cursor)
    if start < 0 and cursor > 0:
        start = sentence.find(token_text, 0)
    if start < 0:
        normalized = " ".join(token_text.split())
        if normalized != token_text:
            start = sentence.find(normalized, cursor)
            if start < 0 and cursor > 0:
                start = sentence.find(normalized, 0)
            if start >= 0:
                return start, start + len(normalized)
        return -1, -1

    return start, start + len(token_text)


def compute_coarse_node_char_spans(record: Dict) -> List[Tuple[int, int]]:
    """
    Compute character spans for coarse nodes.

    Entity super-nodes reuse the dataset-provided entity character spans.
    Normal nodes are matched by greedy left-to-right search in `sentence`.
    """
    sentence = str(record.get("sentence", ""))
    coarse_tokens = list(record.get("coarse_tokens", []))
    node_types = list(record.get("coarse_node_types", []))
    entity1 = record.get("entity1", {}) or {}
    entity2 = record.get("entity2", {}) or {}

    e1_span = (_as_int(entity1.get("start_char")), _as_int(entity1.get("end_char")))
    e2_span = (_as_int(entity2.get("start_char")), _as_int(entity2.get("end_char")))

    spans: List[Tuple[int, int]] = []
    cursor = 0
    for idx, token_text in enumerate(coarse_tokens):
        node_type = str(node_types[idx]) if idx < len(node_types) else "normal"

        if node_type == "entity1":
            span = e1_span
        elif node_type == "entity2":
            span = e2_span
        elif node_type == "entity1+entity2":
            start = min(e1_span[0], e2_span[0])
            end = max(e1_span[1], e2_span[1])
            span = (start, end)
        else:
            span = _find_span(sentence, str(token_text), cursor)

        if span[0] >= 0 and span[1] >= span[0]:
            cursor = max(cursor, span[1])
        spans.append(span)

    return spans


class GraphBuilder:
    """
    Build graph features from one coarsened record.

    Input fields expected:
    - coarse_tokens / coarse_heads / coarse_labels
    - coarse_e1_idx / coarse_e2_idx
    """

    def __init__(self, dep_type_vocab: Dict[str, int]):
        if PAD_LABEL not in dep_type_vocab or UNK_LABEL not in dep_type_vocab:
            raise ValueError("dep_type_vocab must contain <PAD> and <UNK>.")
        self.dep_type_vocab = dep_type_vocab
        self.pad_id = dep_type_vocab[PAD_LABEL]
        self.unk_id = dep_type_vocab[UNK_LABEL]

    @classmethod
    def from_vocab_path(cls, vocab_path: str | Path) -> "GraphBuilder":
        return cls(load_dep_type_vocab(vocab_path))

    def _dep_label_to_id(self, label: str) -> int:
        return self.dep_type_vocab.get(str(label), self.unk_id)

    def build_from_record(self, record: Dict) -> GraphFeatures:
        coarse_tokens = list(record.get("coarse_tokens", []))
        coarse_heads = list(record.get("coarse_heads", []))
        coarse_labels = list(record.get("coarse_labels", []))

        node_count = len(coarse_tokens)
        if len(coarse_heads) != node_count or len(coarse_labels) != node_count:
            raise ValueError(
                "coarse_tokens/coarse_heads/coarse_labels length mismatch "
                f"(tokens={node_count}, heads={len(coarse_heads)}, labels={len(coarse_labels)})"
            )

        adj_matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]
        dep_type_ids = [
            [self.pad_id for _ in range(node_count)] for _ in range(node_count)
        ]

        for dep_idx in range(node_count):
            head_idx = _as_int(coarse_heads[dep_idx], default=-1)
            if head_idx < 0 or head_idx >= node_count:
                continue
            if head_idx == dep_idx:
                # Skip ROOT/self-loop.
                continue

            label_id = self._dep_label_to_id(coarse_labels[dep_idx])

            adj_matrix[head_idx][dep_idx] = 1
            adj_matrix[dep_idx][head_idx] = 1
            dep_type_ids[head_idx][dep_idx] = label_id
            dep_type_ids[dep_idx][head_idx] = label_id

        e1_idx = _as_int(record.get("coarse_e1_idx"), default=-1)
        e2_idx = _as_int(record.get("coarse_e2_idx"), default=-1)
        if not (0 <= e1_idx < node_count):
            raise ValueError(f"Invalid coarse_e1_idx={e1_idx} for node_count={node_count}")
        if not (0 <= e2_idx < node_count):
            raise ValueError(f"Invalid coarse_e2_idx={e2_idx} for node_count={node_count}")

        node_char_spans = compute_coarse_node_char_spans(record)
        if len(node_char_spans) != node_count:
            raise ValueError("node_char_spans length mismatch with node_count.")

        return GraphFeatures(
            adj_matrix=adj_matrix,
            dep_type_ids=dep_type_ids,
            node_count=node_count,
            e1_node_idx=e1_idx,
            e2_node_idx=e2_idx,
            node_char_spans=node_char_spans,
        )


def iter_jsonl(path: str | Path) -> Iterable[Dict]:
    """Yield records from jsonl file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
