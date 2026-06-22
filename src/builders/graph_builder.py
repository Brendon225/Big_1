"""
Build graph-side tensors from JSONL records.

This module is intentionally model-agnostic. It only converts one record into:
- adjacency matrix;
- dependency-type id matrix;
- node-level metadata (entity node indices, node char spans).

Both raw and entity-coarsened dependency views are supported so that Stage-3
models can run a clean raw-graph vs coarse-graph ablation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAD_LABEL = "<PAD>"
UNK_LABEL = "<UNK>"
SUPPORTED_GRAPH_VIEWS = {"raw", "coarse"}


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


def compute_raw_node_char_spans(record: Dict) -> List[Tuple[int, int]]:
    """
    Compute character spans for raw dependency nodes.

    Raw graph nodes correspond to parser tokens. Spans are recovered by
    left-to-right string matching against the sentence, matching the existing
    coarse-node alignment convention.
    """
    sentence = str(record.get("sentence", ""))
    tokens = list(record.get("tokens", []))

    spans: List[Tuple[int, int]] = []
    cursor = 0
    for token_text in tokens:
        span = _find_span(sentence, str(token_text), cursor)
        if span[0] >= 0 and span[1] >= span[0]:
            cursor = max(cursor, span[1])
        spans.append(span)

    return spans


def compute_node_char_spans(record: Dict, dep_view: str) -> List[Tuple[int, int]]:
    """Compute node character spans for the selected dependency graph view."""
    if dep_view == "raw":
        return compute_raw_node_char_spans(record)
    if dep_view == "coarse":
        return compute_coarse_node_char_spans(record)
    raise ValueError(
        f"Unsupported dep_view={dep_view!r}. "
        f"Expected one of {sorted(SUPPORTED_GRAPH_VIEWS)}."
    )


class GraphBuilder:
    """
    Build graph features from one record.

    Input fields expected for dep_view="raw":
    - tokens / dep_heads / dep_labels
    - entity1.start_tok / entity2.start_tok

    Input fields expected for dep_view="coarse":
    - coarse_tokens / coarse_heads / coarse_labels
    - coarse_e1_idx / coarse_e2_idx
    """

    def __init__(self, dep_type_vocab: Dict[str, int], dep_view: str = "coarse"):
        if PAD_LABEL not in dep_type_vocab or UNK_LABEL not in dep_type_vocab:
            raise ValueError("dep_type_vocab must contain <PAD> and <UNK>.")
        dep_view = str(dep_view).lower()
        if dep_view not in SUPPORTED_GRAPH_VIEWS:
            raise ValueError(
                f"Unsupported dep_view={dep_view!r}. "
                f"Expected one of {sorted(SUPPORTED_GRAPH_VIEWS)}."
            )
        self.dep_type_vocab = dep_type_vocab
        self.pad_id = dep_type_vocab[PAD_LABEL]
        self.unk_id = dep_type_vocab[UNK_LABEL]
        self.dep_view = dep_view

    @classmethod
    def from_vocab_path(cls, vocab_path: str | Path, dep_view: str = "coarse") -> "GraphBuilder":
        return cls(load_dep_type_vocab(vocab_path), dep_view=dep_view)

    def _dep_label_to_id(self, label: str) -> int:
        return self.dep_type_vocab.get(str(label), self.unk_id)

    @staticmethod
    def _select_graph_fields(record: Dict, dep_view: str) -> Tuple[List, List, List, int, int]:
        if dep_view == "raw":
            tokens = list(record.get("tokens", []))
            heads = list(record.get("dep_heads", []))
            labels = list(record.get("dep_labels", []))
            entity1 = record.get("entity1", {}) or {}
            entity2 = record.get("entity2", {}) or {}
            e1_idx = _as_int(entity1.get("start_tok"), default=-1)
            e2_idx = _as_int(entity2.get("start_tok"), default=-1)
            return tokens, heads, labels, e1_idx, e2_idx

        if dep_view == "coarse":
            tokens = list(record.get("coarse_tokens", []))
            heads = list(record.get("coarse_heads", []))
            labels = list(record.get("coarse_labels", []))
            e1_idx = _as_int(record.get("coarse_e1_idx"), default=-1)
            e2_idx = _as_int(record.get("coarse_e2_idx"), default=-1)
            return tokens, heads, labels, e1_idx, e2_idx

        raise ValueError(
            f"Unsupported dep_view={dep_view!r}. "
            f"Expected one of {sorted(SUPPORTED_GRAPH_VIEWS)}."
        )

    def build_from_record(self, record: Dict, dep_view: str | None = None) -> GraphFeatures:
        dep_view = self.dep_view if dep_view is None else str(dep_view).lower()
        tokens, heads, labels, e1_idx, e2_idx = self._select_graph_fields(record, dep_view)

        node_count = len(tokens)
        if len(heads) != node_count or len(labels) != node_count:
            raise ValueError(
                f"{dep_view} graph field length mismatch "
                f"(tokens={node_count}, heads={len(heads)}, labels={len(labels)})"
            )

        adj_matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]
        dep_type_ids = [
            [self.pad_id for _ in range(node_count)] for _ in range(node_count)
        ]

        for dep_idx in range(node_count):
            head_idx = _as_int(heads[dep_idx], default=-1)
            if head_idx < 0 or head_idx >= node_count:
                continue
            if head_idx == dep_idx:
                # Skip ROOT/self-loop.
                continue

            label_id = self._dep_label_to_id(labels[dep_idx])

            adj_matrix[head_idx][dep_idx] = 1
            adj_matrix[dep_idx][head_idx] = 1
            dep_type_ids[head_idx][dep_idx] = label_id
            dep_type_ids[dep_idx][head_idx] = label_id

        if not (0 <= e1_idx < node_count):
            raise ValueError(
                f"Invalid {dep_view} e1_node_idx={e1_idx} for node_count={node_count}"
            )
        if not (0 <= e2_idx < node_count):
            raise ValueError(
                f"Invalid {dep_view} e2_node_idx={e2_idx} for node_count={node_count}"
            )

        node_char_spans = compute_node_char_spans(record, dep_view=dep_view)
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
