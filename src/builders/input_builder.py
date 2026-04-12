"""
Input building utilities for baseline experiments.

This module centralizes:
- dependency sequence construction (`tree` / `sdp`);
- encoder input text construction;
- decoder target text construction.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Dict, List, Optional, Tuple


SUPPORTED_DEP_VIEWS = {"raw", "coarse"}
SUPPORTED_DEP_FORMS = {"none", "tree", "sdp"}
SUPPORTED_TARGET_MODES = {"relation_only", "natural_language"}


def _normalize_node_text(text: str) -> str:
    """Keep node text single-token-ish inside the dependency sequence."""
    return "_".join(str(text).split())


def linearize_dependency_arcs(
    tokens: List[str],
    heads: List[int],
    labels: List[str],
    window_start: int = 0,
    window_end: Optional[int] = None,
    max_arcs: Optional[int] = None,
) -> str:
    """
    Linearize dependency arcs that remain fully inside the visible window.

    Self-loops (ROOT-style arcs) and edges pointing outside the visible window
    are skipped so the dependency text stays aligned with the visible context.
    """
    if window_end is None:
        window_end = len(tokens)

    parts: List[str] = []
    limit = min(len(tokens), len(heads), len(labels), window_end)

    for dep_idx in range(window_start, limit):
        head_idx = heads[dep_idx]
        if head_idx is None or head_idx < window_start or head_idx >= window_end:
            continue
        if head_idx == dep_idx:
            continue

        src = _normalize_node_text(tokens[head_idx])
        tgt = _normalize_node_text(tokens[dep_idx])
        parts.append(f"{src}->{tgt}:{labels[dep_idx]}")

        if max_arcs is not None and len(parts) >= max_arcs:
            break

    return " ".join(parts)


def linearize_shortest_dependency_path(
    tokens: List[str],
    heads: List[int],
    labels: List[str],
    source_idx: int,
    target_idx: int,
    window_start: int = 0,
    window_end: Optional[int] = None,
    max_arcs: Optional[int] = None,
) -> str:
    """
    Linearize the shortest undirected path between two entity anchors.

    The output keeps dependency direction per edge:
    - head -> dep:  src->tgt:label
    - dep  -> head: src<-tgt:label
    """
    if window_end is None:
        window_end = len(tokens)

    limit = min(len(tokens), len(heads), len(labels), window_end)
    if limit <= window_start:
        return ""
    if source_idx < window_start or source_idx >= limit:
        return ""
    if target_idx < window_start or target_idx >= limit:
        return ""

    adjacency: Dict[int, List[int]] = {i: [] for i in range(window_start, limit)}

    for dep_idx in range(window_start, limit):
        head_idx = heads[dep_idx]
        if head_idx is None or head_idx < window_start or head_idx >= limit:
            continue
        if head_idx == dep_idx:
            continue
        adjacency[dep_idx].append(head_idx)
        adjacency[head_idx].append(dep_idx)

    queue: deque[int] = deque([source_idx])
    prev: Dict[int, Optional[int]] = {source_idx: None}

    while queue:
        node = queue.popleft()
        if node == target_idx:
            break
        for nei in adjacency.get(node, []):
            if nei in prev:
                continue
            prev[nei] = node
            queue.append(nei)

    if target_idx not in prev:
        return ""

    path: List[int] = []
    cur: Optional[int] = target_idx
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    parts: List[str] = []
    for left, right in zip(path, path[1:]):
        left_text = _normalize_node_text(tokens[left])
        right_text = _normalize_node_text(tokens[right])

        if heads[right] == left:
            edge = f"{left_text}->{right_text}:{labels[right]}"
        elif heads[left] == right:
            edge = f"{left_text}<-{right_text}:{labels[left]}"
        else:
            edge = f"{left_text}--{right_text}:dep"

        parts.append(edge)
        if max_arcs is not None and len(parts) >= max_arcs:
            break

    return " ".join(parts)


def select_dependency_sequence(
    raw: Dict,
    dep_view: str,
    dep_form: str,
    use_dep: bool,
    truncated: bool,
    window_start: int,
    window_end: int,
    raw_e1_idx: int,
    raw_e2_idx: int,
    max_dep_arcs: Optional[int],
) -> Tuple[Optional[str], Optional[str]]:
    if not use_dep or dep_form == "none":
        return None, None

    dep_view_used = dep_view

    # A document-level coarse graph is not aligned anymore after text
    # truncation. Fall back to the raw window-local dependency sequence.
    if truncated and dep_view_used == "coarse":
        dep_view_used = "raw"

    if dep_view_used == "raw":
        dep_tokens = raw.get("tokens", [])
        dep_heads = raw.get("dep_heads", [])
        dep_labels = raw.get("dep_labels", [])
        dep_source_idx = int(raw_e1_idx)
        dep_target_idx = int(raw_e2_idx)
        dep_window_start = window_start
        dep_window_end = window_end
    else:
        dep_tokens = raw.get("coarse_tokens", [])
        dep_heads = raw.get("coarse_heads", [])
        dep_labels = raw.get("coarse_labels", [])
        dep_source_idx = raw.get("coarse_e1_idx")
        dep_target_idx = raw.get("coarse_e2_idx")
        dep_window_start = 0
        dep_window_end = None

    if dep_form == "tree":
        dep_seq = linearize_dependency_arcs(
            dep_tokens,
            dep_heads,
            dep_labels,
            window_start=dep_window_start,
            window_end=dep_window_end,
            max_arcs=max_dep_arcs,
        )
    elif dep_form == "sdp":
        if dep_source_idx is None or dep_target_idx is None:
            dep_seq = ""
        else:
            dep_seq = linearize_shortest_dependency_path(
                dep_tokens,
                dep_heads,
                dep_labels,
                source_idx=int(dep_source_idx),
                target_idx=int(dep_target_idx),
                window_start=dep_window_start,
                window_end=dep_window_end,
                max_arcs=max_dep_arcs,
            )
    else:
        dep_seq = ""

    return dep_seq, dep_view_used


def build_input_text(
    tokens: List[str],
    e1_start: int,
    e1_end: int,
    e2_start: int,
    e2_end: int,
    dep_seq: Optional[str] = None,
) -> str:
    """
    Build the encoder-side input string.

    Format:
      [E1S] ... [E1E] ... [E2S] ... [E2E] [DEP] dep_seq [/DEP]
    """
    marked: List[str] = []
    for idx, token in enumerate(tokens):
        if idx == e1_start:
            marked.append("[E1S]")
        if idx == e2_start:
            marked.append("[E2S]")

        marked.append(token)

        if idx == e1_end - 1:
            marked.append("[E1E]")
        if idx == e2_end - 1:
            marked.append("[E2E]")

    if e1_end >= len(tokens) and "[E1E]" not in marked:
        marked.append("[E1E]")
    if e2_end >= len(tokens) and "[E2E]" not in marked:
        marked.append("[E2E]")

    text = " ".join(marked)
    if dep_seq:
        text = f"{text} [DEP] {dep_seq} [/DEP]"
    return text


def build_target_text(
    e1_text: str,
    e2_text: str,
    label: str,
    target_mode: str = "relation_only",
) -> str:
    """Build the decoder-side target string."""
    if target_mode == "relation_only":
        return label
    if target_mode == "natural_language":
        return f"The relation between {e1_text} and {e2_text} is {label}."
    raise ValueError(
        f"Unsupported target_mode={target_mode!r}. "
        f"Expected one of {sorted(SUPPORTED_TARGET_MODES)}."
    )


def normalize_label_prediction(prediction: str) -> str:
    """A tiny utility for downstream metric code."""
    return re.sub(r"\s+", " ", prediction.strip())

