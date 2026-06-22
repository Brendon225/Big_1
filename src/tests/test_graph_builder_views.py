"""
Lightweight checks for Stage-3 graph view switching.

These tests avoid torch/transformers and specifically guard the B7 raw-graph
ablation: dep_view="raw" must consume raw dependency fields, while
dep_view="coarse" must consume entity-folded fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.builders.graph_builder import GraphBuilder, PAD_LABEL, UNK_LABEL  # noqa: E402


VOCAB = {
    PAD_LABEL: 0,
    UNK_LABEL: 1,
    "ROOT": 2,
    "compound": 3,
    "nsubj": 4,
    "obj": 5,
}

RECORD = {
    "id": "toy_1",
    "sentence": "Alpha beta gamma activates delta.",
    "tokens": ["Alpha", "beta", "gamma", "activates", "delta", "."],
    "dep_heads": [3, 2, 3, 3, 3, 3],
    "dep_labels": ["nsubj", "compound", "nsubj", "ROOT", "obj", "punct"],
    "entity1": {
        "text": "beta gamma",
        "start_char": 6,
        "end_char": 16,
        "start_tok": 1,
        "end_tok": 3,
    },
    "entity2": {
        "text": "delta",
        "start_char": 27,
        "end_char": 32,
        "start_tok": 4,
        "end_tok": 5,
    },
    "coarse_tokens": ["Alpha", "beta gamma", "activates", "delta", "."],
    "coarse_heads": [2, 2, 2, 2, 2],
    "coarse_labels": ["nsubj", "nsubj", "ROOT", "obj", "punct"],
    "coarse_e1_idx": 1,
    "coarse_e2_idx": 3,
    "coarse_node_types": ["normal", "entity1", "normal", "entity2", "normal"],
}


def test_raw_and_coarse_graph_views_are_distinct() -> None:
    raw_features = GraphBuilder(VOCAB, dep_view="raw").build_from_record(RECORD)
    coarse_features = GraphBuilder(VOCAB, dep_view="coarse").build_from_record(RECORD)

    assert raw_features.node_count == len(RECORD["tokens"])
    assert coarse_features.node_count == len(RECORD["coarse_tokens"])
    assert raw_features.node_count > coarse_features.node_count

    assert raw_features.e1_node_idx == RECORD["entity1"]["start_tok"]
    assert raw_features.e2_node_idx == RECORD["entity2"]["start_tok"]
    assert coarse_features.e1_node_idx == RECORD["coarse_e1_idx"]
    assert coarse_features.e2_node_idx == RECORD["coarse_e2_idx"]

    # Raw graph keeps the internal multi-token entity arc beta-gamma.
    assert raw_features.adj_matrix[1][2] == 1
    assert raw_features.adj_matrix[2][1] == 1

    # Coarse graph has a single entity super-node, so no internal entity arc exists.
    assert coarse_features.adj_matrix[1][1] == 0


if __name__ == "__main__":
    test_raw_and_coarse_graph_views_are_distinct()
    print("ALL TESTS PASSED")
