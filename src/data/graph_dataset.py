"""
Graph-augmented dataset for Stage-3 experiments.

This class extends BioREDataset without modifying Stage-2 data code.
It adds graph features produced by `GraphBuilder`.
"""

from __future__ import annotations

from typing import Dict

from src.builders.graph_builder import GraphBuilder
from src.builders.input_builder import build_input_text
from src.data.dataset import (
    BioREDataset,
    entity_centered_truncate,
    entity_centered_window,
)


class BioREGraphDataset(BioREDataset):
    """
    Stage-3 dataset with both text-side and graph-side features.

    Added sample keys:
    - adj_matrix
    - dep_type_ids
    - node_count
    - e1_node_idx
    - e2_node_idx
    - node_char_spans
    - semantics_text
    """

    def __init__(
        self,
        *args,
        dep_type_vocab_path: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.graph_builder = GraphBuilder.from_vocab_path(dep_type_vocab_path)

    def _build_semantics_text(self, raw: Dict) -> str:
        full_tokens = raw.get("tokens", [])
        entity1 = raw.get("entity1", {})
        entity2 = raw.get("entity2", {})

        e1_start = int(entity1.get("start_tok", 0))
        e1_end = max(int(entity1.get("end_tok", e1_start + 1)), e1_start + 1)
        e2_start = int(entity2.get("start_tok", 0))
        e2_end = max(int(entity2.get("end_tok", e2_start + 1)), e2_start + 1)

        if len(full_tokens) > self.max_src_len:
            window_start, window_end = entity_centered_window(
                len(full_tokens),
                e1_start,
                e1_end,
                e2_start,
                e2_end,
                self.max_src_len,
            )
            tokens, e1_start, e1_end, e2_start, e2_end = entity_centered_truncate(
                full_tokens,
                e1_start,
                e1_end,
                e2_start,
                e2_end,
                self.max_src_len,
            )
            _ = (window_start, window_end)
        else:
            tokens = full_tokens

        return build_input_text(
            tokens=tokens,
            e1_start=e1_start,
            e1_end=e1_end,
            e2_start=e2_start,
            e2_end=e2_end,
            dep_seq=None,
        )

    def __getitem__(self, idx: int) -> Dict:
        raw = self._read_raw_record(self.selected_offsets[idx])

        text_sample = self._build_sample(raw)
        graph_features = self.graph_builder.build_from_record(raw)
        semantics_text = self._build_semantics_text(raw)

        text_sample.update(
            {
                "adj_matrix": graph_features.adj_matrix,
                "dep_type_ids": graph_features.dep_type_ids,
                "node_count": graph_features.node_count,
                "e1_node_idx": graph_features.e1_node_idx,
                "e2_node_idx": graph_features.e2_node_idx,
                "node_char_spans": graph_features.node_char_spans,
                "semantics_text": semantics_text,
                "raw_sentence": raw.get("sentence", ""),
            }
        )
        return text_sample
