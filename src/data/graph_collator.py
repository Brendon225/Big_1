"""
Batch collation for Stage-3 graph + dual-tokenizer inputs.

Text side:
- BioBART tokenizer for seq2seq training labels.

Semantics side:
- PubMedBERT tokenizer for contextual token embeddings.

Graph side:
- padded adjacency / dep-type matrices and node masks.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class BioREGraphCollator:
    """Collate Stage-3 samples into tensors."""

    def __init__(
        self,
        generator_tokenizer,
        pubmedbert_tokenizer,
        max_input_len: int = 512,
        max_target_len: int = 64,
        max_semantic_len: int = 256,
    ) -> None:
        self.generator_tokenizer = generator_tokenizer
        self.pubmedbert_tokenizer = pubmedbert_tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.max_semantic_len = max_semantic_len

    def _encode_generator_text(
        self,
        input_texts: List[str],
        target_texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        enc = self.generator_tokenizer(
            input_texts,
            max_length=self.max_input_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        try:
            dec = self.generator_tokenizer(
                text_target=target_texts,
                max_length=self.max_target_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        except TypeError:
            dec = self.generator_tokenizer(
                target_texts,
                max_length=self.max_target_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        labels = dec["input_ids"].clone()
        labels[labels == self.generator_tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    def _encode_semantics_text(self, semantics_texts: List[str]) -> Dict[str, torch.Tensor]:
        sem_enc = self.pubmedbert_tokenizer(
            semantics_texts,
            max_length=self.max_semantic_len,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = sem_enc.pop("offset_mapping")
        out = {
            "pubmedbert_input_ids": sem_enc["input_ids"],
            "pubmedbert_attention_mask": sem_enc["attention_mask"],
            "pubmedbert_offset_mapping": offsets.to(torch.long),
        }
        if "token_type_ids" in sem_enc:
            out["pubmedbert_token_type_ids"] = sem_enc["token_type_ids"]
        return out

    def _collate_graph(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_nodes = max(int(sample["node_count"]) for sample in batch)

        adj_matrix = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.float32)
        dep_type_ids = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.long)
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        node_char_spans = torch.full((batch_size, max_nodes, 2), -1, dtype=torch.long)
        e1_node_idx = torch.zeros((batch_size,), dtype=torch.long)
        e2_node_idx = torch.zeros((batch_size,), dtype=torch.long)

        for row, sample in enumerate(batch):
            node_count = int(sample["node_count"])
            node_mask[row, :node_count] = True
            e1_node_idx[row] = int(sample["e1_node_idx"])
            e2_node_idx[row] = int(sample["e2_node_idx"])

            adj_tensor = torch.tensor(sample["adj_matrix"], dtype=torch.float32)
            dep_tensor = torch.tensor(sample["dep_type_ids"], dtype=torch.long)
            span_tensor = torch.tensor(sample["node_char_spans"], dtype=torch.long)

            adj_matrix[row, :node_count, :node_count] = adj_tensor
            dep_type_ids[row, :node_count, :node_count] = dep_tensor
            node_char_spans[row, :node_count, :] = span_tensor

        return {
            "adj_matrix": adj_matrix,
            "dep_type_ids": dep_type_ids,
            "node_mask": node_mask,
            "node_char_spans": node_char_spans,
            "e1_node_idx": e1_node_idx,
            "e2_node_idx": e2_node_idx,
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_texts = [sample["input_text"] for sample in batch]
        target_texts = [sample["target_text"] for sample in batch]
        semantics_texts = [sample["semantics_text"] for sample in batch]

        text_tensors = self._encode_generator_text(input_texts, target_texts)
        semantic_tensors = self._encode_semantics_text(semantics_texts)
        graph_tensors = self._collate_graph(batch)

        meta = [
            {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "split": sample["split"],
                "relation": sample["relation"],
                "target": sample["target_text"],
                "e1_text": sample["e1_text"],
                "e2_text": sample["e2_text"],
                "dep_view_used": sample.get("dep_view_used"),
                "dep_form_used": sample.get("dep_form_used"),
                "graph_view_used": sample.get("graph_view_used"),
                "truncated": sample.get("truncated", False),
            }
            for sample in batch
        ]

        return {
            **text_tensors,
            **semantic_tensors,
            **graph_tensors,
            "meta": meta,
        }
