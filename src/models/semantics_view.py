"""
Sequential semantics view encoder for Stage-3.

Responsibilities:
- encode entity-marked sentence text with PubMedBERT;
- output sentence-level semantic representation H_sem (CLS);
- output node initialization embeddings X_init aligned to coarse graph nodes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class SemanticsView(nn.Module):
    """PubMedBERT-based semantics encoder."""

    def __init__(
        self,
        model_name_or_path: str,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _span_token_mask(
        offsets: torch.Tensor,
        span_start: int,
        span_end: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a WordPiece mask that overlaps with one char span.

        Overlap criterion:
        token_end > span_start and token_start < span_end
        """
        token_start = offsets[:, 0]
        token_end = offsets[:, 1]
        valid_attn = attention_mask.bool()
        overlap = (token_end > span_start) & (token_start < span_end)
        return valid_attn & overlap

    @staticmethod
    def _fallback_single_token_mask(
        offsets: torch.Tensor,
        span_start: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback to the closest valid token by start offset."""
        valid = attention_mask.bool() & (offsets[:, 1] > offsets[:, 0])
        if not bool(valid.any()):
            return torch.zeros_like(valid)

        starts = offsets[:, 0]
        # Large number for invalid positions so they are never selected.
        distance = torch.where(
            valid,
            (starts - int(span_start)).abs(),
            torch.full_like(starts, 10**9),
        )
        best_idx = int(torch.argmin(distance).item())
        out = torch.zeros_like(valid)
        out[best_idx] = True
        return out

    def _pool_node_embeddings(
        self,
        token_embeddings: torch.Tensor,
        offset_mapping: torch.Tensor,
        node_char_spans: torch.Tensor,
        node_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token embeddings into node embeddings by char-span overlap.

        Args:
            token_embeddings: [B, L, H]
            offset_mapping: [B, L, 2]
            node_char_spans: [B, N, 2]
            node_mask: [B, N]
            attention_mask: [B, L]
        Returns:
            X_init: [B, N, H]
        """
        batch_size, _, hidden_size = token_embeddings.shape
        node_count = node_char_spans.shape[1]
        x_init = token_embeddings.new_zeros((batch_size, node_count, hidden_size))

        for b in range(batch_size):
            offsets_b = offset_mapping[b]
            token_emb_b = token_embeddings[b]
            attn_b = attention_mask[b]

            for n in range(node_count):
                if not bool(node_mask[b, n]):
                    continue

                span_start = int(node_char_spans[b, n, 0].item())
                span_end = int(node_char_spans[b, n, 1].item())
                if span_start < 0 or span_end <= span_start:
                    continue

                tok_mask = self._span_token_mask(
                    offsets=offsets_b,
                    span_start=span_start,
                    span_end=span_end,
                    attention_mask=attn_b,
                )
                if not bool(tok_mask.any()):
                    tok_mask = self._fallback_single_token_mask(
                        offsets=offsets_b,
                        span_start=span_start,
                        attention_mask=attn_b,
                    )
                if not bool(tok_mask.any()):
                    continue

                x_init[b, n] = token_emb_b[tok_mask].mean(dim=0)

        return x_init

    def forward(
        self,
        pubmedbert_input_ids: torch.Tensor,
        pubmedbert_attention_mask: torch.Tensor,
        node_char_spans: torch.Tensor,
        node_mask: torch.Tensor,
        pubmedbert_offset_mapping: torch.Tensor,
        pubmedbert_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            H_sem: [B, H]
            X_init: [B, N, H]
        """
        encoder_kwargs = {
            "input_ids": pubmedbert_input_ids,
            "attention_mask": pubmedbert_attention_mask,
        }
        if pubmedbert_token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = pubmedbert_token_type_ids

        outputs = self.encoder(**encoder_kwargs, return_dict=True)
        token_embeddings = outputs.last_hidden_state

        h_sem = token_embeddings[:, 0, :]
        h_sem = self.layer_norm(self.dropout(h_sem))

        x_init = self._pool_node_embeddings(
            token_embeddings=token_embeddings,
            offset_mapping=pubmedbert_offset_mapping,
            node_char_spans=node_char_spans,
            node_mask=node_mask,
            attention_mask=pubmedbert_attention_mask,
        )
        x_init = self.layer_norm(self.dropout(x_init))
        x_init = x_init * node_mask.unsqueeze(-1).to(x_init.dtype)
        return h_sem, x_init
