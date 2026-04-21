"""
B3/B4/B5 baseline model for Stage-3.

Model variants:
- B3: syntax-only (Attentive GCN pooled vector injection)
- B4: semantics-only (PubMedBERT CLS injection)
- B5: dual-view concat (syntax + semantics) without IB
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from src.models.semantics_view import SemanticsView
from src.models.syntax_view import SyntaxView

SUPPORTED_B345_MODEL_TYPES = {"b3", "b4", "b5"}


class B345Model(nn.Module):
    """Unified model for B3/B4/B5."""

    def __init__(
        self,
        model_type: str,
        biobart_path: str,
        pubmedbert_path: str,
        dep_type_vocab_size: int,
        gcn_hidden_dim: int = 256,
        gcn_layers: int = 2,
        gcn_attention_heads: int = 8,
        dep_type_dim: int = 32,
        dropout: float = 0.1,
        freeze_pubmedbert: bool = False,
        model_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        model_type = str(model_type).lower()
        if model_type not in SUPPORTED_B345_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type={model_type!r}. "
                f"Expected one of {sorted(SUPPORTED_B345_MODEL_TYPES)}."
            )
        self.model_type = model_type

        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            biobart_path,
            dtype=model_dtype,
        )
        self.generator_hidden_size = int(self.generator.config.d_model)

        self.semantics_view = SemanticsView(
            model_name_or_path=pubmedbert_path,
            freeze_encoder=freeze_pubmedbert,
            dropout=dropout,
        )
        self.semantic_hidden_size = int(self.semantics_view.hidden_size)

        self.syntax_view: Optional[SyntaxView] = None
        if self.model_type in {"b3", "b5"}:
            self.syntax_view = SyntaxView(
                input_dim=self.semantic_hidden_size,
                hidden_dim=gcn_hidden_dim,
                dep_type_vocab_size=dep_type_vocab_size,
                dep_type_dim=dep_type_dim,
                gcn_layers=gcn_layers,
                attention_heads=gcn_attention_heads,
                dropout=dropout,
            )

        self.syn_proj = nn.Linear(gcn_hidden_dim, self.generator_hidden_size)
        self.sem_proj = nn.Linear(self.semantic_hidden_size, self.generator_hidden_size)
        self.dual_proj = nn.Linear(self.generator_hidden_size * 2, self.generator_hidden_size)
        self.inject_norm = nn.LayerNorm(self.generator_hidden_size)
        self.inject_dropout = nn.Dropout(dropout)

    @staticmethod
    def _masked_mean_pool(node_repr: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        mask = node_mask.unsqueeze(-1).to(node_repr.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (node_repr * mask).sum(dim=1) / denom

    def _build_injection_vector(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h_sem, x_init = self.semantics_view(
            pubmedbert_input_ids=batch["pubmedbert_input_ids"],
            pubmedbert_attention_mask=batch["pubmedbert_attention_mask"],
            node_char_spans=batch["node_char_spans"],
            node_mask=batch["node_mask"],
            pubmedbert_offset_mapping=batch["pubmedbert_offset_mapping"],
            pubmedbert_token_type_ids=batch.get("pubmedbert_token_type_ids"),
        )

        if self.model_type == "b4":
            feat = self.sem_proj(h_sem)
            return self.inject_dropout(self.inject_norm(feat))

        assert self.syntax_view is not None
        h_syn, _ = self.syntax_view(
            x_init=x_init,
            adj_matrix=batch["adj_matrix"],
            dep_type_ids=batch["dep_type_ids"],
            node_mask=batch["node_mask"],
        )
        syn_pool = self._masked_mean_pool(h_syn, batch["node_mask"])
        syn_feat = self.syn_proj(syn_pool)

        if self.model_type == "b3":
            return self.inject_dropout(self.inject_norm(syn_feat))

        sem_feat = self.sem_proj(h_sem)
        dual = torch.cat([syn_feat, sem_feat], dim=-1)
        feat = self.dual_proj(dual)
        return self.inject_dropout(self.inject_norm(feat))

    def _build_augmented_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        injection_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_emb = self.generator.get_input_embeddings()(input_ids)
        encoder = self.generator.model.encoder
        embed_scale = float(getattr(encoder, "embed_scale", 1.0))
        token_emb = token_emb * embed_scale

        injection_token = injection_vec.unsqueeze(1)
        aug_emb = torch.cat([token_emb, injection_token], dim=1)

        extra_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        aug_mask = torch.cat([attention_mask, extra_mask], dim=1)
        return aug_emb, aug_mask

    def forward(self, batch: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None):
        injection_vec = self._build_injection_vector(batch)
        aug_emb, aug_mask = self._build_augmented_inputs(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            injection_vec=injection_vec,
        )
        return self.generator(
            inputs_embeds=aug_emb,
            attention_mask=aug_mask,
            labels=labels,
            return_dict=True,
        )
