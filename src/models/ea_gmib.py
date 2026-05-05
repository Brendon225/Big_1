"""
Complete EA-GMIB model assembly for Stage-3.

This file wires the existing semantic/syntax encoders, the GM-IB module, and
BioBART. It keeps model composition separate from training logic.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from src.models.gmib import GMIB, CompressionLossType
from src.models.semantics_view import SemanticsView
from src.models.syntax_view import SyntaxView


class EAGMIBModel(nn.Module):
    """Entity-aware graph mutual information bottleneck model."""

    def __init__(
        self,
        biobart_path: str,
        pubmedbert_path: str,
        dep_type_vocab_size: int,
        gcn_hidden_dim: int = 256,
        gcn_layers: int = 2,
        gcn_attention_heads: int = 8,
        dep_type_dim: int = 32,
        gmib_hidden_dim: int = 256,
        beta: float = 1e-6,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_anneal_rate: float = 0.95,
        selection_threshold: float = 0.5,
        compression_loss_type: CompressionLossType = "l1",
        dropout: float = 0.1,
        freeze_pubmedbert: bool = False,
        model_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        self.beta = float(beta)

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

        self.syntax_view = SyntaxView(
            input_dim=self.semantic_hidden_size,
            hidden_dim=gcn_hidden_dim,
            dep_type_vocab_size=dep_type_vocab_size,
            dep_type_dim=dep_type_dim,
            gcn_layers=gcn_layers,
            attention_heads=gcn_attention_heads,
            dropout=dropout,
        )

        self.gmib = GMIB(
            syntax_dim=gcn_hidden_dim,
            semantic_dim=self.semantic_hidden_size,
            fused_dim=self.generator_hidden_size,
            hidden_dim=gmib_hidden_dim,
            dropout=dropout,
            tau_init=tau_init,
            tau_min=tau_min,
            tau_anneal_rate=tau_anneal_rate,
            selection_threshold=selection_threshold,
            compression_loss_type=compression_loss_type,
        )
        self.inject_norm = nn.LayerNorm(self.generator_hidden_size)
        self.inject_dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_augmented_inputs(
        generator: AutoModelForSeq2SeqLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        injection_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_emb = generator.get_input_embeddings()(input_ids)
        encoder = generator.model.encoder
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

    @staticmethod
    def _pool_selected_nodes(
        h_fused: torch.Tensor,
        z_ij: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = node_mask.to(h_fused.dtype)
        node_weights = (z_ij.sum(dim=1) + z_ij.sum(dim=2)) * mask
        fallback_weights = mask
        has_selected = node_weights.sum(dim=1, keepdim=True) > 0
        node_weights = torch.where(has_selected, node_weights, fallback_weights)
        denom = node_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h_fused * node_weights.unsqueeze(-1)).sum(dim=1) / denom

    def update_temperature(self, tau: float) -> None:
        self.gmib.set_tau(tau)

    def anneal_temperature(self) -> float:
        return self.gmib.anneal_tau()

    @property
    def current_tau(self) -> float:
        return self.gmib.current_tau

    def _build_injection_vector(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_sem, x_init = self.semantics_view(
            pubmedbert_input_ids=batch["pubmedbert_input_ids"],
            pubmedbert_attention_mask=batch["pubmedbert_attention_mask"],
            node_char_spans=batch["node_char_spans"],
            node_mask=batch["node_mask"],
            pubmedbert_offset_mapping=batch["pubmedbert_offset_mapping"],
            pubmedbert_token_type_ids=batch.get("pubmedbert_token_type_ids"),
        )
        h_syn, a_ij = self.syntax_view(
            x_init=x_init,
            adj_matrix=batch["adj_matrix"],
            dep_type_ids=batch["dep_type_ids"],
            node_mask=batch["node_mask"],
        )
        gmib_out = self.gmib(
            h_syn=h_syn,
            h_sem=h_sem,
            a_ij=a_ij,
            adj_matrix=batch["adj_matrix"],
            node_mask=batch["node_mask"],
        )
        selected_pool = self._pool_selected_nodes(
            h_fused=gmib_out["h_fused"],
            z_ij=gmib_out["z_ij"],
            node_mask=batch["node_mask"],
        )
        injection_vec = self.inject_dropout(self.inject_norm(selected_pool))
        gmib_out["injection_vec"] = injection_vec
        return gmib_out

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        gmib_out = self._build_injection_vector(batch)
        aug_emb, aug_mask = self._build_augmented_inputs(
            generator=self.generator,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            injection_vec=gmib_out["injection_vec"],
        )
        generator_outputs = self.generator(
            inputs_embeds=aug_emb,
            attention_mask=aug_mask,
            labels=labels,
            return_dict=True,
        )

        gen_loss = generator_outputs.loss
        compress_loss = gmib_out["compress_loss"]
        total_loss = None
        if gen_loss is not None:
            total_loss = gen_loss + self.beta * compress_loss

        return {
            "loss": total_loss,
            "gen_loss": gen_loss,
            "compress_loss": compress_loss,
            "logits": generator_outputs.logits,
            "avg_retained_arcs": gmib_out["avg_retained_arcs"],
            "compression_ratio": gmib_out["compression_ratio"],
            "valid_arc_count": gmib_out["valid_arc_count"],
            "retained_arc_count": gmib_out["retained_arc_count"],
            "p_ij": gmib_out["p_ij"],
            "z_ij": gmib_out["z_ij"],
        }
