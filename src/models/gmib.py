"""
Graph Mutual Information Bottleneck core module for Stage-3 Ours.

The module is intentionally model-agnostic: it receives semantic/syntax
representations and returns edge-level bottleneck variables plus compression
statistics. The seq2seq generator is wired in `ea_gmib.py`.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

CompressionLossType = Literal["l1", "expected_l0", "entropy", "target_ratio", "none"]


class GMIB(nn.Module):
    """Edge-level graph bottleneck over dual-view node representations."""

    def __init__(
        self,
        syntax_dim: int,
        semantic_dim: int,
        fused_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_anneal_rate: float = 0.95,
        selection_threshold: float = 0.5,
        compression_loss_type: CompressionLossType = "l1",
        target_compression_ratio: Optional[float] = None,
        target_ratio_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if fused_dim <= 0:
            raise ValueError("fused_dim must be positive.")
        if tau_init <= 0.0 or tau_min <= 0.0:
            raise ValueError("tau_init and tau_min must be positive.")
        if not 0.0 < tau_anneal_rate <= 1.0:
            raise ValueError("tau_anneal_rate must be in (0, 1].")
        if not 0.0 <= selection_threshold <= 1.0:
            raise ValueError("selection_threshold must be in [0, 1].")
        if compression_loss_type not in {"l1", "expected_l0", "entropy", "target_ratio", "none"}:
            raise ValueError(f"Unsupported compression_loss_type={compression_loss_type!r}")
        if target_compression_ratio is not None and not 0.0 <= target_compression_ratio <= 1.0:
            raise ValueError("target_compression_ratio must be in [0, 1].")
        if target_ratio_loss_weight < 0.0:
            raise ValueError("target_ratio_loss_weight must be non-negative.")

        self.syntax_dim = int(syntax_dim)
        self.semantic_dim = int(semantic_dim)
        self.fused_dim = int(fused_dim)
        self.tau_min = float(tau_min)
        self.tau_anneal_rate = float(tau_anneal_rate)
        self.selection_threshold = float(selection_threshold)
        self.compression_loss_type = compression_loss_type
        self.target_compression_ratio = (
            float(target_compression_ratio)
            if target_compression_ratio is not None
            else None
        )
        self.target_ratio_loss_weight = float(target_ratio_loss_weight)

        self.syntax_proj = nn.Linear(self.syntax_dim, self.fused_dim)
        self.semantic_proj = nn.Linear(self.semantic_dim, self.fused_dim)
        self.fusion_gate = nn.Linear(self.syntax_dim + self.semantic_dim, self.fused_dim)
        self.fusion_norm = nn.LayerNorm(self.fused_dim)
        self.dropout = nn.Dropout(dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.fused_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "tau",
            torch.tensor(float(tau_init), dtype=torch.float32),
            persistent=False,
        )

    @property
    def current_tau(self) -> float:
        return float(self.tau.detach().cpu().item())

    def set_tau(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("tau must be positive.")
        self.tau.fill_(float(value))

    def anneal_tau(self) -> float:
        next_tau = max(self.tau_min, self.current_tau * self.tau_anneal_rate)
        self.set_tau(next_tau)
        return next_tau

    @staticmethod
    def _binary_concrete(prob: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(prob.dtype).eps
        prob = prob.clamp(min=eps, max=1.0 - eps)
        uniform = torch.rand_like(prob).clamp(min=eps, max=1.0 - eps)
        logit = torch.log(prob) - torch.log1p(-prob)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((logit + logistic_noise) / tau.to(prob.device, prob.dtype))

    def _fuse_nodes(
        self,
        h_syn: torch.Tensor,
        h_sem: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        semantic_nodes = h_sem.unsqueeze(1).expand(-1, h_syn.size(1), -1)
        syn_repr = self.syntax_proj(h_syn)
        sem_repr = self.semantic_proj(semantic_nodes)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_syn, semantic_nodes], dim=-1)))
        fused = gate * syn_repr + (1.0 - gate) * sem_repr
        fused = self.fusion_norm(self.dropout(fused))
        return fused * node_mask.unsqueeze(-1).to(fused.dtype)

    def _compression_loss(self, edge_probs: torch.Tensor) -> torch.Tensor:
        if self.compression_loss_type in {"none", "target_ratio"}:
            return edge_probs.new_zeros(())
        if self.compression_loss_type in {"l1", "expected_l0"}:
            return edge_probs.mean()

        eps = torch.finfo(edge_probs.dtype).eps
        probs = edge_probs.clamp(min=eps, max=1.0 - eps)
        entropy = -probs * torch.log(probs) - (1.0 - probs) * torch.log1p(-probs)
        return entropy.mean()

    def _target_ratio_loss(self, edge_probs: torch.Tensor) -> torch.Tensor:
        if self.target_compression_ratio is None:
            return edge_probs.new_zeros(())
        target = edge_probs.new_tensor(self.target_compression_ratio)
        return (edge_probs.mean() - target).pow(2)

    def forward(
        self,
        h_syn: torch.Tensor,
        h_sem: torch.Tensor,
        a_ij: torch.Tensor,
        adj_matrix: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_syn: [B, N, D_syn]
            h_sem: [B, D_sem]
            a_ij: [B, N, N], syntax-view arc importance scores
            adj_matrix: [B, N, N], 0/1 valid dependency edges
            node_mask: [B, N], valid node mask

        Returns:
            Dictionary with dense edge probabilities/selections and summary stats.
        """
        batch_size, node_count, _ = h_syn.shape
        h_fused = self._fuse_nodes(h_syn=h_syn, h_sem=h_sem, node_mask=node_mask)

        edge_mask = (adj_matrix > 0) & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        p_ij = h_syn.new_zeros((batch_size, node_count, node_count))
        z_ij = h_syn.new_zeros((batch_size, node_count, node_count))
        edge_prob_parts = []
        edge_z_parts = []

        for batch_idx in range(batch_size):
            edge_index = edge_mask[batch_idx].nonzero(as_tuple=False)
            if edge_index.numel() == 0:
                continue

            src_idx = edge_index[:, 0]
            dst_idx = edge_index[:, 1]
            edge_feat = torch.cat(
                [
                    h_fused[batch_idx, src_idx],
                    h_fused[batch_idx, dst_idx],
                    a_ij[batch_idx, src_idx, dst_idx].unsqueeze(-1),
                ],
                dim=-1,
            )
            edge_prob = torch.sigmoid(self.edge_mlp(edge_feat).squeeze(-1))
            if self.training:
                edge_z = self._binary_concrete(edge_prob, self.tau)
            else:
                edge_z = (edge_prob >= self.selection_threshold).to(edge_prob.dtype)

            p_ij[batch_idx, src_idx, dst_idx] = edge_prob
            z_ij[batch_idx, src_idx, dst_idx] = edge_z
            edge_prob_parts.append(edge_prob)
            edge_z_parts.append(edge_z)

        if edge_prob_parts:
            edge_probs = torch.cat(edge_prob_parts, dim=0)
            edge_z = torch.cat(edge_z_parts, dim=0)
            base_compress_loss = self._compression_loss(edge_probs)
            target_ratio_loss = self._target_ratio_loss(edge_probs)
            compress_loss = (
                base_compress_loss
                + self.target_ratio_loss_weight * target_ratio_loss
            )
            prob_compression_ratio = edge_probs.mean()
            valid_arc_count = edge_probs.new_tensor(float(edge_probs.numel()))
            retained_arc_count = edge_z.sum()
        else:
            base_compress_loss = h_syn.new_zeros(())
            target_ratio_loss = h_syn.new_zeros(())
            compress_loss = h_syn.new_zeros(())
            prob_compression_ratio = h_syn.new_zeros(())
            valid_arc_count = h_syn.new_zeros(())
            retained_arc_count = h_syn.new_zeros(())

        avg_retained_arcs = retained_arc_count / max(batch_size, 1)
        compression_ratio = retained_arc_count / valid_arc_count.clamp_min(1.0)

        return {
            "h_fused": h_fused,
            "p_ij": p_ij,
            "z_ij": z_ij,
            "compress_loss": compress_loss,
            "base_compress_loss": base_compress_loss,
            "target_ratio_loss": target_ratio_loss,
            "prob_compression_ratio": prob_compression_ratio,
            "avg_retained_arcs": avg_retained_arcs,
            "compression_ratio": compression_ratio,
            "valid_arc_count": valid_arc_count,
            "retained_arc_count": retained_arc_count,
        }
