"""
Entity-aware syntax view encoder for Stage-3.

Responsibilities:
- attentive message preparation on node embeddings;
- GCN-style structural aggregation on coarse dependency graph;
- arc-level importance scoring a_ij.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SyntaxView(nn.Module):
    """Attentive GCN encoder with arc importance scorer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dep_type_vocab_size: int,
        dep_type_dim: int = 32,
        gcn_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
        arc_mlp_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if gcn_layers < 1:
            raise ValueError("gcn_layers must be >= 1.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.gcn_layers = int(gcn_layers)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.input_dim)
        self.dropout = nn.Dropout(dropout)

        gcn_projs = []
        in_dim = self.input_dim
        for _ in range(self.gcn_layers):
            gcn_projs.append(nn.Linear(in_dim, self.hidden_dim, bias=False))
            in_dim = self.hidden_dim
        self.gcn_projs = nn.ModuleList(gcn_projs)
        self.gcn_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.gcn_layers)]
        )

        self.dep_type_embedding = nn.Embedding(
            num_embeddings=dep_type_vocab_size,
            embedding_dim=dep_type_dim,
            padding_idx=0,
        )
        self.arc_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + dep_type_dim, arc_mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(arc_mlp_hidden_dim, 1),
        )

    @staticmethod
    def _normalized_adj(adj_matrix: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Build symmetrically normalized adjacency with self-loops.

        Args:
            adj_matrix: [B, N, N], 0/1
            node_mask: [B, N], bool
        """
        dtype = adj_matrix.dtype
        node_mask_f = node_mask.to(dtype)
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

        adj = adj_matrix * pair_mask.to(dtype)
        n = adj.size(-1)
        eye = torch.eye(n, device=adj.device, dtype=dtype).unsqueeze(0)
        eye = eye * node_mask_f.unsqueeze(-1)
        adj_with_self = adj + eye

        degree = adj_with_self.sum(dim=-1).clamp(min=1.0)
        degree_inv_sqrt = degree.pow(-0.5)
        return (
            degree_inv_sqrt.unsqueeze(-1)
            * adj_with_self
            * degree_inv_sqrt.unsqueeze(-2)
        )

    def _attend_nodes(self, x_init: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~node_mask
        attn_out, _ = self.self_attn(
            query=x_init,
            key=x_init,
            value=x_init,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.attn_norm(x_init + self.dropout(attn_out))
        h = h * node_mask.unsqueeze(-1).to(h.dtype)
        return h

    def _gcn_encode(
        self,
        h_in: torch.Tensor,
        adj_norm: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = h_in
        for layer_idx, proj in enumerate(self.gcn_projs):
            h = torch.bmm(adj_norm, h)
            h = proj(h)
            h = self.gcn_norms[layer_idx](h)
            h = F.relu(h)
            h = self.dropout(h)
            h = h * node_mask.unsqueeze(-1).to(h.dtype)
        return h

    def _arc_score(
        self,
        h_syn: torch.Tensor,
        dep_type_ids: torch.Tensor,
        adj_matrix: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, _ = h_syn.shape

        dep_ids = dep_type_ids.clamp(min=0, max=self.dep_type_embedding.num_embeddings - 1)
        arc_mask = (adj_matrix > 0) & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        arc_scores = h_syn.new_zeros((batch_size, node_count, node_count))

        for batch_idx in range(batch_size):
            edge_index = arc_mask[batch_idx].nonzero(as_tuple=False)
            if edge_index.numel() == 0:
                continue

            src_idx = edge_index[:, 0]
            dst_idx = edge_index[:, 1]
            dep_emb = self.dep_type_embedding(dep_ids[batch_idx, src_idx, dst_idx])
            arc_feat = torch.cat(
                [
                    h_syn[batch_idx, src_idx],
                    h_syn[batch_idx, dst_idx],
                    dep_emb,
                ],
                dim=-1,
            )
            arc_scores[batch_idx, src_idx, dst_idx] = torch.sigmoid(
                self.arc_mlp(arc_feat).squeeze(-1)
            )
        return arc_scores

    def forward(
        self,
        x_init: torch.Tensor,
        adj_matrix: torch.Tensor,
        dep_type_ids: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            H_syn: [B, N, hidden_dim]
            a_ij:  [B, N, N]
        """
        h_att = self._attend_nodes(x_init=x_init, node_mask=node_mask)
        adj_norm = self._normalized_adj(adj_matrix=adj_matrix, node_mask=node_mask)
        h_syn = self._gcn_encode(h_in=h_att, adj_norm=adj_norm, node_mask=node_mask)
        a_ij = self._arc_score(
            h_syn=h_syn,
            dep_type_ids=dep_type_ids,
            adj_matrix=adj_matrix,
            node_mask=node_mask,
        )
        return h_syn, a_ij
