"""
Smoke tests for the Stage-3 GM-IB core module.

Run:
  venv/Scripts/python.exe src/tests/test_stage3_gmib.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.gmib import GMIB  # noqa: E402


print("=" * 60)
print("Stage-3 GM-IB smoke test")
torch.manual_seed(42)

batch_size = 2
node_count = 5
syntax_dim = 8
semantic_dim = 10
fused_dim = 12

h_syn = torch.randn(batch_size, node_count, syntax_dim, requires_grad=True)
h_sem = torch.randn(batch_size, semantic_dim, requires_grad=True)
a_ij = torch.rand(batch_size, node_count, node_count)
adj_matrix = torch.zeros(batch_size, node_count, node_count)
node_mask = torch.tensor(
    [
        [True, True, True, True, False],
        [True, True, True, False, False],
    ]
)

adj_matrix[0, 0, 1] = 1
adj_matrix[0, 1, 0] = 1
adj_matrix[0, 1, 2] = 1
adj_matrix[0, 2, 1] = 1
adj_matrix[0, 2, 3] = 1
adj_matrix[0, 3, 2] = 1
adj_matrix[1, 0, 1] = 1
adj_matrix[1, 1, 0] = 1
adj_matrix[1, 1, 2] = 1
adj_matrix[1, 2, 1] = 1

gmib = GMIB(
    syntax_dim=syntax_dim,
    semantic_dim=semantic_dim,
    fused_dim=fused_dim,
    hidden_dim=16,
    dropout=0.0,
    tau_init=1.0,
    tau_min=0.2,
    tau_anneal_rate=0.5,
    selection_threshold=0.5,
    compression_loss_type="l1",
)

gmib.train()
out = gmib(
    h_syn=h_syn,
    h_sem=h_sem,
    a_ij=a_ij,
    adj_matrix=adj_matrix,
    node_mask=node_mask,
)

assert out["h_fused"].shape == (batch_size, node_count, fused_dim)
assert out["p_ij"].shape == (batch_size, node_count, node_count)
assert out["z_ij"].shape == (batch_size, node_count, node_count)
assert out["compress_loss"].ndim == 0
assert float(out["p_ij"].min().item()) >= 0.0
assert float(out["p_ij"].max().item()) <= 1.0
assert float(out["z_ij"].min().item()) >= 0.0
assert float(out["z_ij"].max().item()) <= 1.0
assert float(out["valid_arc_count"].item()) == float(adj_matrix.sum().item())
print("  OK forward shapes and valid ranges")

loss = out["h_fused"].sum() + out["z_ij"].sum() + out["compress_loss"]
loss.backward()
assert gmib.edge_mlp[0].weight.grad is not None
assert gmib.fusion_gate.weight.grad is not None
assert h_syn.grad is not None
assert h_sem.grad is not None
print("  OK gradient flow")

old_tau = gmib.current_tau
new_tau = gmib.anneal_tau()
assert new_tau == max(0.2, old_tau * 0.5)
assert gmib.current_tau == new_tau
print("  OK temperature annealing")

gmib.eval()
with torch.no_grad():
    eval_out = gmib(
        h_syn=h_syn.detach(),
        h_sem=h_sem.detach(),
        a_ij=a_ij,
        adj_matrix=adj_matrix,
        node_mask=node_mask,
    )
selected_values = eval_out["z_ij"][adj_matrix.bool()].unique().tolist()
assert all(value in {0.0, 1.0} for value in selected_values)
print("  OK eval hard selection")

print("ALL STAGE-3 GM-IB TESTS PASSED")
