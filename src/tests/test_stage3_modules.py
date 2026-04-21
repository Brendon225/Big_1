"""
Smoke tests for Stage-3 semantics/syntax view modules.

Run:
  venv/Scripts/python.exe src/tests/test_stage3_modules.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.builders.graph_builder import build_dep_type_vocab, save_dep_type_vocab  # noqa: E402
from src.data.graph_collator import BioREGraphCollator  # noqa: E402
from src.data.graph_dataset import BioREGraphDataset  # noqa: E402
from src.models.semantics_view import SemanticsView  # noqa: E402
from src.models.syntax_view import SyntaxView  # noqa: E402


def build_tiny_input() -> tuple[Path, Path]:
    src_path = ROOT / "data" / "experiment_views" / "coarsened" / "ChemProtSent_train.json"
    tmp_dir = Path(tempfile.mkdtemp(prefix="stage3_modules_test_"))
    data_path = tmp_dir / "ChemProtSent_train.json"
    vocab_path = tmp_dir / "dep_type_vocab.json"

    with src_path.open("r", encoding="utf-8") as handle:
        lines = [next(handle), next(handle)]
    with data_path.open("w", encoding="utf-8") as out:
        for line in lines:
            out.write(line)

    vocab = build_dep_type_vocab([data_path], labels_key="coarse_labels")
    save_dep_type_vocab(vocab, vocab_path)
    return data_path, vocab_path


print("=" * 60)
print("Stage-3 module smoke test")
data_path, vocab_path = build_tiny_input()
dep_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

generator_tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models" / "biobart-base"))
generator_tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            "[E1S]",
            "[E1E]",
            "[E2S]",
            "[E2E]",
            "[DEP]",
            "[/DEP]",
        ]
    }
)
pubmedbert_tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models" / "pubmedbert-base"))

dataset = BioREGraphDataset(
    file_path=str(data_path),
    use_dep=False,
    dep_view="raw",
    dep_form="none",
    target_mode="relation_only",
    max_src_len=256,
    dep_type_vocab_path=str(vocab_path),
)

collator = BioREGraphCollator(
    generator_tokenizer=generator_tokenizer,
    pubmedbert_tokenizer=pubmedbert_tokenizer,
    max_input_len=128,
    max_target_len=16,
    max_semantic_len=128,
)
batch = collator([dataset[0], dataset[1]])

semantics = SemanticsView(
    model_name_or_path=str(ROOT / "models" / "pubmedbert-base"),
    freeze_encoder=True,
    dropout=0.0,
)
h_sem, x_init = semantics(
    pubmedbert_input_ids=batch["pubmedbert_input_ids"],
    pubmedbert_attention_mask=batch["pubmedbert_attention_mask"],
    node_char_spans=batch["node_char_spans"],
    node_mask=batch["node_mask"],
    pubmedbert_offset_mapping=batch["pubmedbert_offset_mapping"],
    pubmedbert_token_type_ids=batch.get("pubmedbert_token_type_ids"),
)

assert h_sem.ndim == 2 and h_sem.shape[0] == 2
assert x_init.ndim == 3 and x_init.shape[0] == 2
assert x_init.shape[1] == batch["node_mask"].shape[1]
print("  OK semantics view output shapes")

syntax = SyntaxView(
    input_dim=h_sem.shape[-1],
    hidden_dim=256,
    dep_type_vocab_size=len(dep_vocab),
    dep_type_dim=32,
    gcn_layers=2,
    attention_heads=8,
    dropout=0.0,
)
h_syn, a_ij = syntax(
    x_init=x_init,
    adj_matrix=batch["adj_matrix"],
    dep_type_ids=batch["dep_type_ids"],
    node_mask=batch["node_mask"],
)

assert h_syn.shape[:2] == x_init.shape[:2]
assert h_syn.shape[-1] == 256
assert a_ij.shape == batch["adj_matrix"].shape
assert float(a_ij.min().item()) >= 0.0
assert float(a_ij.max().item()) <= 1.0
print("  OK syntax view output shapes and score range")

# Gradient flow check on syntax module.
syntax.train()
x_rand = torch.randn_like(x_init, requires_grad=True)
h_syn2, a_ij2 = syntax(
    x_init=x_rand,
    adj_matrix=batch["adj_matrix"],
    dep_type_ids=batch["dep_type_ids"],
    node_mask=batch["node_mask"],
)
loss = h_syn2.sum() + a_ij2.sum()
loss.backward()
assert syntax.gcn_projs[0].weight.grad is not None
assert syntax.arc_mlp[0].weight.grad is not None
print("  OK syntax view gradient flow")

dataset.close()
print("ALL STAGE-3 MODULE TESTS PASSED")
