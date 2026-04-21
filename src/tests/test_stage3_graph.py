"""
Smoke checks for Stage-3 graph bridge modules.

Run:
  venv/Scripts/python.exe src/tests/test_stage3_graph.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from src.builders.graph_builder import (  # noqa: E402
    build_dep_type_vocab,
    save_dep_type_vocab,
)
from src.data.graph_collator import BioREGraphCollator  # noqa: E402
from src.data.graph_dataset import BioREGraphDataset  # noqa: E402


def build_tiny_graph_file() -> tuple[Path, Path]:
    src_path = ROOT / "data" / "experiment_views" / "coarsened" / "ChemProtSent_train.json"
    with src_path.open("r", encoding="utf-8") as handle:
        lines = [next(handle), next(handle)]

    tmp_dir = Path(tempfile.mkdtemp(prefix="stage3_graph_test_"))
    file_path = tmp_dir / "ChemProtSent_train.json"
    vocab_path = tmp_dir / "dep_type_vocab.json"

    with file_path.open("w", encoding="utf-8") as out:
        for line in lines:
            out.write(line)

    vocab = build_dep_type_vocab([file_path], labels_key="coarse_labels")
    save_dep_type_vocab(vocab, vocab_path)
    return file_path, vocab_path


print("=" * 60)
print("Stage-3 graph bridge smoke test")
data_path, vocab_path = build_tiny_graph_file()

dataset = BioREGraphDataset(
    file_path=str(data_path),
    use_dep=False,
    dep_view="raw",
    dep_form="none",
    target_mode="relation_only",
    max_src_len=256,
    dep_type_vocab_path=str(vocab_path),
)
assert len(dataset) == 2

sample = dataset[0]
for key in [
    "adj_matrix",
    "dep_type_ids",
    "node_count",
    "e1_node_idx",
    "e2_node_idx",
    "node_char_spans",
    "semantics_text",
]:
    assert key in sample, f"missing key: {key}"
assert sample["node_count"] == len(sample["adj_matrix"])
print("  OK dataset graph fields")

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

collator = BioREGraphCollator(
    generator_tokenizer=generator_tokenizer,
    pubmedbert_tokenizer=pubmedbert_tokenizer,
    max_input_len=128,
    max_target_len=16,
    max_semantic_len=128,
)

batch = collator([dataset[0], dataset[1]])
assert batch["adj_matrix"].shape[0] == 2
assert batch["dep_type_ids"].shape == batch["adj_matrix"].shape
assert batch["node_mask"].shape[0] == 2
assert batch["pubmedbert_input_ids"].shape[0] == 2
assert batch["pubmedbert_offset_mapping"].shape[0] == 2
print("  OK collator batch fields and shapes")

dataset.close()
print("ALL STAGE-3 GRAPH TESTS PASSED")
