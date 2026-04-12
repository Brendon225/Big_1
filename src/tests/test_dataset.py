"""
Lightweight tests for the dataset layer.

These checks intentionally avoid torch/transformers so they can run in a plain
Python environment before model training dependencies are installed.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import (  # noqa: E402
    BioREDataset,
    entity_centered_truncate,
)
from src.builders.input_builder import (  # noqa: E402
    build_input_text,
    build_target_text,
    linearize_dependency_arcs,
    linearize_shortest_dependency_path,
)


BASE = str(ROOT / "data" / "coarsened")
TMP_DIRS = []


def make_tiny_ddi_file() -> str:
    records = [
        {
            "id": "DDI_train_pos_1",
            "dataset": "DDI",
            "split": "train",
            "tokens": ["A", "interacts", "with", "B", "."],
            "dep_heads": [1, 1, 1, 2, 1],
            "dep_labels": ["nsubj", "ROOT", "prep", "pobj", "punct"],
            "coarse_tokens": ["A", "interacts", "with", "B", "."],
            "coarse_heads": [1, 1, 1, 2, 1],
            "coarse_labels": ["nsubj", "ROOT", "prep", "pobj", "punct"],
            "entity1": {"text": "A", "start_tok": 0, "end_tok": 1},
            "entity2": {"text": "B", "start_tok": 3, "end_tok": 4},
            "relation": "DDI-effect",
            "is_positive": True,
            "coarse_status": "coarsened",
        },
        {
            "id": "DDI_train_pos_2",
            "dataset": "DDI",
            "split": "train",
            "tokens": ["C", "blocks", "D", "."],
            "dep_heads": [1, 1, 1, 1],
            "dep_labels": ["nsubj", "ROOT", "obj", "punct"],
            "coarse_tokens": ["C", "blocks", "D", "."],
            "coarse_heads": [1, 1, 1, 1],
            "coarse_labels": ["nsubj", "ROOT", "obj", "punct"],
            "entity1": {"text": "C", "start_tok": 0, "end_tok": 1},
            "entity2": {"text": "D", "start_tok": 2, "end_tok": 3},
            "relation": "DDI-mechanism",
            "is_positive": True,
            "coarse_status": "coarsened",
        },
    ]

    for idx in range(10):
        records.append(
            {
                "id": f"DDI_train_neg_{idx}",
                "dataset": "DDI",
                "split": "train",
                "tokens": ["X", "with", "Y", "."],
                "dep_heads": [1, 1, 1, 1],
                "dep_labels": ["nsubj", "ROOT", "obj", "punct"],
                "coarse_tokens": ["X", "with", "Y", "."],
                "coarse_heads": [1, 1, 1, 1],
                "coarse_labels": ["nsubj", "ROOT", "obj", "punct"],
                "entity1": {"text": "X", "start_tok": 0, "end_tok": 1},
                "entity2": {"text": "Y", "start_tok": 2, "end_tok": 3},
                "relation": "DDI-false",
                "is_positive": False,
                "coarse_status": "coarsened",
            }
        )

    tmp_dir = tempfile.TemporaryDirectory()
    TMP_DIRS.append(tmp_dir)
    path = Path(tmp_dir.name) / "DDI_train.json"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


print("=" * 60)
print("Test 1: entity_centered_truncate")
tokens = list(range(200))
new_toks, s1, e1, s2, e2 = entity_centered_truncate(
    tokens, e1_start=10, e1_end=12, e2_start=180, e2_end=183, max_len=100
)
assert len(new_toks) <= 100, f"truncated len={len(new_toks)} > 100"
assert 0 <= s1 < e1 <= len(new_toks), f"e1 invalid: [{s1},{e1})"
assert 0 <= s2 < e2 <= len(new_toks), f"e2 invalid: [{s2},{e2})"
print(f"  OK  len={len(new_toks)}  e1=[{s1},{e1})  e2=[{s2},{e2})")

new_toks2, *_ = entity_centered_truncate(
    tokens[:50], e1_start=5, e1_end=7, e2_start=30, e2_end=32, max_len=100
)
assert len(new_toks2) == 50, "short sentence should stay untouched"
print(f"  OK  short sentence untouched len={len(new_toks2)}")

print("=" * 60)
print("Test 2: build_input_text")
toks = ["Aspirin", "inhibits", "COX-2", "expression", "."]
inp = build_input_text(toks, 0, 1, 2, 3)
assert "[E1S]" in inp and "[E1E]" in inp
assert "[E2S]" in inp and "[E2E]" in inp
assert "[DEP]" not in inp
print(f"  OK  input_text: {inp}")

inp_dep = build_input_text(
    toks, 0, 1, 2, 3, dep_seq="Aspirin->inhibits:nsubj"
)
assert "[DEP]" in inp_dep and "[/DEP]" in inp_dep
print(f"  OK  with dep:   {inp_dep}")

print("=" * 60)
print("Test 3: build_target_text")
tgt = build_target_text("Aspirin", "COX-2", "CPR:4")
assert tgt == "CPR:4"
nl_tgt = build_target_text(
    "Aspirin", "COX-2", "CPR:4", target_mode="natural_language"
)
assert nl_tgt == "The relation between Aspirin and COX-2 is CPR:4."
print(f"  OK  target: {tgt}")

print("=" * 60)
print("Test 4: linearize_dependency_arcs")
dep_seq = linearize_dependency_arcs(
    tokens=["Aspirin", "inhibits", "COX-2", "."],
    heads=[1, 1, 1, 1],
    labels=["nsubj", "ROOT", "obj", "punct"],
)
assert "inhibits->Aspirin:nsubj" in dep_seq
assert "inhibits->COX-2:obj" in dep_seq
assert "ROOT" not in dep_seq
print(f"  OK  dep_seq: {dep_seq}")

print("=" * 60)
print("Test 4b: linearize_shortest_dependency_path")
sdp_seq = linearize_shortest_dependency_path(
    tokens=["Aspirin", "inhibits", "COX-2", "."],
    heads=[1, 1, 1, 1],
    labels=["nsubj", "ROOT", "obj", "punct"],
    source_idx=0,
    target_idx=2,
)
assert "Aspirin<-inhibits:nsubj" in sdp_seq
assert "inhibits->COX-2:obj" in sdp_seq
print(f"  OK  sdp_seq: {sdp_seq}")

print("=" * 60)
print("Test 5: BioREDataset CDR_train (text-only)")
ds = BioREDataset(f"{BASE}/CDR_train.json", use_dep=False, max_src_len=512)
assert len(ds) == 1037, f"expected 1037, got {len(ds)}"
s0 = ds[0]
assert "input_text" in s0 and "target_text" in s0
assert "[E1S]" in s0["input_text"]
assert "[DEP]" not in s0["input_text"]
assert s0["target_text"] == "CID"
print(f"  OK  n={len(ds)}  sample input:  {s0['input_text'][:80]}...")
print(f"             sample target: {s0['target_text']}")
ds.close()

print("=" * 60)
print("Test 6: BioREDataset CDR_train (raw dep)")
ds_dep = BioREDataset(
    f"{BASE}/CDR_train.json", use_dep=True, dep_view="raw", max_src_len=512
)
s0_dep = ds_dep[0]
assert "[DEP]" in s0_dep["input_text"], "DEP marker missing"
assert s0_dep["dep_view_used"] == "raw"
print(f"  OK  input with dep: {s0_dep['input_text'][:100]}...")
ds_dep.close()

print("=" * 60)
print("Test 7: DDI negative sampling stays bounded")
tiny_ddi = make_tiny_ddi_file()
ddi_ds = BioREDataset(
    tiny_ddi, use_dep=False, max_src_len=128, neg_ratio=3, seed=42
)
assert len(ddi_ds) == 8, f"expected 8 samples (2 pos + 6 neg), got {len(ddi_ds)}"
print(f"  OK  sampled DDI dataset size={len(ddi_ds)}")
ddi_ds.close()

print("=" * 60)
print("Test 8: DDI truncation keeps dep local to visible window")
import json as _json

with open(f"{BASE}/DDI_train.json", encoding="utf-8") as handle:
    for line in handle:
        record = _json.loads(line)
        if len(record.get("tokens", [])) > 900:
            long_path = tempfile.TemporaryDirectory()
            TMP_DIRS.append(long_path)
            sample_path = Path(long_path.name) / "DDI_train.json"
            with sample_path.open("w", encoding="utf-8") as out:
                out.write(_json.dumps(record, ensure_ascii=False) + "\n")
            long_ds = BioREDataset(
                str(sample_path),
                use_dep=True,
                dep_view="coarse",
                max_src_len=512,
            )
            sample = long_ds[0]
            assert sample["truncated"] is True
            assert sample["dep_view_used"] == "raw"
            print(
                f"  OK  truncated={sample['truncated']}  "
                f"dep_view_used={sample['dep_view_used']}"
            )
            long_ds.close()
            break

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
