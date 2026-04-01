#!/usr/bin/env python3
"""
Build experiment-ready sentence-level views for Stage 1.

Outputs:
  data/experiment_views/processed/
    - ChemProtSent_{train,dev,test}.json
    - CDRIntra_{train,dev,test}.json

  docs/data_docs/experiment_view_summary.json
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import spacy
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VIEW_DIR = DATA_DIR / "experiment_views"
PROCESSED_DIR = VIEW_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = ROOT / "docs" / "data_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

CHEMPROT_POSITIVE = {"CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"}
NO_RELATION = "NO_RELATION"


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text)


def char_to_tok(tokens: Sequence[str], start_char: int, end_char: int, text: str) -> Tuple[int, int]:
    pos = 0
    start_tok, end_tok = None, None
    for idx, tok in enumerate(tokens):
        tok_start = text.find(tok, pos)
        if tok_start == -1:
            tok_start = pos
        tok_end = tok_start + len(tok)
        if start_tok is None and tok_start <= start_char < tok_end:
            start_tok = idx
        if tok_start < end_char:
            end_tok = idx + 1
        pos = tok_end
    return start_tok or 0, end_tok or len(tokens)


def write_jsonl(records: Iterable[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_sentencizer():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def sentence_spans(nlp, text: str) -> List[Tuple[int, int, str]]:
    doc = nlp(text)
    return [(sent.start_char, sent.end_char, sent.text) for sent in doc.sents]


def locate_sentence(spans: Sequence[Tuple[int, int, str]], start_char: int, end_char: int) -> Optional[int]:
    for idx, (sent_start, sent_end, _sent_text) in enumerate(spans):
        if sent_start <= start_char and end_char <= sent_end:
            return idx
    return None


def find_chemprot_cache_dir() -> Path:
    base = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "bigbio___chemprot"
        / "chemprot_shared_task_eval_source"
        / "0.0.0"
    )
    if not base.exists():
        raise FileNotFoundError("ChemProt cache directory not found.")

    candidates = [path for path in base.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError("No ChemProt cached configuration directory found.")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def load_chemprot_split(split_name: str) -> Dataset:
    cache_dir = find_chemprot_cache_dir()
    suffix = {
        "train": "chemprot-train.arrow",
        "dev": "chemprot-validation.arrow",
        "test": "chemprot-test.arrow",
    }[split_name]
    return Dataset.from_file(str(cache_dir / suffix))


def build_chemprot_sent_view(nlp) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}

    for split in ("train", "dev", "test"):
        ds = load_chemprot_split(split)
        records: List[Dict] = []
        skipped_cross_sentence_entities = 0
        total_entities = 0
        total_gold_pos = 0
        same_sentence_gold = 0

        for ex in ds:
            text = ex["text"]
            pmid = str(ex["pmid"])
            sent_spans = sentence_spans(nlp, text)

            ent_map: Dict[str, Dict] = {}
            sentence_entities: Dict[int, List[Dict]] = {}

            entities = ex["entities"]
            for idx, ent_id in enumerate(entities["id"]):
                total_entities += 1
                start_char, end_char = entities["offsets"][idx]
                sent_idx = locate_sentence(sent_spans, start_char, end_char)
                ent = {
                    "id": ent_id,
                    "text": entities["text"][idx],
                    "type": entities["type"][idx],
                    "start_char": start_char,
                    "end_char": end_char,
                    "sent_idx": sent_idx,
                }
                ent_map[ent_id] = ent
                if sent_idx is None:
                    skipped_cross_sentence_entities += 1
                    continue
                sentence_entities.setdefault(sent_idx, []).append(ent)

            gold_map: Dict[Tuple[str, str], str] = {}
            relations = ex["relations"]
            for idx, rel_type in enumerate(relations["type"]):
                if rel_type not in CHEMPROT_POSITIVE:
                    continue
                total_gold_pos += 1
                arg1 = relations["arg1"][idx]
                arg2 = relations["arg2"][idx]
                ent1 = ent_map.get(arg1)
                ent2 = ent_map.get(arg2)
                if ent1 is None or ent2 is None:
                    continue
                if ent1.get("sent_idx") is not None and ent1.get("sent_idx") == ent2.get("sent_idx"):
                    same_sentence_gold += 1
                gold_map[(arg1, arg2)] = rel_type

            for sent_idx, ents in sentence_entities.items():
                sent_start, _sent_end, sent_text = sent_spans[sent_idx]
                local_tokens = simple_tokenize(sent_text)
                chems = [e for e in ents if e["type"] == "CHEMICAL"]
                genes = [e for e in ents if e["type"].startswith("GENE")]

                for chem, gene in product(chems, genes):
                    local_e1_sc = chem["start_char"] - sent_start
                    local_e1_ec = chem["end_char"] - sent_start
                    local_e2_sc = gene["start_char"] - sent_start
                    local_e2_ec = gene["end_char"] - sent_start
                    e1_st, e1_et = char_to_tok(local_tokens, local_e1_sc, local_e1_ec, sent_text)
                    e2_st, e2_et = char_to_tok(local_tokens, local_e2_sc, local_e2_ec, sent_text)
                    label = gold_map.get((chem["id"], gene["id"]), NO_RELATION)
                    is_positive = label != NO_RELATION

                    records.append(
                        {
                            "id": f"ChemProtSent_{split}_{len(records):05d}",
                            "dataset": "ChemProtSent",
                            "source_dataset": "ChemProt",
                            "split": split,
                            "view_name": "same_sentence_local",
                            "sentence": sent_text,
                            "tokens": local_tokens,
                            "source_doc_id": pmid,
                            "source_sent_id": sent_idx,
                            "entity1": {
                                "text": chem["text"],
                                "type": chem["type"],
                                "entity_id": chem["id"],
                                "start_char": local_e1_sc,
                                "end_char": local_e1_ec,
                                "start_tok": e1_st,
                                "end_tok": e1_et,
                            },
                            "entity2": {
                                "text": gene["text"],
                                "type": gene["type"],
                                "entity_id": gene["id"],
                                "start_char": local_e2_sc,
                                "end_char": local_e2_ec,
                                "start_tok": e2_st,
                                "end_tok": e2_et,
                            },
                            "relation": label,
                            "is_positive": is_positive,
                        }
                    )

        out_path = PROCESSED_DIR / f"ChemProtSent_{split}.json"
        write_jsonl(records, out_path)
        pos = sum(1 for record in records if record["is_positive"])
        neg = len(records) - pos
        summary[split] = {
            "records": len(records),
            "positive": pos,
            "negative": neg,
            "total_gold_positive_relations": total_gold_pos,
            "same_sentence_gold_positive_relations": same_sentence_gold,
            "same_sentence_gold_ratio": round(same_sentence_gold / total_gold_pos, 4) if total_gold_pos else 0.0,
            "total_entities": total_entities,
            "skipped_cross_sentence_entities": skipped_cross_sentence_entities,
            "output_file": str(out_path),
        }

    return summary


def build_cdr_intra_view(nlp) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    split_to_file = {
        "train": RAW_DIR / "CDR" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml",
        "dev": RAW_DIR / "CDR" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_DevelopmentSet.BioC.xml",
        "test": RAW_DIR / "CDR" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.BioC.xml",
    }

    for split, xml_path in split_to_file.items():
        root = ET.parse(xml_path).getroot()
        records: List[Dict] = []
        doc_level_relations = 0
        doc_relations_with_same_sentence_support = 0

        for doc in root.findall("document"):
            doc_id = doc.findtext("id", default="")
            passages_info = []
            for passage in doc.findall("passage"):
                base_offset = int(passage.findtext("offset", default="0"))
                passage_text = passage.findtext("text", default="")
                passages_info.append((base_offset, passage_text))
            if not passages_info:
                continue

            passages_info.sort(key=lambda item: item[0])
            full_text = ""
            seg_starts = []
            for base_offset, passage_text in passages_info:
                seg_starts.append((len(full_text), base_offset))
                full_text += passage_text + " "
            full_text = full_text.rstrip()

            def doc_offset(orig_offset: int) -> int:
                best_seg_start, best_base = 0, 0
                for seg_start, base in seg_starts:
                    if base <= orig_offset:
                        best_seg_start, best_base = seg_start, base
                    else:
                        break
                return best_seg_start + (orig_offset - best_base)

            sent_spans = sentence_spans(nlp, full_text)
            sentence_entities: Dict[int, List[Dict]] = {}

            for passage in doc.findall("passage"):
                for ann in passage.findall("annotation"):
                    infons = {inf.get("key"): inf.text for inf in ann.findall("infon")}
                    ent_type = infons.get("type")
                    if ent_type not in ("Chemical", "Disease"):
                        continue
                    mesh_id = infons.get("MESH", "")
                    loc = ann.find("location")
                    if loc is None:
                        continue
                    orig_off = int(loc.get("offset", 0))
                    length = int(loc.get("length", 0))
                    ent_text = ann.findtext("text", default="")
                    start_char = doc_offset(orig_off)
                    end_char = start_char + length
                    sent_idx = locate_sentence(sent_spans, start_char, end_char)
                    mention = {
                        "text": ent_text,
                        "type": ent_type,
                        "mesh_id": mesh_id,
                        "start_char": start_char,
                        "end_char": end_char,
                        "sent_idx": sent_idx,
                    }
                    if sent_idx is not None:
                        sentence_entities.setdefault(sent_idx, []).append(mention)

            positive_mesh_pairs = set()
            for rel in doc.findall("relation"):
                infons = {inf.get("key"): inf.text for inf in rel.findall("infon")}
                if infons.get("relation") != "CID":
                    continue
                chem_mesh = infons.get("Chemical", "")
                disease_mesh = infons.get("Disease", "")
                if not chem_mesh or not disease_mesh:
                    continue
                doc_level_relations += 1
                positive_mesh_pairs.add((chem_mesh, disease_mesh))

            supported_pairs = set()
            for ents in sentence_entities.values():
                chems = [e for e in ents if e["type"] == "Chemical"]
                diseases = [e for e in ents if e["type"] == "Disease"]
                for chem, disease in product(chems, diseases):
                    if (chem["mesh_id"], disease["mesh_id"]) in positive_mesh_pairs:
                        supported_pairs.add((chem["mesh_id"], disease["mesh_id"]))
            doc_relations_with_same_sentence_support += len(supported_pairs)

            for sent_idx, ents in sentence_entities.items():
                sent_start, _sent_end, sent_text = sent_spans[sent_idx]
                local_tokens = simple_tokenize(sent_text)
                chems = [e for e in ents if e["type"] == "Chemical"]
                diseases = [e for e in ents if e["type"] == "Disease"]

                for chem, disease in product(chems, diseases):
                    local_e1_sc = chem["start_char"] - sent_start
                    local_e1_ec = chem["end_char"] - sent_start
                    local_e2_sc = disease["start_char"] - sent_start
                    local_e2_ec = disease["end_char"] - sent_start
                    e1_st, e1_et = char_to_tok(local_tokens, local_e1_sc, local_e1_ec, sent_text)
                    e2_st, e2_et = char_to_tok(local_tokens, local_e2_sc, local_e2_ec, sent_text)
                    label = "CID" if (chem["mesh_id"], disease["mesh_id"]) in positive_mesh_pairs else NO_RELATION
                    is_positive = label != NO_RELATION

                    records.append(
                        {
                            "id": f"CDRIntra_{split}_{len(records):05d}",
                            "dataset": "CDRIntra",
                            "source_dataset": "CDR",
                            "split": split,
                            "view_name": "same_sentence_local",
                            "sentence": sent_text,
                            "tokens": local_tokens,
                            "source_doc_id": doc_id,
                            "source_sent_id": sent_idx,
                            "entity1": {
                                "text": chem["text"],
                                "type": chem["type"],
                                "mesh_id": chem["mesh_id"],
                                "start_char": local_e1_sc,
                                "end_char": local_e1_ec,
                                "start_tok": e1_st,
                                "end_tok": e1_et,
                            },
                            "entity2": {
                                "text": disease["text"],
                                "type": disease["type"],
                                "mesh_id": disease["mesh_id"],
                                "start_char": local_e2_sc,
                                "end_char": local_e2_ec,
                                "start_tok": e2_st,
                                "end_tok": e2_et,
                            },
                            "relation": label,
                            "is_positive": is_positive,
                        }
                    )

        out_path = PROCESSED_DIR / f"CDRIntra_{split}.json"
        write_jsonl(records, out_path)
        pos = sum(1 for record in records if record["is_positive"])
        neg = len(records) - pos
        summary[split] = {
            "records": len(records),
            "positive": pos,
            "negative": neg,
            "doc_level_gold_relations": doc_level_relations,
            "doc_relations_with_same_sentence_support": doc_relations_with_same_sentence_support,
            "same_sentence_support_ratio": round(doc_relations_with_same_sentence_support / doc_level_relations, 4) if doc_level_relations else 0.0,
            "output_file": str(out_path),
        }

    return summary


def write_build_summary(summary: Dict[str, Dict]) -> None:
    out_path = DOCS_DIR / "experiment_view_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("  Build Stage 1 experiment views")
    print("=" * 60)

    nlp = make_sentencizer()

    print("\n[1/2] Building ChemProtSent ...")
    chemprot_summary = build_chemprot_sent_view(nlp)
    print("[DONE] ChemProtSent")

    print("\n[2/2] Building CDRIntra ...")
    cdr_summary = build_cdr_intra_view(nlp)
    print("[DONE] CDRIntra")

    summary = {
        "ChemProtSent": chemprot_summary,
        "CDRIntra": cdr_summary,
    }
    write_build_summary(summary)

    print("\nOutput:")
    print("  ", PROCESSED_DIR)
    print("  ", DOCS_DIR / "experiment_view_summary.json")


if __name__ == "__main__":
    main()