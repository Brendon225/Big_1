#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_and_convert.py (修复版)
从可用数据源下载三个生物医学关系抽取数据集，统一转换为 JSON Lines 格式。

数据源：
- CDR: HuggingFace bigbio/bc5cdr 的原始 zip 文件（BioC XML 格式）
- ChemProt: HuggingFace bigbio/chemprot 的 shared_task_eval_source 配置
- DDI: GitHub DDICorpus-2013 的 BRAT 格式 zip 文件
"""

import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlretrieve

# ── 路径设置 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────────

def simple_tokenize(text):
    """简单空格+标点分词"""
    return re.findall(r"\w+|[^\w\s]", text)


def char_to_tok(tokens, start_char, end_char, sentence):
    """字符偏移 → token 偏移"""
    pos = 0
    start_tok, end_tok = None, None
    for i, tok in enumerate(tokens):
        tok_start = sentence.find(tok, pos)
        if tok_start == -1:
            tok_start = pos
        tok_end = tok_start + len(tok)
        if start_tok is None and tok_start <= start_char < tok_end:
            start_tok = i
        if tok_start < end_char:
            end_tok = i + 1
        pos = tok_end
    return start_tok or 0, end_tok or len(tokens)


def make_id(dataset, split, idx):
    return f"{dataset}_{split}_{idx:05d}"


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  写入 {len(records):>5} 条 -> {path.name}")


def download_file(url, dest):
    """下载文件，带进度提示"""
    if dest.exists():
        print(f"  文件已存在，跳过下载: {dest.name}")
        return
    print(f"  下载中: {url}")
    print(f"  保存至: {dest}")
    urlretrieve(url, dest)
    print(f"  下载完成: {dest.stat().st_size / 1024 / 1024:.1f} MB")


# ── CDR 转换 ──────────────────────────────────────────────

def convert_cdr():
    """
    从 HuggingFace 下载 CDR_Data.zip，解析 BioC XML 格式。
    """
    print("\n[CDR] 下载数据集...")
    url = "https://huggingface.co/datasets/bigbio/bc5cdr/resolve/main/CDR_Data.zip"
    zip_path = RAW_DIR / "CDR_Data.zip"
    download_file(url, zip_path)

    print("[CDR] 解压并解析 BioC XML...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR / "CDR")

    # BioC XML 文件路径
    files = {
        "train": RAW_DIR / "CDR" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml",
        "dev":   RAW_DIR / "CDR" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_DevelopmentSet.BioC.xml",
        "test":  RAW_DIR / "CDR" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.BioC.xml",
    }

    results = {}
    for split_name, xml_path in files.items():
        if not xml_path.exists():
            print(f"  [WARN] {split_name} 文件不存在: {xml_path}")
            continue

        records = []
        tree = ET.parse(xml_path)
        root = tree.getroot()

        idx = 0
        for doc in root.findall("document"):
            # 1. 拼接所有 passage 文本，记录每个 passage 的起始偏移
            passages_info = []
            for passage in doc.findall("passage"):
                offset_elem = passage.find("offset")
                base_offset = int(offset_elem.text) if offset_elem is not None else 0
                text_elem = passage.find("text")
                p_text = text_elem.text if text_elem is not None else ""
                passages_info.append((base_offset, p_text))

            if not passages_info:
                continue

            # 按 offset 排序，拼接成文档全文
            passages_info.sort(key=lambda x: x[0])
            # 用空格连接，记录每段在 full_text 中的起始位置
            full_text = ""
            seg_starts = []   # (在 full_text 中的起始, 原始 base_offset)
            for base_off, p_text in passages_info:
                seg_starts.append((len(full_text), base_off))
                full_text += p_text + " "
            full_text = full_text.rstrip()

            def doc_offset(orig_off):
                """将原始 BioC offset 转换为 full_text 中的字符位置"""
                best_seg_start, best_base = 0, 0
                for seg_start, base in seg_starts:
                    if base <= orig_off:
                        best_seg_start, best_base = seg_start, base
                    else:
                        break
                return best_seg_start + (orig_off - best_base)

            # 2. 提取实体：annotation 在 passage 内，用 MESH ID 建立映射
            #    同一 MESH ID 可能对应多个 annotation（同义词），取第一个
            mesh_to_ents = {}   # MESH_ID -> list of entity dicts
            for passage in doc.findall("passage"):
                for ann in passage.findall("annotation"):
                    infons = {inf.get("key"): inf.text
                              for inf in ann.findall("infon")}
                    ent_type = infons.get("type")
                    if ent_type not in ("Chemical", "Disease"):
                        continue
                    mesh_id = infons.get("MESH", "")
                    loc = ann.find("location")
                    if loc is None:
                        continue
                    orig_off = int(loc.get("offset", 0))
                    length   = int(loc.get("length", 0))
                    text_elem = ann.find("text")
                    ent_text  = text_elem.text if text_elem is not None else ""
                    ent = {
                        "type":   ent_type,
                        "text":   ent_text,
                        "offset": doc_offset(orig_off),
                        "length": length,
                    }
                    mesh_to_ents.setdefault(mesh_id, []).append(ent)

            # 3. 提取关系：relation 在 document 级，用 MESH ID 关联
            for rel in doc.findall("relation"):
                infons = {inf.get("key"): inf.text for inf in rel.findall("infon")}
                if infons.get("relation") != "CID":
                    continue
                chem_mesh = infons.get("Chemical", "")
                dis_mesh  = infons.get("Disease",  "")
                if chem_mesh not in mesh_to_ents or dis_mesh not in mesh_to_ents:
                    continue

                e1 = mesh_to_ents[chem_mesh][0]
                e2 = mesh_to_ents[dis_mesh][0]
                sent   = full_text
                tokens = simple_tokenize(sent)

                e1_sc = max(0, min(e1["offset"], len(sent)))
                e1_ec = max(e1_sc, min(e1["offset"] + e1["length"], len(sent)))
                e2_sc = max(0, min(e2["offset"], len(sent)))
                e2_ec = max(e2_sc, min(e2["offset"] + e2["length"], len(sent)))

                e1_st, e1_et = char_to_tok(tokens, e1_sc, e1_ec, sent)
                e2_st, e2_et = char_to_tok(tokens, e2_sc, e2_ec, sent)

                records.append({
                    "id": make_id("CDR", split_name, idx),
                    "dataset": "CDR",
                    "split": split_name,
                    "sentence": sent,
                    "tokens": tokens,
                    "entity1": {
                        "text": e1["text"],
                        "type": e1["type"],
                        "start_char": e1_sc,
                        "end_char": e1_ec,
                        "start_tok": e1_st,
                        "end_tok": e1_et,
                    },
                    "entity2": {
                        "text": e2["text"],
                        "type": e2["type"],
                        "start_char": e2_sc,
                        "end_char": e2_ec,
                        "start_tok": e2_st,
                        "end_tok": e2_et,
                    },
                    "relation": "CID",
                    "is_positive": True,
                })
                idx += 1

        results[split_name] = records
        write_jsonl(records, PROCESSED_DIR / f"CDR_{split_name}.json")

    return results


# ── ChemProt 转换 ─────────────────────────────────────────

def convert_chemprot():
    """
    从 HuggingFace 加载 bigbio/chemprot 的 shared_task_eval_source 配置。
    """
    from datasets import load_dataset

    print("\n[ChemProt] 加载数据集...")
    ds = load_dataset("bigbio/chemprot", name="chemprot_shared_task_eval_source")

    POSITIVE_LABELS = {"CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"}

    results = {}
    for split_name, split_key in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        records = []
        for idx, ex in enumerate(ds[split_key]):
            text = ex["text"]
            tokens = simple_tokenize(text)
            entities = ex["entities"]
            relations = ex["relations"]

            # 构建实体 ID 映射
            ent_map = {}
            for i, ent_id in enumerate(entities["id"]):
                ent_map[ent_id] = {
                    "text": entities["text"][i],
                    "type": entities["type"][i],
                    "offset": entities["offsets"][i],
                }

            # 处理关系
            for i, rel_type in enumerate(relations["type"]):
                e1_id = relations["arg1"][i]
                e2_id = relations["arg2"][i]
                if e1_id not in ent_map or e2_id not in ent_map:
                    continue

                e1, e2 = ent_map[e1_id], ent_map[e2_id]
                is_pos = rel_type in POSITIVE_LABELS

                e1_sc, e1_ec = e1["offset"]
                e2_sc, e2_ec = e2["offset"]
                e1_sc = max(0, min(e1_sc, len(text)))
                e1_ec = max(e1_sc, min(e1_ec, len(text)))
                e2_sc = max(0, min(e2_sc, len(text)))
                e2_ec = max(e2_sc, min(e2_ec, len(text)))

                e1_st, e1_et = char_to_tok(tokens, e1_sc, e1_ec, text)
                e2_st, e2_et = char_to_tok(tokens, e2_sc, e2_ec, text)

                records.append({
                    "id": make_id("ChemProt", split_name, len(records)),
                    "dataset": "ChemProt",
                    "split": split_name,
                    "sentence": text,
                    "tokens": tokens,
                    "entity1": {
                        "text": e1["text"],
                        "type": e1["type"],
                        "start_char": e1_sc,
                        "end_char": e1_ec,
                        "start_tok": e1_st,
                        "end_tok": e1_et,
                    },
                    "entity2": {
                        "text": e2["text"],
                        "type": e2["type"],
                        "start_char": e2_sc,
                        "end_char": e2_ec,
                        "start_tok": e2_st,
                        "end_tok": e2_et,
                    },
                    "relation": rel_type,
                    "is_positive": is_pos,
                })

        results[split_name] = records
        write_jsonl(records, PROCESSED_DIR / f"ChemProt_{split_name}.json")

    return results


# ── DDI 转换 ──────────────────────────────────────────────

def convert_ddi():
    """
    从 GitHub 下载 DDICorpus-2013 BRAT 格式 zip，解析 .ann 文件。
    """
    print("\n[DDI] 下载数据集...")
    url = "https://github.com/isegura/DDICorpus/raw/master/DDICorpus-2013(BRAT).zip"
    zip_path = RAW_DIR / "DDICorpus-2013-BRAT.zip"
    download_file(url, zip_path)

    print("[DDI] 解压并解析 BRAT 格式...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR / "DDI")

    # BRAT 格式：.txt 文件存文本，.ann 文件存标注
    base_dir = RAW_DIR / "DDI" / "DDICorpusBrat"
    train_dir = base_dir / "Train"
    test_dir = base_dir / "Test"

    # DDI 关系类型映射（BRAT 原始标签 → 标准标签）
    REL_MAP = {
        "MECHANISM": "DDI-mechanism",
        "EFFECT":    "DDI-effect",
        "ADVISE":    "DDI-advise",
        "INT":       "DDI-int",
    }

    def parse_brat_dir(dir_path):
        records_local = []
        for txt_file in dir_path.rglob("*.txt"):
            ann_file = txt_file.with_suffix(".ann")
            if not ann_file.exists():
                continue

            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = simple_tokenize(text)

            # 解析 .ann 文件
            entities = {}
            pos_pairs = set()   # 已有正例关系的实体对 (id1, id2)
            relations = []
            with open(ann_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if line.startswith("T") and len(parts) >= 3:
                        ent_id = parts[0]
                        type_span = parts[1].split()
                        if len(type_span) < 3:
                            continue
                        ent_type = type_span[0]
                        # 处理不连续 span，如 "700 710;726 747"
                        # 取第一段 start 和最后一段 end
                        span_str = " ".join(type_span[1:])
                        spans = [s.strip() for s in span_str.replace(";", " ").split()]
                        start = int(spans[0])
                        end   = int(spans[-1])
                        ent_text = parts[2]
                        entities[ent_id] = {
                            "type": ent_type,
                            "start": start,
                            "end":   end,
                            "text":  ent_text,
                        }
                    elif line.startswith("R"):
                        rparts = line.split()
                        if len(rparts) < 4:
                            continue
                        raw_type = rparts[1]
                        arg1 = rparts[2].split(":")[1] if ":" in rparts[2] else rparts[2]
                        arg2 = rparts[3].split(":")[1] if ":" in rparts[3] else rparts[3]
                        rel_type = REL_MAP.get(raw_type, raw_type)
                        relations.append((rel_type, arg1, arg2))
                        pos_pairs.add((arg1, arg2))
                        pos_pairs.add((arg2, arg1))

            def make_rec(e1, e2, rel_type, is_pos):
                e1_sc = max(0, min(e1["start"], len(text)))
                e1_ec = max(e1_sc, min(e1["end"], len(text)))
                e2_sc = max(0, min(e2["start"], len(text)))
                e2_ec = max(e2_sc, min(e2["end"], len(text)))
                e1_st, e1_et = char_to_tok(tokens, e1_sc, e1_ec, text)
                e2_st, e2_et = char_to_tok(tokens, e2_sc, e2_ec, text)
                return {
                    "sentence": text,
                    "tokens": tokens,
                    "entity1": {
                        "text": e1["text"], "type": e1["type"],
                        "start_char": e1_sc, "end_char": e1_ec,
                        "start_tok": e1_st,  "end_tok": e1_et,
                    },
                    "entity2": {
                        "text": e2["text"], "type": e2["type"],
                        "start_char": e2_sc, "end_char": e2_ec,
                        "start_tok": e2_st,  "end_tok": e2_et,
                    },
                    "relation": rel_type,
                    "is_positive": is_pos,
                }

            # 正例关系
            for rel_type, e1_id, e2_id in relations:
                if e1_id not in entities or e2_id not in entities:
                    continue
                records_local.append(
                    make_rec(entities[e1_id], entities[e2_id], rel_type, True)
                )

            # 负例：所有未出现在正例中的实体对
            ent_ids = list(entities.keys())
            for i in range(len(ent_ids)):
                for j in range(i + 1, len(ent_ids)):
                    a, b = ent_ids[i], ent_ids[j]
                    if (a, b) not in pos_pairs:
                        records_local.append(
                            make_rec(entities[a], entities[b], "DDI-false", False)
                        )

        return records_local

    train_raw = parse_brat_dir(train_dir)
    test_raw = parse_brat_dir(test_dir)

    # 从 train 切出 10% 作为 dev
    import random
    random.seed(42)
    random.shuffle(train_raw)
    dev_size = max(1, int(len(train_raw) * 0.1))
    dev_raw = train_raw[:dev_size]
    train_raw = train_raw[dev_size:]

    results = {}
    for split_name, raw in [("train", train_raw), ("dev", dev_raw), ("test", test_raw)]:
        records = []
        for idx, rec in enumerate(raw):
            rec["id"] = make_id("DDI", split_name, idx)
            rec["dataset"] = "DDI"
            rec["split"] = split_name
            records.append(rec)
        results[split_name] = records
        write_jsonl(records, PROCESSED_DIR / f"DDI_{split_name}.json")

    return results


# ── 统计摘要 ──────────────────────────────────────────────

def compute_stats(all_results):
    stats = {}
    for dataset_name, splits in all_results.items():
        stats[dataset_name] = {}
        for split_name, records in splits.items():
            pos = sum(1 for r in records if r["is_positive"])
            neg = len(records) - pos
            rel_dist = {}
            for r in records:
                rel_dist[r["relation"]] = rel_dist.get(r["relation"], 0) + 1
            stats[dataset_name][split_name] = {
                "total_pairs": len(records),
                "positive": pos,
                "negative": neg,
                "relation_distribution": rel_dist,
            }
    return stats


# ── 主函数 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  生物医学关系抽取数据集下载与转换（修复版）")
    print("=" * 60)

    # 设置 HuggingFace 镜像（可选）
    # os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    all_results = {}

    try:
        all_results["CDR"] = convert_cdr()
    except Exception as e:
        print(f"[ERROR] CDR 转换失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results["ChemProt"] = convert_chemprot()
    except Exception as e:
        print(f"[ERROR] ChemProt 转换失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results["DDI"] = convert_ddi()
    except Exception as e:
        print(f"[ERROR] DDI 转换失败: {e}")
        import traceback
        traceback.print_exc()

    # 生成统计摘要
    if all_results:
        stats = compute_stats(all_results)
        stats_path = PROCESSED_DIR / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n统计摘要已写入 -> {stats_path.name}")

        print("\n" + "=" * 60)
        print("  数据集统计摘要")
        print("=" * 60)
        for ds_name, splits in stats.items():
            print(f"\n[{ds_name}]")
            for sp, info in splits.items():
                print(f"  {sp:6s}: {info['total_pairs']:>5} 对  "
                      f"(正例 {info['positive']:>4} / 负例 {info['negative']:>4})")

    print("\n完成！请运行 verify_data.py 进行验证。")


if __name__ == "__main__":
    main()
