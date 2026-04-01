#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
entity_coarsen.py
对 data/parsed/ 下的 JSON Lines 文件执行实体感知图粗化：
将多词实体折叠为超级节点，消除实体内部依存弧，
超级节点继承其 head token 的外部依存弧。
输出到 data/coarsened/。
"""

import json
import sys
import time
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[2]
PARSED_DIR    = ROOT / "data" / "parsed"
COARSENED_DIR = ROOT / "data" / "coarsened"
COARSENED_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "CDR_train.json", "CDR_dev.json", "CDR_test.json",
    "ChemProt_train.json", "ChemProt_dev.json", "ChemProt_test.json",
    "DDI_train.json", "DDI_dev.json", "DDI_test.json",
]

DUAL_ENTITY_NODE = "entity1+entity2"


def find_head_token(token_indices: list[int], dep_heads: list[int]) -> int:
    """
    在一组 token 中找到 head token：
    head token 定义为其 dep_head 指向该组之外的 token（即依存深度最浅的）。
    如果所有 token 的 head 都在组内（极少数情况），返回第一个 token。
    """
    token_set = set(token_indices)
    for idx in token_indices:
        head = dep_heads[idx]
        # head 指向实体外部，或者指向自身（ROOT 节点）
        if head not in token_set or head == idx:
            return idx
    return token_indices[0]


def build_passthrough_node_types(n: int, e1_idx: int, e2_idx: int) -> list[str]:
    """为保底透传分支构建节点类型标记，只保留两个实体代表节点。"""
    node_types = ["normal"] * n
    if 0 <= e1_idx < n and 0 <= e2_idx < n and e1_idx == e2_idx:
        node_types[e1_idx] = DUAL_ENTITY_NODE
        return node_types
    if 0 <= e1_idx < n:
        node_types[e1_idx] = "entity1"
    if 0 <= e2_idx < n:
        node_types[e2_idx] = "entity2"
    return node_types


def passthrough_record(
    rec: dict,
    tokens: list[str],
    dep_heads: list[int],
    dep_labels: list[str],
    e1_idx: int,
    e2_idx: int,
    reason: str,
) -> dict:
    """在无法安全粗化时直接透传原图，并保留实体角色信息。"""
    n = len(tokens)
    coarse_tokens = list(tokens)
    if 0 <= e1_idx < n and 0 <= e2_idx < n and e1_idx == e2_idx:
        if rec["entity1"]["text"] == rec["entity2"]["text"]:
            coarse_tokens[e1_idx] = rec["entity1"]["text"]
        else:
            coarse_tokens[e1_idx] = f"{rec['entity1']['text']} | {rec['entity2']['text']}"
    else:
        if 0 <= e1_idx < n:
            coarse_tokens[e1_idx] = rec["entity1"]["text"]
        if 0 <= e2_idx < n:
            coarse_tokens[e2_idx] = rec["entity2"]["text"]

    rec["coarse_tokens"] = coarse_tokens
    rec["coarse_heads"] = dep_heads
    rec["coarse_labels"] = dep_labels
    rec["coarse_e1_idx"] = e1_idx
    rec["coarse_e2_idx"] = e2_idx
    rec["coarse_node_types"] = build_passthrough_node_types(n, e1_idx, e2_idx)
    rec["coarse_status"] = reason
    return rec


def coarsen_record(rec: dict) -> dict:
    """
    对单条记录执行实体感知图粗化。

    输入字段：tokens, dep_heads, dep_labels, entity1, entity2
    新增字段：
      coarse_tokens    : 粗化后的 token 列表（实体被替换为单个超级节点）
      coarse_heads     : 粗化后的 dep_heads
      coarse_labels    : 粗化后的 dep_labels
      coarse_e1_idx    : entity1 超级节点在 coarse_tokens 中的索引
      coarse_e2_idx    : entity2 超级节点在 coarse_tokens 中的索引
      coarse_node_types: 每个节点的类型标记
    """
    tokens     = rec["tokens"]
    dep_heads  = rec["dep_heads"]
    dep_labels = rec["dep_labels"]
    e1         = rec["entity1"]
    e2         = rec["entity2"]

    n = len(tokens)

    # 如果依存解析结果长度不匹配，跳过粗化，直接透传
    if len(dep_heads) != n or len(dep_labels) != n:
        e1_toks = list(range(e1["start_tok"], e1["end_tok"]))
        e2_toks = list(range(e2["start_tok"], e2["end_tok"]))
        return passthrough_record(
            rec, tokens, dep_heads, dep_labels, e1["start_tok"], e2["start_tok"], "length_mismatch_passthrough"
        )

    # ── 1. 确定两个实体的 token 范围 ──────────────────────
    e1_toks = list(range(e1["start_tok"], e1["end_tok"]))
    e2_toks = list(range(e2["start_tok"], e2["end_tok"]))

    # spaCy 在生物医学连字符/突变名上可能把两个实体压成同一 token。
    # 这时继续做“实体折叠”会导致一个实体被覆盖掉，因此安全退化为原图透传。
    if set(e1_toks) & set(e2_toks):
        e1_head = find_head_token(e1_toks, dep_heads) if e1_toks else e1["start_tok"]
        e2_head = find_head_token(e2_toks, dep_heads) if e2_toks else e2["start_tok"]
        return passthrough_record(
            rec, tokens, dep_heads, dep_labels, e1_head, e2_head, "overlap_passthrough"
        )

    # 找各自的 head token
    e1_head = find_head_token(e1_toks, dep_heads) if e1_toks else e1["start_tok"]
    e2_head = find_head_token(e2_toks, dep_heads) if e2_toks else e2["start_tok"]

    # 标记哪些 token 属于哪个实体（0=普通, 1=e1, 2=e2）
    tok_entity = [0] * n
    for i in e1_toks:
        tok_entity[i] = 1
    for i in e2_toks:
        tok_entity[i] = 2

    # ── 2. 建立旧索引 → 新索引的映射 ──────────────────────
    # 规则：
    #   - 实体内部的非 head token 被删除（映射到 -1）
    #   - 实体的 head token 保留，代表整个实体（超级节点）
    #   - 普通 token 保留
    old_to_new = [-1] * n
    new_idx = 0
    new_tokens     = []
    new_node_types = []

    for i in range(n):
        ent = tok_entity[i]
        if ent == 0:
            # 普通 token，直接保留
            old_to_new[i] = new_idx
            new_tokens.append(tokens[i])
            new_node_types.append("normal")
            new_idx += 1
        elif ent == 1 and i == e1_head:
            # e1 的 head token → 超级节点，文本替换为实体全称
            old_to_new[i] = new_idx
            new_tokens.append(e1["text"])
            new_node_types.append("entity1")
            new_idx += 1
        elif ent == 2 and i == e2_head:
            # e2 的 head token → 超级节点
            old_to_new[i] = new_idx
            new_tokens.append(e2["text"])
            new_node_types.append("entity2")
            new_idx += 1
        # else: 实体内部非 head token，跳过（old_to_new[i] 保持 -1）

    # ── 3. 重建依存弧 ──────────────────────────────────────
    # 规则：
    #   - 被删除的 token（old_to_new == -1）的弧直接丢弃
    #   - 保留 token 的 head：
    #       若 head 被删除（实体内部非 head），则将 head 替换为该实体的 head token
    #   - 根节点（head == self）保持不变
    def resolve_head(orig_head: int, orig_self: int) -> int:
        """将原始 head 索引解析为粗化后的有效索引。"""
        if orig_head == orig_self:
            return orig_self  # 根节点，稍后用 new_idx 处理
        ent = tok_entity[orig_head]
        if ent == 1:
            return e1_head
        elif ent == 2:
            return e2_head
        return orig_head

    new_heads  = []
    new_labels = []
    for i in range(n):
        if old_to_new[i] == -1:
            continue  # 被删除的 token
        resolved_head = resolve_head(dep_heads[i], i)
        new_head_idx  = old_to_new[resolved_head] if resolved_head != i else old_to_new[i]
        # 如果 resolved_head 也被删除（极端情况），指向自身
        if new_head_idx == -1:
            new_head_idx = old_to_new[i]
        new_heads.append(new_head_idx)
        new_labels.append(dep_labels[i])

    # ── 4. 找到超级节点在新序列中的索引 ──────────────────
    coarse_e1_idx = old_to_new[e1_head] if e1_toks else 0
    coarse_e2_idx = old_to_new[e2_head] if e2_toks else 0

    rec["coarse_tokens"]     = new_tokens
    rec["coarse_heads"]      = new_heads
    rec["coarse_labels"]     = new_labels
    rec["coarse_e1_idx"]     = coarse_e1_idx
    rec["coarse_e2_idx"]     = coarse_e2_idx
    rec["coarse_node_types"] = new_node_types
    rec["coarse_status"]     = "coarsened"
    return rec


def process_file(src_path: Path, dst_path: Path):
    """流式处理：逐行读取、折叠、写出，避免大文件 OOM。"""
    t0 = time.time()
    total = 0
    sum_orig = sum_coarse = 0

    with open(src_path, "r", encoding="utf-8") as fin, \
         open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            orig_n = len(rec["tokens"])
            rec = coarsen_record(rec)
            coarse_n = len(rec["coarse_tokens"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total    += 1
            sum_orig   += orig_n
            sum_coarse += coarse_n
            if total % 50000 == 0:
                print(f"    {total} 条已处理...", end="\r")

    if total == 0:
        print(f"  [SKIP] {src_path.name} 为空")
        return 0

    elapsed   = time.time() - t0
    avg_orig   = sum_orig   / total
    avg_coarse = sum_coarse / total
    reduction  = (1 - avg_coarse / avg_orig) * 100 if avg_orig > 0 else 0

    print(f"  {src_path.name}: {total} 条  "
          f"平均节点 {avg_orig:.1f} -> {avg_coarse:.1f}  "
          f"(-{reduction:.1f}%)  ({elapsed:.1f}s)")
    return total


def main():
    print("=" * 60)
    print("  实体感知图粗化阶段")
    print("=" * 60)

    total_records = 0
    t_start = time.time()

    for fname in FILES:
        src = PARSED_DIR    / fname
        dst = COARSENED_DIR / fname
        if not src.exists():
            print(f"  [WARN] 文件不存在，跳过: {fname}")
            continue
        n = process_file(src, dst)
        total_records += n

    elapsed = time.time() - t_start
    print(f"\n完成！共处理 {total_records} 条记录，耗时 {elapsed:.1f}s")
    print("输出目录:", COARSENED_DIR)
    print("\n下一步: 开始搭建 Baseline 模型 src/models/")


if __name__ == "__main__":
    main()
