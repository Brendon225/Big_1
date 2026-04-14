"""
src/data/collator.py
DataCollator — 把一个 batch 的样本字典转为 BioBART 可用的 tensor。
依赖：transformers（BartTokenizerFast）
"""

from typing import List, Dict, Any
import torch


class BioRECollator:
    """
    参数
    ----
    tokenizer       : 已扩展特殊 token 的 BioBART tokenizer
    max_input_len   : encoder 最大长度（BPE tokens）
    max_target_len  : decoder 最大长度（BPE tokens），目标串通常很短
    """

    def __init__(
        self,
        tokenizer,
        max_input_len:  int = 1024,
        max_target_len: int = 64,
    ):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_texts  = [s["input_text"]  for s in batch]
        target_texts = [s["target_text"] for s in batch]

        # encoder 编码
        enc = self.tokenizer(
            input_texts,
            max_length=self.max_input_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # decoder 编码（labels）
        # 优先使用新式 text_target 接口；若 tokenizer 不支持，则退回普通编码。
        try:
            dec = self.tokenizer(
                text_target=target_texts,
                max_length=self.max_target_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        except TypeError:
            dec = self.tokenizer(
                target_texts,
                max_length=self.max_target_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        labels = dec["input_ids"].clone()
        # 将 padding token 替换为 -100（不计入损失）
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
            # 保留元数据（不参与训练，供评估使用）
            "meta": [
                {
                    "id":       s["id"],
                    "dataset":  s["dataset"],
                    "split":    s["split"],
                    "relation": s["relation"],
                    "e1_text":  s["e1_text"],
                    "e2_text":  s["e2_text"],
                    "target":   s["target_text"],
                    "dep_view_used": s.get("dep_view_used"),
                    "truncated": s.get("truncated", False),
                    "coarse_status": s.get("coarse_status"),
                }
                for s in batch
            ],
        }
