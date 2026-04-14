"""
Public helpers for tokenizer construction and torch DataLoader assembly.
"""

from pathlib import Path

from torch.utils.data import DataLoader

from src.data.collator import BioRECollator
from src.data.dataset import (
    DDI_DEFAULT_NEG_RATIO,
    BioREDataset,
    SPECIAL_TOKENS,
    infer_dataset_and_split,
)


DEFAULT_MODEL = "GanjinZero/biobart-base"

DATASET_MAX_SRC_LEN = {
    "CDR": 512,
    "ChemProt": 512,
    "DDI": 512,
}

DATASET_MAX_INPUT_LEN = {
    "CDR": 1024,
    "ChemProt": 1024,
    "DDI": 1024,
}


def get_tokenizer(model_name: str = DEFAULT_MODEL):
    """
    Load the BioBART tokenizer and register project-specific special tokens.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    print(
        f"[Tokenizer] {model_name}  vocab_size={len(tokenizer)}  "
        f"added={added} special tokens"
    )
    return tokenizer


def build_dataloader(
    file_path: str,
    tokenizer,
    split: str = "train",
    use_dep: bool = False,
    dep_view: str = "raw",
    dep_form: str = "tree",
    target_mode: str = "relation_only",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    max_src_len: int = None,
    max_input_len: int = None,
    max_target_len: int = 64,
    max_dep_arcs: int = None,
    neg_ratio: int = None,
    seed: int = 42,
) -> DataLoader:
    """
    Build a torch DataLoader around the byte-offset-backed BioREDataset.

    Notes:
    - DDI negative sampling is enabled automatically only for the train split.
    - dev/test keep the original class distribution unless the caller overrides
      neg_ratio explicitly.
    """
    dataset_name, inferred_split = infer_dataset_and_split(Path(file_path))

    if max_src_len is None:
        max_src_len = DATASET_MAX_SRC_LEN.get(dataset_name, 512)

    if max_input_len is None:
        max_input_len = DATASET_MAX_INPUT_LEN.get(dataset_name, 1024)

    if split is None:
        split = inferred_split or "train"

    if neg_ratio is None and dataset_name == "DDI" and split == "train":
        neg_ratio = DDI_DEFAULT_NEG_RATIO

    if split in ("dev", "test"):
        shuffle = False

    dataset = BioREDataset(
        file_path=file_path,
        use_dep=use_dep,
        dep_view=dep_view,
        dep_form=dep_form,
        target_mode=target_mode,
        max_src_len=max_src_len,
        max_dep_arcs=max_dep_arcs,
        neg_ratio=neg_ratio,
        seed=seed,
    )

    collator = BioRECollator(
        tokenizer=tokenizer,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    print(
        f"[DataLoader] {Path(file_path).stem}  split={split}  "
        f"use_dep={use_dep}  dep_view={dep_view}  dep_form={dep_form}  "
        f"target_mode={target_mode}  "
        f"n={len(dataset):,}  "
        f"batch={batch_size}  max_src={max_src_len}  max_bpe={max_input_len}  "
        f"neg_ratio={neg_ratio}"
    )
    return loader
