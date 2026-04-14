"""
Core dataset utilities for the BioRE experiments.

The dataset works directly on the coarsened JSONL files and is designed to:
1. keep large DDI files off memory by indexing byte offsets lazily;
2. apply DDI negative sampling only when requested;
3. keep dependency text aligned with the visible text window after truncation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Set, Tuple

from src.builders.input_builder import (
    SUPPORTED_DEP_FORMS,
    SUPPORTED_DEP_VIEWS,
    SUPPORTED_TARGET_MODES,
    build_input_text,
    build_target_text,
    linearize_dependency_arcs,
    linearize_shortest_dependency_path,
    select_dependency_sequence,
)

SPECIAL_TOKENS = ["[E1S]", "[E1E]", "[E2S]", "[E2E]", "[DEP]", "[/DEP]"]

# Negative labels currently present in the processed datasets.
NEG_LABELS = {
    "CDR": set(),
    "ChemProt": set(),
    "DDI": {"DDI-false"},
    "ChemProtSent": {"NO_RELATION"},
    "CDRIntra": {"NO_RELATION"},
}

DDI_DEFAULT_NEG_RATIO = 3
CACHE_VERSION = "v2"


def infer_dataset_and_split(file_path: str | Path) -> Tuple[str, str]:
    """Infer dataset name and split from a file path like DDI_train.json."""
    stem = Path(file_path).stem
    parts = stem.split("_", 1)
    dataset_name = parts[0]
    split_name = parts[1] if len(parts) > 1 else ""
    return dataset_name, split_name


def entity_centered_window(
    seq_len: int,
    e1_start: int,
    e1_end: int,
    e2_start: int,
    e2_end: int,
    max_len: int,
) -> Tuple[int, int]:
    """
    Compute a text window that keeps both entities visible whenever possible.

    If the entity span itself is longer than max_len, the window falls back to
    e1-centered retention and lets e2 drift to the end of the window.
    """
    if seq_len <= max_len:
        return 0, seq_len

    span_start = min(e1_start, e2_start)
    span_end = max(e1_end, e2_end)

    if span_end - span_start >= max_len:
        win_start = max(0, min(e1_start, seq_len - max_len))
        return win_start, min(seq_len, win_start + max_len)

    budget = max_len - (span_end - span_start)
    left_ctx = budget // 2
    right_ctx = budget - left_ctx

    win_start = max(0, span_start - left_ctx)
    win_end = min(seq_len, span_end + right_ctx)

    if win_start == 0:
        missing_left = max(0, left_ctx - span_start)
        win_end = min(seq_len, win_end + missing_left)
    if win_end == seq_len:
        missing_right = max(0, span_end + right_ctx - seq_len)
        win_start = max(0, win_start - missing_right)

    return win_start, win_end


def entity_centered_truncate(
    tokens: List[str],
    e1_start: int,
    e1_end: int,
    e2_start: int,
    e2_end: int,
    max_len: int,
) -> Tuple[List[str], int, int, int, int]:
    """
    Truncate the token list with an entity-centered window.

    Returns:
        (new_tokens, new_e1_start, new_e1_end, new_e2_start, new_e2_end)
    """
    win_start, win_end = entity_centered_window(
        len(tokens), e1_start, e1_end, e2_start, e2_end, max_len
    )
    new_tokens = tokens[win_start:win_end]
    n_tokens = len(new_tokens)

    if max(e1_end, e2_end) - min(e1_start, e2_start) >= max_len:
        # The span is too wide. Keep e1 exact and clamp e2 into the window tail.
        new_e1_start = max(0, e1_start - win_start)
        new_e1_end = min(n_tokens, e1_end - win_start)
        new_e2_start = min(max(0, e2_start - win_start), n_tokens - 1)
        new_e2_end = min(max(1, e2_end - win_start), n_tokens)
        if new_e2_start >= new_e2_end:
            new_e2_start = n_tokens - 1
            new_e2_end = n_tokens
        return new_tokens, new_e1_start, new_e1_end, new_e2_start, new_e2_end

    return (
        new_tokens,
        max(0, e1_start - win_start),
        min(n_tokens, e1_end - win_start),
        max(0, e2_start - win_start),
        min(n_tokens, e2_end - win_start),
    )




class BioREDataset:
    """
    Dataset backed by byte offsets instead of fully materialized samples.

    This keeps DDI files off memory while still behaving like an indexable
    dataset for torch DataLoader.
    """

    def __init__(
        self,
        file_path: str,
        use_dep: bool = False,
        dep_view: str = "raw",
        dep_form: str = "tree",
        target_mode: str = "relation_only",
        max_src_len: int = 512,
        max_dep_arcs: Optional[int] = None,
        neg_ratio: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.file_path = Path(file_path)
        self.use_dep = use_dep
        self.dep_view = dep_view
        self.dep_form = dep_form
        self.target_mode = target_mode
        self.max_src_len = max_src_len
        self.max_dep_arcs = max_dep_arcs
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.rng = random.Random(seed)

        self.dataset_name, self.split_name = infer_dataset_and_split(self.file_path)
        self.neg_label_set = NEG_LABELS.get(self.dataset_name, set())
        self.label_set: Set[str] = set()
        self.selected_offsets: List[int] = []
        self.selected_pos = 0
        self.selected_neg = 0
        self._fh: Optional[BinaryIO] = None
        self.cache_dir = self.file_path.parent / ".index_cache"

        if self.dep_view not in SUPPORTED_DEP_VIEWS:
            raise ValueError(
                f"Unsupported dep_view={self.dep_view!r}. "
                f"Expected one of {sorted(SUPPORTED_DEP_VIEWS)}."
            )
        if self.dep_form not in SUPPORTED_DEP_FORMS:
            raise ValueError(
                f"Unsupported dep_form={self.dep_form!r}. "
                f"Expected one of {sorted(SUPPORTED_DEP_FORMS)}."
            )
        if self.target_mode not in SUPPORTED_TARGET_MODES:
            raise ValueError(
                f"Unsupported target_mode={self.target_mode!r}. "
                f"Expected one of {sorted(SUPPORTED_TARGET_MODES)}."
            )

        self._build_index()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            finally:
                self._fh = None

    def _iter_offset_records(self) -> Iterable[Tuple[int, Dict]]:
        with self.file_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.strip():
                    continue
                yield offset, json.loads(raw_line.decode("utf-8"))

    def _cache_path(self) -> Path:
        neg_tag = f"neg{self.neg_ratio}" if self.neg_ratio is not None else "full"
        return self.cache_dir / f"{self.file_path.stem}.{neg_tag}.seed{self.seed}.{CACHE_VERSION}.json"

    def _load_cached_index(self) -> bool:
        cache_path = self._cache_path()
        if not cache_path.exists():
            return False

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        current_size = self.file_path.stat().st_size
        if payload.get("source_size") != current_size:
            return False

        self.selected_offsets = payload.get("selected_offsets", [])
        self.selected_pos = payload.get("selected_pos", 0)
        self.selected_neg = payload.get("selected_neg", 0)
        self.label_set = set(payload.get("labels", []))
        return True

    def _save_cached_index(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_name": self.file_path.name,
            "source_size": self.file_path.stat().st_size,
            "neg_ratio": self.neg_ratio,
            "selected_offsets": self.selected_offsets,
            "selected_pos": self.selected_pos,
            "selected_neg": self.selected_neg,
            "labels": sorted(self.label_set),
        }
        self._cache_path().write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def _build_index(self) -> None:
        if self._load_cached_index():
            print(
                f"[BioREDataset] loaded cached index: {self._cache_path().name}"
            )
            return

        if self.neg_ratio is not None and self.neg_label_set:
            self._build_sampled_index()
        else:
            self._build_full_index()

        self._save_cached_index()

        print(
            f"[BioREDataset] {self.file_path.name}  "
            f"pos={self.selected_pos:,}  neg={self.selected_neg:,}  "
            f"total={len(self.selected_offsets):,}  "
            f"use_dep={self.use_dep}  dep_view={self.dep_view}  dep_form={self.dep_form}"
        )

    def _build_full_index(self) -> None:
        pos_count = 0
        neg_count = 0

        for offset, raw in self._iter_offset_records():
            relation = raw.get("relation", "")
            self.label_set.add(relation)
            self.selected_offsets.append(offset)
            if relation in self.neg_label_set:
                neg_count += 1
            else:
                pos_count += 1

        self.selected_pos = pos_count
        self.selected_neg = neg_count

    def _build_sampled_index(self) -> None:
        pos_offsets: List[int] = []

        for offset, raw in self._iter_offset_records():
            relation = raw.get("relation", "")
            self.label_set.add(relation)
            if relation in self.neg_label_set:
                continue
            pos_offsets.append(offset)

        target_neg = len(pos_offsets) * self.neg_ratio
        reservoir: List[int] = []
        seen_neg = 0

        if target_neg > 0:
            for offset, raw in self._iter_offset_records():
                relation = raw.get("relation", "")
                if relation not in self.neg_label_set:
                    continue

                seen_neg += 1
                if len(reservoir) < target_neg:
                    reservoir.append(offset)
                    continue

                pick = self.rng.randrange(seen_neg)
                if pick < target_neg:
                    reservoir[pick] = offset

        self.selected_offsets = pos_offsets + reservoir
        if self.split_name == "train":
            self.rng.shuffle(self.selected_offsets)

        self.selected_pos = len(pos_offsets)
        self.selected_neg = len(reservoir)

    def _get_file_handle(self) -> BinaryIO:
        if self._fh is None:
            self._fh = self.file_path.open("rb")
        return self._fh

    def _read_raw_record(self, offset: int) -> Dict:
        handle = self._get_file_handle()
        handle.seek(offset)
        raw_line = handle.readline()
        return json.loads(raw_line.decode("utf-8"))

    def _select_dep_sequence(
        self,
        raw: Dict,
        window_start: int,
        window_end: int,
        truncated: bool,
        raw_e1_idx: int,
        raw_e2_idx: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        return select_dependency_sequence(
            raw=raw,
            dep_view=self.dep_view,
            dep_form=self.dep_form,
            use_dep=self.use_dep,
            truncated=truncated,
            window_start=window_start,
            window_end=window_end,
            raw_e1_idx=raw_e1_idx,
            raw_e2_idx=raw_e2_idx,
            max_dep_arcs=self.max_dep_arcs,
        )

    def _build_sample(self, raw: Dict) -> Dict:
        full_tokens = raw.get("tokens", [])
        entity1 = raw.get("entity1", {})
        entity2 = raw.get("entity2", {})
        relation = raw.get("relation", "")

        e1_start = entity1.get("start_tok", 0)
        e1_end = max(entity1.get("end_tok", e1_start + 1), e1_start + 1)
        e2_start = entity2.get("start_tok", 0)
        e2_end = max(entity2.get("end_tok", e2_start + 1), e2_start + 1)
        raw_e1_idx = int(e1_start)
        raw_e2_idx = int(e2_start)

        e1_text = entity1.get("text", " ".join(full_tokens[e1_start:e1_end]))
        e2_text = entity2.get("text", " ".join(full_tokens[e2_start:e2_end]))

        window_start, window_end = 0, len(full_tokens)
        truncated = False
        tokens = full_tokens

        if len(full_tokens) > self.max_src_len:
            window_start, window_end = entity_centered_window(
                len(full_tokens),
                e1_start,
                e1_end,
                e2_start,
                e2_end,
                self.max_src_len,
            )
            tokens, e1_start, e1_end, e2_start, e2_end = entity_centered_truncate(
                full_tokens, e1_start, e1_end, e2_start, e2_end, self.max_src_len
            )
            truncated = True

        dep_seq, dep_view_used = self._select_dep_sequence(
            raw=raw,
            window_start=window_start,
            window_end=window_end,
            truncated=truncated,
            raw_e1_idx=raw_e1_idx,
            raw_e2_idx=raw_e2_idx,
        )

        input_text = build_input_text(
            tokens=tokens,
            e1_start=e1_start,
            e1_end=e1_end,
            e2_start=e2_start,
            e2_end=e2_end,
            dep_seq=dep_seq,
        )
        target_text = build_target_text(
            e1_text=e1_text,
            e2_text=e2_text,
            label=relation,
            target_mode=self.target_mode,
        )

        return {
            "id": raw.get("id", ""),
            "dataset": raw.get("dataset", self.dataset_name),
            "split": raw.get("split", self.split_name),
            "relation": relation,
            "is_positive": raw.get("is_positive", True),
            "input_text": input_text,
            "target_text": target_text,
            "target_mode": self.target_mode,
            "e1_text": e1_text,
            "e2_text": e2_text,
            "dep_view_used": dep_view_used,
            "dep_form_used": self.dep_form if self.use_dep else "none",
            "truncated": truncated,
            "coarse_status": raw.get("coarse_status"),
        }

    def __len__(self) -> int:
        return len(self.selected_offsets)

    def __getitem__(self, idx: int) -> Dict:
        raw = self._read_raw_record(self.selected_offsets[idx])
        return self._build_sample(raw)

    def get_labels(self) -> List[str]:
        return sorted(self.label_set)
