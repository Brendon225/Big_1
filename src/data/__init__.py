from src.data.dataset import BioREDataset, SPECIAL_TOKENS
from src.data.graph_dataset import BioREGraphDataset

__all__ = [
    "BioREDataset",
    "BioREGraphDataset",
    "SPECIAL_TOKENS",
]

# collator / dataloader 依赖 torch + transformers，懒加载
def __getattr__(name):
    if name == "BioRECollator":
        from src.data.collator import BioRECollator
        return BioRECollator
    if name == "BioREGraphCollator":
        from src.data.graph_collator import BioREGraphCollator
        return BioREGraphCollator
    if name in ("get_tokenizer", "build_dataloader"):
        import src.data.dataloader as _dl
        return getattr(_dl, name)
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
