"""Fusing sparse and dense rankings."""
from __future__ import annotations

import numpy as np

from .bm25 import top_k


def rrf(rankings: list[np.ndarray], k: int = 100, c: int = 60) -> np.ndarray:
    """Reciprocal rank fusion (Cormack et al., 2009). Uses ranks only, so no score calibration."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, doc in enumerate(r.tolist()):
            score[doc] = score.get(doc, 0.0) + 1.0 / (c + rank + 1)
    docs = np.array(list(score), dtype=np.int64)
    vals = np.array(list(score.values()))
    idx, _ = top_k(vals, k)
    return docs[idx]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def convex(sparse: np.ndarray, dense: np.ndarray, alpha: float, k: int = 100) -> np.ndarray:
    """alpha * dense + (1 - alpha) * sparse on min-max normalised full score vectors."""
    return top_k(alpha * minmax(dense) + (1 - alpha) * minmax(sparse), k)[0]
