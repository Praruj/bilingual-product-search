"""Ranking metrics with ESCI graded relevance, and bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# gains used for ESCI ranking in KDD Cup 2022 task 1: Exact, Substitute, Complement, Irrelevant
GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def dcg(gains: Sequence[float]) -> float:
    g = np.asarray(gains, dtype=float)
    return float((g / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(ranked: Sequence[str], labels: Mapping[str, float], k: int = 10) -> float:
    """labels: product id -> gain. Unjudged products count as gain 0."""
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    if ideal == 0:
        return float("nan")
    return dcg([labels.get(p, 0.0) for p in ranked[:k]]) / ideal


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int = 100) -> float:
    if not relevant:
        return float("nan")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, resampling queries."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0) -> float:
    """One-sided p-value that mean(b) > mean(a) is a fluke, resampling the same queries for both."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[keep]
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float((means <= 0).mean())
