"""Amazon Shopping Queries (ESCI) loading and evaluation-corpus construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import GAIN

RAW = Path("data/raw/esci")
BULLET_CHARS = 300  # enough for the key attributes; full bullets dilute BM25 and overflow the encoder


def load_examples(locale: str, raw: Path = RAW) -> pd.DataFrame:
    cols = ["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"]
    df = pd.read_parquet(raw / "shopping_queries_dataset_examples.parquet", columns=cols,
                         filters=[("product_locale", "=", locale)])
    return df[df["small_version"] == 1].drop(columns="small_version").reset_index(drop=True)


def load_products(locale: str, ids: set[str] | None = None, raw: Path = RAW) -> pd.DataFrame:
    cols = ["product_id", "product_title", "product_brand", "product_color", "product_bullet_point"]
    df = pd.read_parquet(raw / "shopping_queries_dataset_products.parquet", columns=cols,
                         filters=[("product_locale", "=", locale)])
    if ids is not None:
        df = df[df["product_id"].isin(ids)]
    return df.drop_duplicates("product_id").reset_index(drop=True)


def product_text(row) -> str:
    """Title, brand, colour and the start of the bullet points: the same text for BM25 and the encoders."""
    parts = [row.product_title, row.product_brand, row.product_color]
    bullets = (row.product_bullet_point or "").replace("\n", " ")
    if bullets:
        parts.append(bullets[:BULLET_CHARS])
    return " | ".join(str(p) for p in parts if p and str(p).strip() and str(p) != "None")


@dataclass
class Split:
    queries: pd.DataFrame          # query_id, query
    labels: dict[int, dict[str, float]]  # query_id -> product_id -> gain


def make_split(ex: pd.DataFrame, query_ids) -> Split:
    sub = ex[ex["query_id"].isin(set(query_ids))]
    labels: dict[int, dict[str, float]] = {}
    for qid, pid, lab in zip(sub["query_id"], sub["product_id"], sub["esci_label"]):
        labels.setdefault(int(qid), {})[pid] = GAIN[lab]
    q = sub.drop_duplicates("query_id")[["query_id", "query"]].reset_index(drop=True)
    return Split(q, labels)


def sample_queries(ex: pd.DataFrame, split: str, n: int, seed: int) -> np.ndarray:
    """Queries with at least one Exact product, so NDCG and recall are defined."""
    has_exact = ex[(ex["split"] == split) & (ex["esci_label"] == "E")]["query_id"].unique()
    rng = np.random.default_rng(seed)
    return rng.choice(np.sort(has_exact), size=min(n, len(has_exact)), replace=False)


def build_corpus(ex: pd.DataFrame, must_include: set[str], size: int, seed: int) -> list[str]:
    """Judged products of the evaluation queries plus distractors: products judged for *other* queries
    in the same locale. Distractors are real catalogue items that looked relevant to some shopper,
    so they are harder than random ones."""
    pool = np.sort(ex.loc[~ex["product_id"].isin(must_include), "product_id"].unique())
    rng = np.random.default_rng(seed)
    n_fill = max(0, min(size - len(must_include), len(pool)))
    return sorted(must_include) + sorted(rng.choice(pool, size=n_fill, replace=False).tolist())


def eval_setup(locale: str, n_test: int = 1000, n_dev: int = 300, corpus: int = 100_000, raw: Path = RAW):
    """Test queries (ESCI test split), dev queries (train split) and the shared corpus, all seeded."""
    ex = load_examples(locale, raw)
    test = make_split(ex, sample_queries(ex, "test", n_test, seed=0))
    dev = make_split(ex, sample_queries(ex, "train", n_dev, seed=1))
    judged = {p for lab in [*test.labels.values(), *dev.labels.values()] for p in lab}
    corpus_ids = build_corpus(ex, judged, corpus, seed=2)
    prods = load_products(locale, set(corpus_ids), raw).set_index("product_id").loc[corpus_ids]
    docs = [product_text(r) for r in prods.reset_index().itertuples()]
    return test, dev, corpus_ids, docs
