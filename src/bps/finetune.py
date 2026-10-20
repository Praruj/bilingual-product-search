"""Training data for fine-tuning E5 and the cross-encoder on ESCI, built so it cannot touch the evaluation.

Only train-split queries are used, minus the dev queries that eval_setup samples from the train split for
tuning the hybrid weight. Test queries live in the test split, and no train query has the same text as a test
query (checked in make_finetune_data.py). Product text is the same product_text the evaluation corpus uses.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data
from .metrics import GAIN


def excluded_query_ids(ex: pd.DataFrame, n_dev: int = 300) -> set[int]:
    """The dev queries of eval_setup (same function, same seed)."""
    return {int(q) for q in data.sample_queries(ex, "train", n_dev, seed=1)}


def training_examples(locale: str, n_dev: int = 300, raw: Path = data.RAW) -> tuple[pd.DataFrame, set[int]]:
    """One row per judged (train query, product) pair, with product text and graded gain."""
    ex = data.load_examples(locale, raw)
    drop = excluded_query_ids(ex, n_dev)
    tr = ex[(ex["split"] == "train") & ~ex["query_id"].isin(drop)]
    prods = data.load_products(locale, set(tr["product_id"]), raw)
    text = {r.product_id: data.product_text(r) for r in prods.itertuples()}
    out = tr.assign(locale=locale, doc=tr["product_id"].map(text), gain=tr["esci_label"].map(GAIN))
    out = out[out["doc"].notna() & (out["doc"] != "")]
    cols = ["locale", "query_id", "query", "product_id", "esci_label", "gain", "doc"]
    return out[cols].reset_index(drop=True), drop


def assert_no_eval_leak(train: pd.DataFrame, ex: pd.DataFrame, dev_ids: set[int], n_test: int = 1000) -> None:
    """No evaluation test/dev query, and no query sharing its text with any test-split query."""
    test_ids = {int(q) for q in data.sample_queries(ex, "test", n_test, seed=0)}
    assert not set(train["query_id"]) & (test_ids | dev_ids), "eval query in training data"
    test_text = set(ex.loc[ex["split"] == "test", "query"].str.lower().str.strip())
    n_text = int(train["query"].str.lower().str.strip().isin(test_text).sum())
    assert n_text == 0, f"{n_text} training rows share query text with the test split"
