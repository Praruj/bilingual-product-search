"""LLM relevance judge: which ESCI pairs it labels, the prompt, and how its labels are scored against humans.

Two sets of (query, product) pairs per locale:
  view     every judged product of the first n_view evaluation test queries, in the same order as the
           judged-candidate ("rerank_judged") view of run_search.py, so LLM scores can rerank them and be
           compared query by query with E5 and the cross-encoders. Also the held-out set for label agreement.
  distill  every judged product of a seeded sample of training queries (train split minus the dev queries,
           via finetune.training_examples), which the LLM labels to train a cross-encoder without human labels.
Nothing here needs a GPU; the model code is in scripts/llm_judge.py.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import data
from .metrics import GAIN

LABELS = ["E", "S", "C", "I"]

SYSTEM = (
    "You judge search results for an online shop. Given a shopping query and a product, classify the product:\n"
    "E = Exact: relevant to the query and satisfies all of its specifications.\n"
    "S = Substitute: fails some aspect of the query but could be used as a functional substitute.\n"
    "C = Complement: does not fulfil the query but could be used together with an exact item.\n"
    "I = Irrelevant: irrelevant, or fails a central aspect of the query.\n"
    "Queries and products may be in Japanese or English. Answer with one letter: E, S, C or I."
)
DOC_CHARS = 400  # the product text the LLM sees (title | brand | colour | bullets), cut for speed


def user_message(query: str, doc: str) -> str:
    return f"Query: {query}\nProduct: {doc[:DOC_CHARS]}\nLabel:"


def view_queries(ex: pd.DataFrame, n_test: int = 1000, n_view: int = 200) -> pd.DataFrame:
    """The first n_view test queries in run_search.py's order (data.make_split of the seeded test sample)."""
    test = data.make_split(ex, data.sample_queries(ex, "test", n_test, seed=0))
    return test.queries.iloc[:n_view].reset_index(drop=True)


def view_pairs(ex: pd.DataFrame, n_test: int = 1000, n_view: int = 200) -> pd.DataFrame:
    """All judged products of the view queries; `qorder` keeps run_search.py's query order."""
    q = view_queries(ex, n_test, n_view)
    order = {int(qid): i for i, qid in enumerate(q["query_id"])}
    sub = ex[ex["query_id"].isin(order)].copy()
    sub["qorder"] = sub["query_id"].map(order)
    return sub.sort_values(["qorder", "product_id"], kind="stable").reset_index(drop=True)


def distill_query_ids(train: pd.DataFrame, n_queries: int, seed: int = 3) -> list[int]:
    """Seeded sample of whole training queries (all their judged pairs are labelled), queries with an Exact
    product only, like every other sample in the project."""
    has_exact = np.sort(train.loc[train["esci_label"] == "E", "query_id"].unique())
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(has_exact, size=min(n_queries, len(has_exact)), replace=False))


def fingerprint(df: pd.DataFrame) -> str:
    """Order-independent hash of the pair set and its text, to check the Colab VM built the same file."""
    rows = sorted(f"{r.set}|{r.locale}|{r.query_id}|{r.product_id}|{r.esci_label}|{r.query}|{r.doc}"
                  for r in df.itertuples())
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


# --- scoring the judge


def probs_to_label(p: np.ndarray) -> np.ndarray:
    """p: n x 4 probabilities in LABELS order -> argmax label."""
    return np.array(LABELS)[np.asarray(p).argmax(axis=1)]


def expected_gain(p: np.ndarray) -> np.ndarray:
    """Ranking score: expected ESCI gain under the judge's label distribution."""
    return np.asarray(p, float) @ np.array([GAIN[l] for l in LABELS])


def confusion(human, llm) -> np.ndarray:
    """4 x 4 counts, rows = human label, columns = LLM label, both in LABELS order."""
    idx = {l: i for i, l in enumerate(LABELS)}
    m = np.zeros((4, 4), dtype=int)
    for h, l in zip(human, llm):
        m[idx[h], idx[l]] += 1
    return m


def cohen_kappa(m: np.ndarray) -> float:
    m = np.asarray(m, float)
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def agreement(human, llm) -> dict:
    """Accuracy, 4-class kappa, Exact-vs-rest kappa, per-class recall, and the majority-label baseline."""
    human, llm = np.asarray(human), np.asarray(llm)
    m = confusion(human, llm)
    binary = confusion(np.where(human == "E", "E", "I"), np.where(llm == "E", "E", "I"))
    rows = m.sum(1)
    return {"n": int(len(human)), "accuracy": float(np.trace(m) / m.sum()), "kappa": cohen_kappa(m),
            "exact_vs_rest_accuracy": float(np.trace(binary) / binary.sum()), "exact_vs_rest_kappa": cohen_kappa(binary),
            "recall_by_human_label": {l: (float(m[i, i] / rows[i]) if rows[i] else None) for i, l in enumerate(LABELS)},
            "majority_label_accuracy": float(rows.max() / rows.sum()),
            "human_label_share": {l: float(rows[i] / rows.sum()) for i, l in enumerate(LABELS)},
            "llm_label_share": {l: float(m.sum(0)[i] / m.sum()) for i, l in enumerate(LABELS)},
            "confusion_rows_human_cols_llm": m.tolist()}


def agreement_ci(df: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """95% intervals for accuracy and kappa, resampling whole queries (pairs of one query are correlated).
    df needs query_id, esci_label (human) and llm_label."""
    groups = [g for _, g in df.groupby("query_id")]
    mats = np.stack([confusion(g["esci_label"], g["llm_label"]) for g in groups])
    rng = np.random.default_rng(seed)
    acc, kap = [], []
    for _ in range(n):
        m = mats[rng.integers(0, len(mats), len(mats))].sum(0)
        acc.append(np.trace(m) / m.sum())
        kap.append(cohen_kappa(m))
    return {"accuracy_ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "kappa_ci": [float(np.nanpercentile(kap, 2.5)), float(np.nanpercentile(kap, 97.5))]}
