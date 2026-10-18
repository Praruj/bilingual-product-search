"""Implicit-feedback recommenders on purchase data, with a time-based split.

All models take a user x item count matrix (scipy CSR) and return a dense score matrix for the requested
users. Small enough to be exact: a few thousand items fit an item x item matrix in memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------- data
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Online Retail II: drop cancellations, returns, guest checkouts and non-product codes (postage, fees)."""
    df = df.rename(columns={"Customer ID": "user", "StockCode": "item", "InvoiceDate": "ts"})
    df = df.dropna(subset=["user"])
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["item"].astype(str).str.match(r"^\d{5}")]
    df = df.assign(user=df["user"].astype(int), item=df["item"].astype(str).str.upper(),
                   ts=pd.to_datetime(df["ts"]))
    return df[["Invoice", "user", "item", "ts", "Description"]].reset_index(drop=True)


@dataclass
class TimeSplit:
    train: sp.csr_matrix          # users x items, value = number of invoices containing the item
    test: list[set[int]]          # per test user: items bought in the test window
    test_users: np.ndarray        # row indices into train
    users: np.ndarray             # row index -> customer id
    items: np.ndarray             # column index -> stock code
    cold_item_share: float        # share of test purchases on items never seen in training


def time_split(df: pd.DataFrame, cutoff, end) -> TimeSplit:
    """Train on everything before cutoff; test on [cutoff, end) for users who also bought before cutoff."""
    cutoff, end = pd.Timestamp(cutoff), pd.Timestamp(end)
    tr = df[df["ts"] < cutoff].drop_duplicates(["Invoice", "item"])
    te = df[(df["ts"] >= cutoff) & (df["ts"] < end)].drop_duplicates(["user", "item"])
    users = np.sort(tr["user"].unique())
    items = np.sort(tr["item"].unique())
    u_ix = pd.Series(np.arange(len(users)), index=users)
    i_ix = pd.Series(np.arange(len(items)), index=items)
    counts = tr.groupby(["user", "item"]).size()
    train = sp.csr_matrix((counts.values.astype(np.float32),
                           (u_ix[counts.index.get_level_values(0)].values,
                            i_ix[counts.index.get_level_values(1)].values)), shape=(len(users), len(items)))
    te = te[te["user"].isin(u_ix.index)]
    cold = float((~te["item"].isin(i_ix.index)).mean()) if len(te) else 0.0
    te = te[te["item"].isin(i_ix.index)]
    grouped = te.groupby("user")["item"].apply(lambda s: set(i_ix[s].tolist()))
    return TimeSplit(train, grouped.tolist(), u_ix[grouped.index].values, users, items, cold)


# ---------------------------------------------------------------- models
class Popularity:
    def __init__(self, recent_counts: np.ndarray | None = None):
        self.recent = recent_counts

    def fit(self, X):
        self.pop = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32) if self.recent is None else self.recent
        return self

    def score(self, X, users):
        return np.tile(self.pop, (len(users), 1))


class RepeatPurchase:
    """Recommend what the user already buys, most frequent first, then fall back to popularity."""

    def fit(self, X):
        self.pop = Popularity().fit(X).pop
        return self

    def score(self, X, users):
        own = X[users].toarray()
        return own * 1e6 + self.pop / (self.pop.max() + 1)


class RepeatBlend:
    """Own purchase history and EASE together: beta * log(1 + own count) + EASE score."""

    def __init__(self, beta: float = 1.0, lam: float = 500.0):
        self.beta, self.lam = beta, lam

    def fit(self, X):
        self.ease = EASE(self.lam).fit(X)
        return self

    def score(self, X, users):
        own = X[users].toarray()
        return self.beta * np.log1p(own) + self.ease.score(X, users)


class ItemKNN:
    """Item-item cosine similarity on binary purchases (the 'customers also bought' classic)."""

    def __init__(self, k: int = 100):
        self.k = k

    def fit(self, X):
        B = (X > 0).astype(np.float32)
        norms = np.sqrt(np.asarray(B.sum(axis=0)).ravel()) + 1e-9
        S = (B.T @ B).toarray() / np.outer(norms, norms)
        np.fill_diagonal(S, 0)
        if self.k < S.shape[0]:  # keep each item's k nearest neighbours
            thresh = -np.partition(-S, self.k - 1, axis=0)[self.k - 1]
            S[S < thresh] = 0
        self.S = S.astype(np.float32)
        return self

    def score(self, X, users):
        return ((X[users] > 0).astype(np.float32) @ self.S)


class EASE:
    """Embarrassingly Shallow Autoencoder (Steck, 2019): a closed-form item-item linear model."""

    def __init__(self, lam: float = 200.0):
        self.lam = lam

    def fit(self, X):
        B = (X > 0).astype(np.float64)
        G = (B.T @ B).toarray()
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        W = -P / np.diag(P)
        np.fill_diagonal(W, 0)
        self.W = W.astype(np.float32)
        return self

    def score(self, X, users):
        return np.asarray((X[users] > 0).astype(np.float32) @ self.W)


class ALS:
    """Implicit ALS (Hu, Koren & Volinsky, 2008): confidence 1 + alpha * log(1 + count), exact solves."""

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 20.0, iters: int = 15, seed: int = 0):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, C: sp.csr_matrix, Y: np.ndarray) -> np.ndarray:
        YtY = Y.T @ Y + self.reg * np.eye(self.f)
        out = np.zeros((C.shape[0], self.f))
        for u in range(C.shape[0]):
            lo, hi = C.indptr[u], C.indptr[u + 1]
            if lo == hi:
                continue
            idx, c = C.indices[lo:hi], C.data[lo:hi]
            Yi = Y[idx]
            A = YtY + (Yi.T * (c - 1)) @ Yi
            out[u] = np.linalg.solve(A, Yi.T @ c)
        return out

    def fit(self, X):
        C = X.astype(np.float64).tocsr(copy=True)
        C.data = 1 + self.alpha * np.log1p(C.data)
        Ct = C.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.I = rng.normal(0, 0.01, (X.shape[1], self.f))
        for _ in range(self.iters):
            self.U = self._solve(C, self.I)
            self.I = self._solve(Ct, self.U)
        return self

    def score(self, X, users):
        return (self.U[users] @ self.I.T).astype(np.float32)


# ---------------------------------------------------------------- evaluation
def evaluate(model, split: TimeSplit, k: int = 20, exclude_seen: bool = False, batch: int = 1024) -> dict:
    """Recall@k and NDCG@k (binary). exclude_seen: the 'new items only' setting, where past purchases are
    removed from both the recommendations and the ground truth."""
    X = split.train
    recalls, ndcgs, shown = [], [], set()
    disc = 1 / np.log2(np.arange(2, k + 2))
    for s in range(0, len(split.test_users), batch):
        users = split.test_users[s : s + batch]
        sc = model.score(X, users)
        if exclude_seen:
            seen = X[users]
            sc[seen.nonzero()] = -np.inf
        top = np.argpartition(-sc, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(sc, top, 1), axis=1), 1)
        for row, (u, truth) in enumerate(zip(users, split.test[s : s + batch])):
            if exclude_seen:
                truth = truth - set(X[u].indices.tolist())
            if not truth:
                continue
            hits = np.isin(top[row], list(truth))
            recalls.append(hits.sum() / min(len(truth), k))
            ndcgs.append((hits * disc).sum() / disc[: min(len(truth), k)].sum())
            shown.update(top[row].tolist())
    return {f"recall@{k}": float(np.mean(recalls)), f"ndcg@{k}": float(np.mean(ndcgs)),
            "coverage": len(shown) / X.shape[1], "n_users": len(recalls),
            "_per_user_ndcg": np.array(ndcgs)}
