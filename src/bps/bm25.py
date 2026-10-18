"""BM25 as one sparse matrix product: scoring 100k documents takes milliseconds."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp


class BM25:
    def __init__(self, tokenizer: Callable[[str], list[str]], k1: float = 1.2, b: float = 0.75):
        self.tokenizer, self.k1, self.b = tokenizer, k1, b

    def fit(self, docs: Sequence[str]) -> "BM25":
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            toks = self.tokenizer(d)
            lengths[i] = len(toks)
            tf: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                tf[j] = tf.get(j, 0) + 1
            rows += [i] * len(tf)
            cols += tf.keys()
            vals += tf.values()
        n = len(docs)
        tf_mat = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n, len(vocab)))
        df = np.bincount(tf_mat.indices, minlength=len(vocab))
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg = max(float(lengths.mean()), 1e-9) if n else 1.0
        norm = self.k1 * (1 - self.b + self.b * lengths / avg)
        # term-frequency saturation per stored entry, then idf per column
        tf_mat.data = tf_mat.data * (self.k1 + 1) / (tf_mat.data + np.repeat(norm, np.diff(tf_mat.indptr)))
        self.matrix = (tf_mat @ sp.diags(idf)).tocsc()  # CSC: a query only touches its own columns
        self.vocab = vocab
        return self

    def query_vector(self, q: str) -> np.ndarray:
        ids = [self.vocab[t] for t in self.tokenizer(q) if t in self.vocab]
        return np.array(ids, dtype=np.int64)

    def scores(self, q: str) -> np.ndarray:
        ids = self.query_vector(q)
        if len(ids) == 0:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        # repeated query terms count again, as in the classic formula
        uniq, counts = np.unique(ids, return_counts=True)
        return np.asarray(self.matrix[:, uniq] @ counts.astype(np.float32)).ravel()

    def search(self, q: str, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        return top_k(self.scores(q), k)

    @property
    def nbytes(self) -> int:
        m = self.matrix
        return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return idx, scores[idx]
