"""Multilingual dense retrieval (E5) with exact inner-product search, and a cross-encoder reranker."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .bm25 import top_k

E5 = "intfloat/multilingual-e5-small"
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class DenseIndex:
    """E5 expects 'query: ' and 'passage: ' prefixes. Vectors are L2-normalised, so dot product = cosine."""

    def __init__(self, model=None, name: str = E5, max_len: int = 64):
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(name, device="cpu")
            model.max_seq_length = max_len
        self.model = model
        self.emb: np.ndarray | None = None

    def encode(self, texts: Sequence[str], prefix: str, batch_size: int = 64, progress: bool = False):
        texts = [prefix + t for t in texts]
        # sentence-transformers sorts by length internally, which keeps padding small
        e = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=progress)
        return e.astype(np.float32)

    def fit(self, docs: Sequence[str], **kw) -> "DenseIndex":
        self.emb = self.encode(docs, "passage: ", **kw)
        return self

    def scores(self, q: str) -> np.ndarray:
        return self.emb @ self.encode([q], "query: ")[0]

    def search(self, q: str, k: int = 100):
        return top_k(self.scores(q), k)

    @property
    def nbytes(self) -> int:
        return self.emb.nbytes


class Reranker:
    def __init__(self, model=None, name: str = RERANKER, max_len: int = 128):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(name, device="cpu", max_length=max_len)
        self.model = model

    def rerank(self, q: str, cand: np.ndarray, docs: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Return cand reordered by cross-encoder score."""
        s = np.asarray(self.model.predict([(q, docs[i]) for i in cand], batch_size=batch_size,
                                          show_progress_bar=False))
        return cand[np.argsort(-s, kind="stable")]
