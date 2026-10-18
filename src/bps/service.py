"""FastAPI service: bilingual product search and purchase recommendations.

    uvicorn bps.service:app --app-dir src

Artifacts are produced by scripts/build_index.py under outputs/index/.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException, Query

from .bm25 import BM25, top_k
from .hybrid import convex
from .text import TOKENIZERS, has_japanese

INDEX = Path(os.environ.get("BPS_INDEX", "outputs/index"))
TOKENIZER = {"jp": "mecab", "us": "word"}


class SearchEngine:
    def __init__(self, ids, docs, bm25: BM25, emb: np.ndarray, alpha: float, encoder=None, reranker=None):
        self.ids, self.docs, self.bm25, self.emb, self.alpha = np.asarray(ids), docs, bm25, emb, alpha
        self.encoder, self.reranker = encoder, reranker

    def search(self, q: str, k: int = 10, mode: str = "hybrid", rerank_n: int = 20) -> list[int]:
        if mode == "bm25":
            return top_k(self.bm25.scores(q), k)[0].tolist()
        d = self.emb @ self.encoder.encode([q], "query: ")[0].astype(self.emb.dtype)
        if mode == "dense":
            return top_k(d.astype(np.float32), k)[0].tolist()
        cand = convex(self.bm25.scores(q), d.astype(np.float32), self.alpha, max(k, rerank_n))
        if mode == "rerank":
            cand = np.concatenate([self.reranker.rerank(q, cand[:rerank_n], self.docs), cand[rerank_n:]])
        return cand[:k].tolist()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "emb_fp16.npy", self.emb.astype(np.float16))
        sp.save_npz(path / "bm25.npz", self.bm25.matrix.tocsc())
        (path / "vocab.pkl").write_bytes(pickle.dumps(self.bm25.vocab))
        (path / "docs.json").write_text(json.dumps({"ids": self.ids.tolist(), "docs": self.docs,
                                                    "alpha": self.alpha}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, locale: str, encoder=None, reranker=None) -> "SearchEngine":
        meta = json.loads((path / "docs.json").read_text(encoding="utf-8"))
        bm25 = BM25(TOKENIZERS[TOKENIZER[locale]])
        bm25.matrix, bm25.vocab = sp.load_npz(path / "bm25.npz").tocsc(), pickle.loads((path / "vocab.pkl").read_bytes())
        emb = np.load(path / "emb_fp16.npy").astype(np.float32)
        return cls(meta["ids"], meta["docs"], bm25, emb, meta["alpha"], encoder, reranker)


class Recommender:
    """EASE item-item weights plus each customer's purchase history."""

    def __init__(self, W: np.ndarray, history: sp.csr_matrix, users, items, names):
        self.W, self.history, self.items, self.names = W, history, np.asarray(items), names
        self.row = {int(u): i for i, u in enumerate(users)}
        self.pop = np.asarray((history > 0).sum(axis=0)).ravel()

    def recommend(self, customer_id: int, k: int = 10, new_only: bool = True) -> tuple[list[int], bool]:
        r = self.row.get(customer_id)
        if r is None:  # unknown customer: fall back to best sellers
            return top_k(self.pop.astype(np.float32), k)[0].tolist(), False
        x = (self.history[r] > 0).astype(np.float32)
        s = np.asarray(x @ self.W).ravel()
        if new_only:
            s[self.history[r].indices] = -np.inf
        return top_k(s, k)[0].tolist(), True

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "ease_W_fp16.npy", self.W.astype(np.float16))
        sp.save_npz(path / "history.npz", self.history)
        users = sorted(self.row, key=self.row.get)
        (path / "meta.json").write_text(json.dumps({"users": users, "items": self.items.tolist(),
                                                    "names": self.names}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Recommender":
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        W = np.load(path / "ease_W_fp16.npy").astype(np.float32)
        return cls(W, sp.load_npz(path / "history.npz").tocsr(), meta["users"], meta["items"], meta["names"])


def create_app(engines: dict[str, SearchEngine] | None = None, recommender: Recommender | None = None) -> FastAPI:
    state: dict = {"engines": engines, "rec": recommender}

    @asynccontextmanager
    async def lifespan(_):
        _load()
        yield

    def _load():
        if state["engines"] is None:
            from .dense import DenseIndex, Reranker

            enc, rr = DenseIndex(), Reranker()
            state["engines"] = {loc: SearchEngine.load(INDEX / loc, loc, enc, rr)
                                for loc in ("jp", "us") if (INDEX / loc).exists()}
        if state["rec"] is None and (INDEX / "recs").exists():
            state["rec"] = Recommender.load(INDEX / "recs")

    app = FastAPI(title="Bilingual product search", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "locales": sorted(state["engines"] or {}),
                "recommender": state["rec"] is not None}

    @app.get("/search")
    def search(q: str = Query(..., min_length=1, max_length=200),
               locale: Literal["auto", "jp", "us"] = "auto",
               mode: Literal["bm25", "dense", "hybrid", "rerank"] = "hybrid",
               k: int = Query(10, ge=1, le=100)):
        if locale == "auto":  # a Japanese query searches the Japanese catalogue
            locale = "jp" if has_japanese(q) else "us"
        eng = (state["engines"] or {}).get(locale)
        if eng is None:
            raise HTTPException(404, f"no index for locale {locale}")
        t = time.perf_counter()
        hits = eng.search(q, k, mode)
        ms = (time.perf_counter() - t) * 1e3
        return {"query": q, "locale": locale, "mode": mode, "took_ms": round(ms, 2),
                "results": [{"product_id": str(eng.ids[i]), "text": eng.docs[i][:200]} for i in hits]}

    @app.get("/recommend/{customer_id}")
    def recommend(customer_id: int, k: int = Query(10, ge=1, le=100), new_only: bool = True):
        rec = state["rec"]
        if rec is None:
            raise HTTPException(404, "recommender not built")
        t = time.perf_counter()
        hits, known = rec.recommend(customer_id, k, new_only)
        return {"customer_id": customer_id, "personalised": known,
                "took_ms": round((time.perf_counter() - t) * 1e3, 2),
                "items": [{"stock_code": str(rec.items[i]), "description": rec.names[i]} for i in hits]}

    return app


app = create_app()
