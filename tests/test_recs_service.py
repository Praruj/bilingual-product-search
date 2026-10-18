"""Recommenders and the API on synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps import recs  # noqa: E402
from bps.bm25 import BM25  # noqa: E402
from bps.service import Recommender, SearchEngine, create_app  # noqa: E402
from bps.text import mecab_tokens, word_tokens  # noqa: E402


def synthetic_purchases(n_users=300, seed=0):
    """Two customer segments with disjoint baskets, and a cancellation/postage line to be cleaned."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        seg = u % 2
        for t in range(6):
            day = pd.Timestamp("2011-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
            for item in rng.choice(np.arange(10) + 10 * seg + 10000, size=3, replace=False):
                rows.append({"Invoice": f"{u}-{t}", "StockCode": str(item), "Description": f"item {item}",
                             "Quantity": 1, "InvoiceDate": day, "Price": 1.0, "Customer ID": float(u)})
    rows.append({"Invoice": "C1", "StockCode": "10001", "Description": "x", "Quantity": -1,
                 "InvoiceDate": pd.Timestamp("2011-02-01"), "Price": 1.0, "Customer ID": 1.0})
    rows.append({"Invoice": "9", "StockCode": "POST", "Description": "postage", "Quantity": 1,
                 "InvoiceDate": pd.Timestamp("2011-02-01"), "Price": 5.0, "Customer ID": 1.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def split():
    df = recs.clean_transactions(synthetic_purchases())
    assert not df["item"].eq("POST").any() and not df["Invoice"].str.startswith("C").any()
    return recs.time_split(df, "2011-09-01", "2011-12-31")


def test_time_split_no_leakage(split):
    assert split.train.shape[0] == len(split.users)
    assert len(split.test) == len(split.test_users) > 0
    assert all(t for t in split.test)


@pytest.mark.parametrize("model", [recs.ItemKNN(k=5), recs.EASE(lam=10.0), recs.ALS(factors=4, iters=5)])
def test_models_learn_segments(split, model):
    """Personalised models beat popularity when baskets depend on the segment."""
    m = model.fit(split.train)
    pop = recs.evaluate(recs.Popularity().fit(split.train), split, k=5)
    r = recs.evaluate(m, split, k=5)
    assert r["ndcg@5"] > pop["ndcg@5"] + 0.1


def test_exclude_seen_removes_history(split):
    m = recs.RepeatPurchase().fit(split.train)
    sc = m.score(split.train, split.test_users[:5])
    assert (sc[split.train[split.test_users[:5]].nonzero()] >= 1e6).all()
    r = recs.evaluate(recs.EASE(10.0).fit(split.train), split, k=5, exclude_seen=True)
    assert 0 <= r["recall@5"] <= 1


class FakeEncoder:
    """Bag-of-tokens hashed into 64 dims: stands in for E5 in tests."""

    def encode(self, texts, prefix):
        out = np.zeros((len(texts), 64), np.float32)
        for i, t in enumerate(texts):
            for tok in mecab_tokens(t.replace(prefix, "")):
                out[i, hash(tok) % 64] += 1
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


class FakeReranker:
    def rerank(self, q, cand, docs):
        return cand[::-1]


@pytest.fixture(scope="module")
def client(split, tmp_path_factory):
    enc = FakeEncoder()
    jp_docs = ["ステンレス 水筒 500ml", "ワイヤレス イヤホン", "USB ケーブル"]
    us_docs = ["stainless water bottle", "wireless earbuds", "usb cable"]
    engines = {}
    for loc, docs, tok in (("jp", jp_docs, mecab_tokens), ("us", us_docs, word_tokens)):
        e = SearchEngine([f"{loc}{i}" for i in range(3)], docs, BM25(tok).fit(docs),
                         enc.encode(docs, "passage: "), 0.5, enc, FakeReranker())
        path = tmp_path_factory.mktemp(loc)
        e.save(path)  # round-trip through disk
        engines[loc] = SearchEngine.load(path, loc, enc, FakeReranker())
    W = recs.EASE(10.0).fit(split.train).W
    rec = Recommender(W, split.train, split.users, split.items, [f"item {i}" for i in split.items])
    path = tmp_path_factory.mktemp("recs")
    rec.save(path)
    with TestClient(create_app(engines, Recommender.load(path))) as c:
        yield c


def test_search_api(client):
    assert client.get("/health").json()["locales"] == ["jp", "us"]
    r = client.get("/search", params={"q": "水筒"}).json()
    assert r["locale"] == "jp" and r["results"][0]["product_id"] == "jp0"
    for mode in ("bm25", "dense", "hybrid", "rerank"):
        r = client.get("/search", params={"q": "usb cable", "mode": mode, "k": 2}).json()
        assert r["locale"] == "us" and len(r["results"]) == 2
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_recommend_api(client, split):
    known = int(split.users[0])
    r = client.get(f"/recommend/{known}", params={"k": 3}).json()
    assert r["personalised"] and len(r["items"]) == 3
    seen = {str(split.items[i]) for i in split.train[0].indices}
    assert not seen & {it["stock_code"] for it in r["items"]}
    assert client.get("/recommend/999999").json()["personalised"] is False
