"""Fast tests on synthetic data: no downloads, no pretrained weights."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps.bm25 import BM25, top_k  # noqa: E402
from bps.hybrid import convex, rrf  # noqa: E402
from bps.metrics import GAIN, bootstrap_ci, ndcg_at_k, paired_bootstrap_p, recall_at_k  # noqa: E402
from bps.text import char_bigrams, has_japanese, mecab_tokens, normalize, whitespace_tokens, word_tokens  # noqa: E402

DOCS = [
    "Apple iPhone 15 case, clear silicone",
    "Samsung Galaxy phone charger USB-C",
    "ワイヤレスイヤホン Bluetooth 5.3 ノイズキャンセリング",
    "ステンレス 水筒 500ml 保冷",
    "ＵＳＢ　ケーブル　タイプＣ　１ｍ",
]


def test_normalize_folds_full_width():
    assert normalize("ＵＳＢ　ケーブル　１ｍ") == "usb ケーブル 1m"
    assert normalize("ｶﾀｶﾅ") == "カタカナ"


def test_tokenizers():
    assert word_tokens("USB-C Charger, 2.4A") == ["usb", "c", "charger", "2.4a"]
    assert whitespace_tokens("ステンレス水筒") == ["ステンレス水筒"]  # one token: no spaces in Japanese
    assert char_bigrams("水筒保冷") == ["水筒", "筒保", "保冷"]
    toks = mecab_tokens("ステンレス水筒を買いたい")
    assert "水筒" in toks and "ステンレス" in toks
    assert has_japanese("iPhone ケース") and not has_japanese("iphone case")


def test_bm25_matches_reference_formula():
    docs = ["a b b c", "a c", "d"]
    bm = BM25(str.split, k1=1.2, b=0.75).fit(docs)
    n, avg = 3, (4 + 2 + 1) / 3
    idf_b = np.log1p((n - 1 + 0.5) / (1 + 0.5))
    expect = idf_b * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 4 / avg))
    assert bm.scores("b")[0] == pytest.approx(expect, rel=1e-5)
    assert bm.scores("b")[1] == 0
    assert bm.scores("zzz").sum() == 0


def test_bm25_japanese_needs_segmentation():
    q = "ステンレス水筒"  # shoppers type Japanese without spaces
    assert BM25(whitespace_tokens).fit(DOCS).search(q, 1)[1][0] == 0  # no match without segmentation
    assert BM25(mecab_tokens).fit(DOCS).search(q, 1)[0][0] == 3
    assert BM25(char_bigrams).fit(DOCS).search(q, 1)[0][0] == 3
    assert BM25(mecab_tokens).fit(DOCS).search("usb ケーブル", 1)[0][0] == 4  # via NFKC


def test_top_k_sorted():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    idx, val = top_k(s, 3)
    assert idx.tolist() == [1, 3, 2] and (np.diff(val) <= 0).all()


def test_ndcg_and_recall():
    lab = {"a": GAIN["E"], "b": GAIN["S"], "c": GAIN["I"]}
    assert ndcg_at_k(["a", "b", "c"], lab) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], lab) == 0.0  # unjudged count as irrelevant
    worse = ndcg_at_k(["b", "a"], lab)
    assert 0 < worse < 1
    assert np.isnan(ndcg_at_k(["a"], {"a": 0.0}))
    assert recall_at_k(["a", "x"], {"a", "b"}, 100) == 0.5


def test_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.random(300)
    m, lo, hi = bootstrap_ci(a)
    assert lo < m < hi
    assert paired_bootstrap_p(a, a + 0.1) < 0.01
    assert paired_bootstrap_p(a + 0.1, a) > 0.99


def test_rrf_rewards_agreement():
    fused = rrf([np.array([1, 2, 3]), np.array([3, 1, 4])], k=4)
    assert fused[0] == 1 and set(fused) == {1, 2, 3, 4}


def test_convex_endpoints():
    sparse, dense = np.array([3.0, 0.0, 1.0]), np.array([0.1, 0.9, 0.2])
    assert convex(sparse, dense, 0.0, 1)[0] == 0
    assert convex(sparse, dense, 1.0, 1)[0] == 1


def test_finetune_data_excludes_eval_queries(tmp_path):
    import pandas as pd

    from bps import data
    from bps.finetune import training_examples

    rows, prods = [], []
    for q in range(40):
        split = "test" if q < 10 else "train"
        for j, lab in enumerate("EESI"):
            pid = f"P{q}_{j}"
            rows.append((q, f"query {q}", pid, "jp", lab, 1, split))
            prods.append((pid, f"title {pid}", "brand", None, "bullet", "jp"))
    pd.DataFrame(rows, columns=["query_id", "query", "product_id", "product_locale", "esci_label",
                                "small_version", "split"]).to_parquet(tmp_path / "shopping_queries_dataset_examples.parquet")
    pd.DataFrame(prods, columns=["product_id", "product_title", "product_brand", "product_color",
                                 "product_bullet_point", "product_locale"]).to_parquet(
        tmp_path / "shopping_queries_dataset_products.parquet")

    df, dropped = training_examples("jp", n_dev=5, raw=tmp_path)
    test, dev, _, _ = data.eval_setup("jp", n_test=10, n_dev=5, corpus=200, raw=tmp_path)
    assert dropped == set(dev.labels) and len(dropped) == 5
    assert not set(df["query_id"]) & (set(dev.labels) | set(test.labels))
    assert df["query_id"].nunique() == 25 and df["doc"].str.startswith("title").all()
    assert df.loc[df["esci_label"] == "S", "gain"].eq(GAIN["S"]).all()
