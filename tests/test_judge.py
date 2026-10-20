"""LLM-judge data selection and agreement metrics, on synthetic ESCI-shaped data (no downloads, no models)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bps import data  # noqa: E402
from bps.finetune import assert_no_eval_leak, excluded_query_ids  # noqa: E402
from bps.judge import (LABELS, agreement, agreement_ci, cohen_kappa, confusion, distill_query_ids,  # noqa: E402
                       expected_gain, fingerprint, probs_to_label, user_message, view_pairs, view_queries)


def synthetic_examples(n_train=400, n_test=150, per_query=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for qid in range(n_train + n_test):
        split = "train" if qid < n_train else "test"
        labels = rng.choice(LABELS, size=per_query, p=[0.4, 0.3, 0.1, 0.2])
        if qid % 7 == 0:
            labels[:] = "I"   # some queries have no Exact product
        for j, lab in enumerate(labels):
            rows.append({"query_id": qid, "query": f"query {qid}", "product_id": f"P{qid}_{j}",
                         "product_locale": "jp", "esci_label": lab, "split": split})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ex():
    return synthetic_examples()


def test_view_queries_follow_run_search_order(ex):
    test = data.make_split(ex, data.sample_queries(ex, "test", 100, seed=0))
    q = view_queries(ex, n_test=100, n_view=30)
    assert q["query_id"].tolist() == test.queries["query_id"].iloc[:30].tolist()


def test_view_pairs_cover_every_judged_product_in_order(ex):
    v = view_pairs(ex, n_test=100, n_view=30)
    q = view_queries(ex, n_test=100, n_view=30)
    assert v.drop_duplicates("query_id")["query_id"].tolist() == q["query_id"].tolist()
    assert (v.groupby("query_id").size() == ex[ex["query_id"].isin(q["query_id"])].groupby("query_id").size()).all()
    assert set(v["split"]) == {"test"} and v["qorder"].is_monotonic_increasing


def test_distill_sample_is_seeded_whole_train_queries_without_eval_queries(ex):
    dev = excluded_query_ids(ex, n_dev=50)
    train = ex[(ex["split"] == "train") & ~ex["query_id"].isin(dev)]
    a, b = distill_query_ids(train, 40), distill_query_ids(train, 40)
    assert a == b and len(a) == 40 and len(set(a)) == 40
    assert distill_query_ids(train, 40, seed=9) != a
    assert not set(a) & dev
    assert set(a) <= set(train.loc[train["esci_label"] == "E", "query_id"])
    assert_no_eval_leak(train[train["query_id"].isin(a)], ex, dev, n_test=100)


def test_leak_check_catches_eval_queries_and_test_text(ex):
    dev = excluded_query_ids(ex, n_dev=50)
    with pytest.raises(AssertionError, match="eval query"):
        assert_no_eval_leak(ex[ex["query_id"] == sorted(dev)[0]], ex, dev, n_test=100)
    sneaky = ex[(ex["split"] == "train") & ~ex["query_id"].isin(dev)].head(3).assign(query="QUERY 450 ")
    with pytest.raises(AssertionError, match="share query text"):
        assert_no_eval_leak(sneaky, ex, dev, n_test=100)


def test_make_judge_data_query_rank_is_a_seeded_permutation():
    from make_judge_data import query_rank
    r = query_rank([5, 3, 9, 3, 1], seed=4)
    assert sorted(r) == [1, 3, 5, 9] and sorted(r.values()) == [0, 1, 2, 3]
    assert r == query_rank([1, 3, 5, 9], seed=4)


def test_fingerprint_ignores_row_order_but_not_content():
    df = pd.DataFrame({"set": ["view", "view"], "locale": ["jp", "jp"], "query_id": [1, 2], "product_id": ["a", "b"],
                       "esci_label": ["E", "I"], "query": ["x", "y"], "doc": ["d1", "d2"]})
    assert fingerprint(df) == fingerprint(df.iloc[::-1])
    assert fingerprint(df) != fingerprint(df.assign(esci_label=["E", "S"]))


def test_prompt_cuts_long_products():
    m = user_message("水筒", "x" * 5000)
    assert m.startswith("Query: 水筒\nProduct: ") and len(m) < 450


def test_probs_to_label_and_expected_gain():
    p = np.array([[0.7, 0.2, 0.05, 0.05], [0.0, 0.0, 0.0, 1.0]])
    assert probs_to_label(p).tolist() == ["E", "I"]
    assert np.allclose(expected_gain(p), [0.7 + 0.02 + 0.0005, 0.0])


def test_kappa_matches_sklearn_and_edge_cases():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(1)
    h = rng.choice(LABELS, 500)
    l = np.where(rng.random(500) < 0.6, h, rng.choice(LABELS, 500))
    assert cohen_kappa(confusion(h, l)) == pytest.approx(cohen_kappa_score(h, l))
    assert cohen_kappa(confusion(h, h)) == pytest.approx(1.0)
    m = confusion(["E", "S", "C", "I"], ["E", "E", "E", "E"])
    assert m.sum() == 4 and m[:, 0].sum() == 4 and cohen_kappa(m) == pytest.approx(0.0)


def test_agreement_report_and_query_bootstrap():
    df = pd.DataFrame({"query_id": np.repeat(np.arange(50), 4),
                       "esci_label": np.tile(["E", "S", "I", "E"], 50),
                       "llm_label": np.tile(["E", "E", "I", "E"], 50)})
    a = agreement(df["esci_label"], df["llm_label"])
    assert a["accuracy"] == pytest.approx(0.75) and a["majority_label_accuracy"] == pytest.approx(0.5)
    assert a["recall_by_human_label"]["S"] == 0.0 and a["recall_by_human_label"]["C"] is None
    assert a["exact_vs_rest_accuracy"] == pytest.approx(0.75)
    ci = agreement_ci(df, n=200)
    assert ci["accuracy_ci"] == pytest.approx([0.75, 0.75])   # every query is identical


def test_common_prefix_of_token_lists():
    from llm_judge import common_prefix
    assert common_prefix([[1, 2, 3, 4], [1, 2, 5], [1, 2, 3]]) == [1, 2]
    assert common_prefix([[7, 8], [9]]) == []
    assert common_prefix([[1, 2]]) == [1, 2]
