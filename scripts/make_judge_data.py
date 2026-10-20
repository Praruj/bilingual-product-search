"""Write the (query, product) pairs for the LLM relevance judge (notebooks/colab_llm_judge.ipynb).

Output: data/judge/judge_pairs.parquet, one row per judged pair, column `set`:
  view     all judged products of the first 200 test queries per locale (run_search.py's rerank_judged view),
           for label agreement with humans and for reranking with LLM scores
  distill  all judged products of up to --n-distill training queries per locale (train split minus the eval dev
           queries, never a test query or test query text), for training a cross-encoder on LLM labels

`qrank` is a seeded random order of the queries in each set and locale. The notebook labels queries in qrank
order, alternating locales, so whatever is finished when a Colab session ends is still a random sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps import data  # noqa: E402
from bps.finetune import assert_no_eval_leak, training_examples  # noqa: E402
from bps.judge import distill_query_ids, fingerprint, view_pairs  # noqa: E402
from bps.metrics import GAIN  # noqa: E402

COLS = ["set", "locale", "query_id", "query", "product_id", "esci_label", "gain", "doc", "qorder", "qrank"]


def query_rank(query_ids, seed: int) -> dict[int, int]:
    ids = np.sort(np.unique(query_ids))
    return {int(q): i for i, q in enumerate(np.random.default_rng(seed).permutation(ids))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locales", nargs="+", default=["jp", "us"])
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-view", type=int, default=200)
    ap.add_argument("--n-distill", type=int, default=2000, help="training queries per locale (a pool; the "
                    "notebook labels as many as its time budget allows, in qrank order)")
    ap.add_argument("--raw", default=str(data.RAW))
    ap.add_argument("--out", default="data/judge/judge_pairs.parquet")
    args = ap.parse_args()
    raw = Path(args.raw)

    frames, stats = [], {}
    for loc in args.locales:
        ex = data.load_examples(loc, raw)
        view = view_pairs(ex, args.n_test, args.n_view)
        prods = data.load_products(loc, set(view["product_id"]), raw)
        text = {r.product_id: data.product_text(r) for r in prods.itertuples()}
        view = view.assign(set="view", locale=loc, doc=view["product_id"].map(text),
                           gain=view["esci_label"].map(GAIN), qrank=view["qorder"])

        train, dropped = training_examples(loc, args.n_dev, raw)
        keep = distill_query_ids(train, args.n_distill)
        dist = train[train["query_id"].isin(keep)].assign(set="distill", qorder=-1)
        dist["qrank"] = dist["query_id"].map(query_rank(keep, seed=4))
        assert_no_eval_leak(dist, ex, dropped, args.n_test)
        assert not set(dist["query_id"]) & set(view["query_id"])

        for name, df in (("view", view), ("distill", dist)):
            stats[f"{loc}_{name}"] = {"queries": int(df["query_id"].nunique()), "pairs": len(df),
                                      "labels": df["esci_label"].value_counts().to_dict()}
            print(loc, name, stats[f"{loc}_{name}"], flush=True)
        frames += [view[COLS], dist[COLS]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    df["query_id"] = df["query_id"].astype(int)
    df.to_parquet(out, index=False)
    stats["fingerprint"] = fingerprint(df)
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(df):,} pairs, {out.stat().st_size / 1e6:.1f} MB), fingerprint {stats['fingerprint']}")


if __name__ == "__main__":
    main()
