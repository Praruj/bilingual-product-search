"""Build the artifacts the API serves: per-locale search indexes and the EASE recommender.

Reuses the evaluation corpora (and their cached embeddings) from run_search.py, and the tuned
hyperparameters from outputs/*/results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps import data, recs  # noqa: E402
from bps.bm25 import BM25  # noqa: E402
from bps.dense import DenseIndex  # noqa: E402
from bps.service import TOKENIZER, Recommender, SearchEngine  # noqa: E402
from bps.text import TOKENIZERS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_recs import load as load_transactions  # noqa: E402

INDEX = Path("outputs/index")


def main():
    sres = json.loads(Path("outputs/search/results.json").read_text(encoding="utf-8"))
    cfg = sres["config"]
    enc = None
    for loc in ("jp", "us"):
        _, _, ids, docs = data.eval_setup(loc, cfg["n_test"], cfg["n_dev"], cfg["corpus"])
        cache = Path(f"outputs/search/emb_{loc}_{len(docs)}.npy")
        if cache.exists():
            emb = np.load(cache)
        else:
            enc = enc or DenseIndex()
            emb = enc.fit(docs, progress=True).emb
        bm = BM25(TOKENIZERS[TOKENIZER[loc]]).fit(docs)
        SearchEngine(ids, docs, bm, emb, sres[loc]["hybrid_alpha"]).save(INDEX / loc)
        print(f"{loc}: {len(docs):,} products, alpha {sres[loc]['hybrid_alpha']}")

    rres = json.loads(Path("outputs/recs/results.json").read_text(encoding="utf-8"))
    lam = rres["test"]["new_items"]["ease"]["params"]["lam"]
    df = load_transactions()
    # serve with all history: train on everything (the split's test window is empty)
    split = recs.time_split(df, df["ts"].max() + np.timedelta64(1, "s"), df["ts"].max() + np.timedelta64(2, "s"))
    W = recs.EASE(lam).fit(split.train).W
    desc = df.drop_duplicates("item", keep="last").set_index("item")["Description"]
    names = [str(desc.get(i, i)) for i in split.items]
    Recommender(W, split.train, split.users, split.items, names).save(INDEX / "recs")
    print(f"recs: {len(split.users):,} customers, {len(split.items):,} items, EASE lambda {lam}")


if __name__ == "__main__":
    main()
