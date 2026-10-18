"""Next-period purchase recommendation on UCI Online Retail II, with a time-based split.

Tune on (train < 2011-06-01, validate 2011-06-01 .. 2011-09-01), then retrain on everything before
2011-09-01 and test on 2011-09-01 .. 2011-12-10 (the last three months, including the pre-Christmas peak).
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps import recs  # noqa: E402
from bps.metrics import bootstrap_ci, paired_bootstrap_p  # noqa: E402

RAW = Path("data/raw/online_retail")
CACHE = Path("data/processed/online_retail.parquet")
OUT = Path("outputs/recs")
VAL = ("2011-06-01", "2011-09-01")
TEST = ("2011-09-01", "2011-12-10")
K = 20


def load():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    zp = RAW / "online_retail_ii.zip"
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        z.extract(name, RAW)
    sheets = pd.read_excel(RAW / name, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = recs.clean_transactions(pd.concat(sheets.values(), ignore_index=True))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def recent_popularity(df, split, cutoff, days=30):
    recent = df[(df["ts"] < pd.Timestamp(cutoff)) & (df["ts"] >= pd.Timestamp(cutoff) - pd.Timedelta(days=days))]
    counts = recent.drop_duplicates(["user", "item"])["item"].value_counts()
    pos = pd.Series(np.arange(len(split.items)), index=split.items)
    pop = np.zeros(len(split.items), dtype=np.float32)
    counts = counts[counts.index.isin(pos.index)]
    pop[pos[counts.index].values] = counts.values
    return pop


GRIDS = {
    "item_knn": [dict(k=k) for k in (50, 200, 1000)],
    "ease": [dict(lam=l) for l in (50.0, 200.0, 500.0, 1000.0)],
    "als": [dict(factors=f, reg=r, alpha=a) for f in (32, 64) for r in (0.1, 1.0) for a in (5.0, 20.0)],
}
MODELS = {"item_knn": recs.ItemKNN, "ease": recs.EASE, "als": recs.ALS, "repeat_ease": recs.RepeatBlend}


def build(name, df, split, cutoff, params=None):
    if name == "popularity":
        return recs.Popularity()
    if name == "popularity_30d":
        return recs.Popularity(recent_popularity(df, split, cutoff))
    if name == "repeat_purchase":
        return recs.RepeatPurchase()
    return MODELS[name](**params)


def main():
    t0 = time.time()
    df = load()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"data": {"transactions": len(df), "customers": int(df["user"].nunique()),
                    "items": int(df["item"].nunique()),
                    "period": [str(df["ts"].min().date()), str(df["ts"].max().date())]},
           "split": {"validation": VAL, "test": TEST}, "k": K, "tuning": {}, "test": {}}

    # ---- tuning on the validation window, for both settings
    val = recs.time_split(df, *VAL)
    best = {}
    for setting, excl in (("all_items", False), ("new_items", True)):
        best[setting] = {}
        for name, grid in list(GRIDS.items()) + ([("repeat_ease", None)] if not excl else []):
            if name == "repeat_ease":  # blend weight, with EASE's lambda as tuned above
                grid = [dict(beta=b, lam=best[setting]["ease"]["lam"]) for b in (0.1, 0.3, 1.0, 3.0, 10.0)]
            scores = []
            for p in grid:
                m = build(name, df, val, VAL[0], p).fit(val.train)
                scores.append(recs.evaluate(m, val, K, exclude_seen=excl)[f"ndcg@{K}"])
            best[setting][name] = grid[int(np.argmax(scores))]
            res["tuning"].setdefault(setting, {})[name] = [{"params": p, f"ndcg@{K}": s} for p, s in zip(grid, scores)]
            print(f"tuned {setting} {name}: {best[setting][name]} ({max(scores):.3f})", flush=True)

    # ---- test
    test = recs.time_split(df, *TEST)
    res["split"].update({"train_users": int(test.train.shape[0]), "train_items": int(test.train.shape[1]),
                         "test_users": len(test.test_users),
                         "test_purchases_on_cold_items": test.cold_item_share,
                         "test_items_per_user_median": float(np.median([len(t) for t in test.test])),
                         "share_of_test_items_already_bought": float(np.mean(
                             [len(t & set(test.train[u].indices.tolist())) / len(t)
                              for u, t in zip(test.test_users, test.test)]))})
    for setting, excl in (("all_items", False), ("new_items", True)):
        out, per_user = {}, {}
        names = ["popularity", "popularity_30d"] + (["repeat_purchase"] if not excl else []) + list(GRIDS) \
            + (["repeat_ease"] if not excl else [])
        for name in names:
            t = time.perf_counter()
            m = build(name, df, test, TEST[0], best[setting].get(name)).fit(test.train)
            fit_s = time.perf_counter() - t
            r = recs.evaluate(m, test, K, exclude_seen=excl)
            per_user[name] = r.pop("_per_user_ndcg")
            m_, lo, hi = bootstrap_ci(per_user[name])
            out[name] = r | {f"ndcg@{K}_ci": [lo, hi], "fit_s": fit_s, "params": best[setting].get(name)}
            print(f"test {setting} {name}: ndcg@{K} {r[f'ndcg@{K}']:.3f} recall@{K} {r[f'recall@{K}']:.3f} "
                  f"coverage {r['coverage']:.2f}", flush=True)
        base = "repeat_purchase" if not excl else "popularity_30d"
        out["paired_p_vs_" + base] = {n: paired_bootstrap_p(per_user[base], v) for n, v in per_user.items() if n != base}
        res["test"][setting] = out

    # ---- example: one customer's history and EASE recommendations
    desc = df.drop_duplicates("item").set_index("item")["Description"]
    m = recs.EASE(**best["new_items"]["ease"]).fit(test.train)
    u = test.test_users[np.argsort([-len(t) for t in test.test])[len(test.test) // 2]]
    sc = m.score(test.train, np.array([u]))[0]
    sc[test.train[u].indices] = -np.inf
    hist = test.train[u].indices[np.argsort(-test.train[u].data)][:5]
    truth = test.test[list(test.test_users).index(u)]
    top = np.argsort(-sc)[:5]
    res["example"] = {"history_top5": [str(desc.get(test.items[i], test.items[i])) for i in hist],
                      "ease_new_items_top5": [{"item": str(desc.get(test.items[i], test.items[i])),
                                               "bought_in_test": bool(i in truth)} for i in top]}
    res["wall_s"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT / 'results.json'} ({res['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
