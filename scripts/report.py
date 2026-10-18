"""Figures from outputs/*/results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
BLUE, ORANGE, AQUA, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#e4e3df"
LOC = {"jp": ("Japanese (amazon.co.jp)", BLUE), "us": ("English (amazon.com)", ORANGE)}
NAMES = {"bm25_whitespace": "BM25\nwhitespace", "bm25_char_bigram": "BM25\nchar bigrams",
         "bm25_mecab": "BM25\nMeCab", "bm25_word": "BM25\nwords", "e5_small": "E5-small\n(dense)",
         "hybrid_rrf": "Hybrid\nRRF", "hybrid_convex": "Hybrid\ntuned weight"}


def grouped_bars(ax, groups, series, metric, ylabel):
    """groups: x labels; series: {locale: {group: result dict or None}}."""
    x = np.arange(len(groups))
    w = 0.38
    for j, (loc, vals) in enumerate(series.items()):
        label, color = LOC[loc]
        for i, g in enumerate(groups):
            r = vals.get(g)
            if r is None:
                continue
            m, (lo, hi) = r[metric], r[metric + "_ci"]
            xi = x[i] + (j - 0.5) * w
            ax.bar(xi, m, w - 0.04, color=color, label=label if i == next(
                k for k, gg in enumerate(groups) if vals.get(gg)) else None)
            ax.errorbar(xi, m, [[m - lo], [hi - m]], color="#52514e", lw=1, capsize=2)
            ax.text(xi, hi + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, color="#52514e")
    ax.set_xticks(x, [NAMES.get(g, g) for g in groups], fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def search_fig(res):
    rr = next(k for k in res["jp"]["retrieval"] if "rerank" in k)
    NAMES[rr] = "Hybrid +\ncross-encoder"
    NAMES["bm25_tok"] = "BM25\nMeCab / words"
    groups = ["bm25_whitespace", "bm25_tok", "e5_small", "hybrid_rrf", "hybrid_convex", rr]
    series = {loc: dict(res[loc]["retrieval"]) for loc in ("jp", "us")}
    series["jp"]["bm25_tok"], series["us"]["bm25_tok"] = series["jp"]["bm25_mecab"], series["us"]["bm25_word"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.4))
    grouped_bars(axes[0], groups, series, "ndcg@10", "NDCG@10")
    axes[0].set_title("Ranking quality: NDCG@10 over a 50k-product corpus (95% bootstrap CI)", loc="left")
    axes[0].legend(frameon=False, loc="upper left")
    grouped_bars(axes[1], [g for g in groups if g != rr], series, "recall@100", "Recall@100 (Exact)")
    axes[1].set_title("Candidate generation: share of Exact products in the top 100", loc="left")
    for ax in axes:  # headroom for the legend and value labels
        ax.set_ylim(0, ax.get_ylim()[1] + 0.1)
    fig.tight_layout()
    fig.savefig(FIG / "search_results.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for loc in ("jp", "us"):
        curve = res[loc]["hybrid_alpha_dev_curve"]
        a = [float(k) for k in curve]
        label, color = LOC[loc]
        ax.plot(a, list(curve.values()), color=color, lw=2, marker="o", ms=4, label=label)
        best = res[loc]["hybrid_alpha"]
        ax.plot([best], [curve[str(best)] if str(best) in curve else curve[best]], "o", ms=9,
                mfc="none", mec=color, mew=2)
    ax.set_xlabel("weight on dense score (0 = BM25 only, 1 = E5 only)")
    ax.set_ylabel("dev NDCG@10")
    ax.set_title("Hybrid weight, tuned on train-split queries", loc="left", fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "hybrid_weight.png", dpi=150)
    plt.close(fig)


def recs_fig(res):
    k = res["k"]
    order = ["popularity", "popularity_30d", "item_knn", "als", "ease", "repeat_purchase", "repeat_ease"]
    names = {"popularity": "Popularity", "popularity_30d": "Popularity\n30 days",
             "repeat_purchase": "Re-buy\nhistory", "item_knn": "Item kNN", "als": "ALS", "ease": "EASE",
             "repeat_ease": "Re-buy\n+ EASE"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [7, 5]})
    for ax, (setting, title, color) in zip(axes, [("all_items", "All purchases (re-buys count)", BLUE),
                                                   ("new_items", "New items only (history excluded)", AQUA)]):
        r = res["test"][setting]
        keys = [o for o in order if o in r]
        x = np.arange(len(keys))
        m = np.array([r[o][f"ndcg@{k}"] for o in keys])
        ci = np.array([r[o][f"ndcg@{k}_ci"] for o in keys])
        ax.bar(x, m, 0.6, color=color)
        ax.errorbar(x, m, [m - ci[:, 0], ci[:, 1] - m], fmt="none", color="#52514e", lw=1, capsize=2)
        for xi, (mi, hi) in enumerate(zip(m, ci[:, 1])):
            ax.text(xi, hi + 0.004, f"{mi:.3f}", ha="center", va="bottom", fontsize=7.5, color="#52514e")
        ax.set_xticks(x, [names[o] for o in keys], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(f"NDCG@{k}")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"Next-quarter purchases, {res['split']['test_users']:,} customers (Sep-Dec 2011, 95% CI)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "recs_results.png", dpi=150)
    plt.close(fig)


def latency_fig(search, bench):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    modes = {"bm25": "bm25_", "dense": "e5_small", "hybrid": "hybrid_convex", "rerank": "hybrid_rerank"}
    for loc in ("jp", "us"):
        label, color = LOC[loc]
        xs, ys = [], []
        for mode, key in modes.items():
            r = search[loc]["retrieval"]
            key = {"bm25_": "bm25_mecab" if loc == "jp" else "bm25_word"}.get(key, key)
            key = next(k for k in r if k.startswith(key))
            xs.append(bench["latency_ms"][f"{loc}_{mode}"]["http_p50"])
            ys.append(r[key]["ndcg@10"])
            ax.annotate(mode, (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -3), fontsize=8,
                        color="#52514e")
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("median latency per query over HTTP, ms (log scale, laptop CPU)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Quality vs latency of the served modes", loc="left", fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "latency_quality.png", dpi=150)
    plt.close(fig)


def main():
    search = json.loads((OUT / "search/results.json").read_text(encoding="utf-8"))
    search_fig(search)
    if (OUT / "recs/results.json").exists():
        recs_fig(json.loads((OUT / "recs/results.json").read_text(encoding="utf-8")))
    if (OUT / "service/benchmark.json").exists():
        latency_fig(search, json.loads((OUT / "service/benchmark.json").read_text(encoding="utf-8")))
    print("figures in", FIG)


if __name__ == "__main__":
    main()
