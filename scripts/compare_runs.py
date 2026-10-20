"""Before/after table for two run_search.py runs on the same test queries (e.g. off-the-shelf vs fine-tuned).

    python scripts/compare_runs.py --new finetuned           # results.json vs results_finetuned.json

Uses the per-query NDCG@10 files, so intervals and p-values are paired on identical queries. Writes
outputs/search/compare_<new>[_vs_<base>].md and .json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps.metrics import bootstrap_ci, paired_bootstrap_p  # noqa: E402

OUT = Path("outputs/search")
ROWS = [("e5_small", "E5 (dense), full corpus"),
        ("hybrid_convex", "Hybrid, full corpus"),
        ("hybrid_rerank20", "Hybrid + rerank top 20, full corpus"),
        ("rerank_judged_e5_small", "E5 on judged candidates"),
        ("rerank_judged_cross_encoder", "Cross-encoder on judged candidates")]


def paired_delta(a, b, n=2000, seed=0):
    """Mean of b - a, its 95% bootstrap interval and a two-sided p-value, on queries scored in both."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[keep]
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    p_better = paired_bootstrap_p(a, b, n, seed)
    p_worse = paired_bootstrap_p(b, a, n, seed)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], \
        min(1.0, 2 * min(p_better, p_worse)), int(keep.sum())


def load(tag):
    s = f"_{tag}" if tag else ""
    return (json.loads((OUT / f"results{s}.json").read_text(encoding="utf-8")),
            json.loads((OUT / f"per_query{s}.json").read_text(encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="", help="tag of the baseline run ('' = results.json)")
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    (rb, qb), (rn, qn) = load(args.base), load(args.new)
    for key in ("n_test", "corpus", "rerank_n", "n_rerank_view"):
        assert rb["config"][key] == rn["config"][key], f"runs differ in {key}"

    out = {"base": rb["config"]["encoder"] + " / " + rb["config"]["reranker"],
           "new": rn["config"]["encoder"] + " / " + rn["config"]["reranker"]}
    lines = [f"Base: `{out['base']}`  \nNew: `{out['new']}`\n",
             "| Locale | NDCG@10 | Base [95% CI] | New [95% CI] | Δ [95% CI] | p (paired, 2-sided) | Queries |",
             "|---|---|---|---|---|---|---|"]
    for loc in [l for l in ("jp", "us") if l in qb and l in qn]:
        assert qb[loc]["query_id"] == qn[loc]["query_id"], f"{loc}: different test queries"
        out[loc] = {}
        for key, label in ROWS:
            if key not in qb[loc] or key not in qn[loc]:
                continue
            a = np.array(qb[loc][key], dtype=float)
            b = np.array(qn[loc][key], dtype=float)
            ma, la, ha = bootstrap_ci(a)
            mb, lb, hb = bootstrap_ci(b)
            d, dci, p, n = paired_delta(a, b)
            out[loc][key] = {"base": ma, "base_ci": [la, ha], "new": mb, "new_ci": [lb, hb],
                             "delta": d, "delta_ci": dci, "p_two_sided": p, "n_queries": n}
            ptxt = "< 0.001" if p < 0.001 else f"{p:.3f}"
            lines.append(f"| {loc} | {label} | {ma:.3f} [{la:.3f}, {ha:.3f}] | {mb:.3f} [{lb:.3f}, {hb:.3f}] | "
                         f"{d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}] | {ptxt} | {n} |")
        out[loc]["hybrid_alpha"] = {"base": rb[loc]["hybrid_alpha"], "new": rn[loc]["hybrid_alpha"]}
        out[loc]["recall@100_hybrid"] = {"base": rb[loc]["retrieval"]["hybrid_convex"]["recall@100"],
                                         "new": rn[loc]["retrieval"]["hybrid_convex"]["recall@100"]}
        lines.append(f"| {loc} | hybrid α (tuned on dev) | {rb[loc]['hybrid_alpha']} | {rn[loc]['hybrid_alpha']} | | | |")
        lines.append(f"| {loc} | recall@100, hybrid | {out[loc]['recall@100_hybrid']['base']:.3f} | "
                     f"{out[loc]['recall@100_hybrid']['new']:.3f} | | | |")
    md = "\n".join(lines) + "\n"
    print(md)
    name = f"compare_{args.new}" + (f"_vs_{args.base}" if args.base else "")
    (OUT / f"{name}.md").write_text(md, encoding="utf-8")
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
