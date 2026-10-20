"""Score the LLM judge (labels from notebooks/colab_llm_judge.ipynb) against the human ESCI labels.

    python scripts/judge_report.py      # data/judge/judge_pairs.parquet + data/judge/labels/ -> outputs/llm_judge/

1. Agreement with humans per locale, on the held-out `view` pairs (test split) and on the `distill` pairs
   (train split, more pairs): accuracy, Cohen's kappa (4 labels, and Exact vs the rest), confusion matrix,
   with 95% intervals that resample whole queries.
2. NDCG@10 when each view query's judged products are ranked by the LLM's expected gain, paired with E5 and the
   cross-encoders on the same queries (outputs/search/per_query*.json, the rerank_judged rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bps.judge import agreement, agreement_ci, expected_gain, probs_to_label  # noqa: E402
from bps.metrics import bootstrap_ci, ndcg_at_k  # noqa: E402
from compare_runs import paired_delta  # noqa: E402
from llm_judge import PROB_COLS, load_labels  # noqa: E402

SEARCH = Path("outputs/search")
# rows of per_query*.json to compare with, on the same first view queries
REFERENCE = {"E5 (off the shelf)": ("", "rerank_judged_e5_small"),
             "E5 (fine-tuned)": ("finetuned", "rerank_judged_e5_small"),
             "cross-encoder (off the shelf)": ("", "rerank_judged_cross_encoder"),
             "cross-encoder (fine-tuned on 600k human labels)": ("finetuned", "rerank_judged_cross_encoder"),
             "cross-encoder (LLM labels)": ("llm_distilled", "rerank_judged_cross_encoder"),
             "cross-encoder (human labels, same pairs)": ("human_subset", "rerank_judged_cross_encoder")}


def labelled(pairs: pd.DataFrame, label_dir: Path) -> pd.DataFrame:
    lab = pd.concat([load_labels(label_dir, s) for s in ("view", "distill")], ignore_index=True)
    df = pairs.merge(lab, on=["set", "locale", "query_id", "product_id"], how="inner")
    df["llm_label"] = probs_to_label(df[PROB_COLS].to_numpy())
    df["llm_score"] = expected_gain(df[PROB_COLS].to_numpy())
    return df


def view_ndcg(view: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    """Per view query (in run_search order): NDCG@10 of its judged products ranked by LLM expected gain."""
    qids, nds = [], []
    for _, g in view.sort_values(["qorder", "product_id"], kind="stable").groupby("qorder", sort=True):
        lab = dict(zip(g["product_id"], g["gain"]))
        ranked = g["product_id"].to_numpy()[np.argsort(-g["llm_score"].to_numpy(), kind="stable")]
        qids.append(int(g["query_id"].iloc[0]))
        nds.append(ndcg_at_k(ranked.tolist(), lab, 10))
    return qids, np.array(nds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/judge/judge_pairs.parquet")
    ap.add_argument("--labels", default="data/judge/labels")
    ap.add_argument("--out", default="outputs/llm_judge")
    args = ap.parse_args()
    pairs = pd.read_parquet(args.pairs)
    df = labelled(pairs, Path(args.labels))
    model = sorted(df["model"].unique())
    out = {"model": model, "agreement": {}, "rerank_judged_ndcg10": {}}
    lines = [f"Judge: `{', '.join(model)}`\n", "### Agreement with ESCI human labels\n",
             "| Set | Locale | Pairs (queries) | Accuracy [95% CI] | Majority-label accuracy | Cohen's κ [95% CI] | "
             "κ, Exact vs rest | Recall E / S / C / I |", "|---|---|---|---|---|---|---|---|"]
    for set_name in ("view", "distill"):
        for loc in ("jp", "us"):
            d = df[(df["set"] == set_name) & (df["locale"] == loc)]
            if d.empty:
                continue
            a = agreement(d["esci_label"], d["llm_label"]) | agreement_ci(d)
            a["queries"] = int(d["query_id"].nunique())
            a["mean_letter_mass"] = float(d["label_mass"].mean())
            out["agreement"][f"{set_name}_{loc}"] = a
            rec = " / ".join("–" if a["recall_by_human_label"][l] is None else f"{a['recall_by_human_label'][l]:.2f}"
                             for l in "ESCI")
            lines.append(f"| {set_name} | {loc} | {a['n']:,} ({a['queries']:,}) | {a['accuracy']:.3f} "
                         f"[{a['accuracy_ci'][0]:.3f}, {a['accuracy_ci'][1]:.3f}] | {a['majority_label_accuracy']:.3f} | "
                         f"{a['kappa']:.3f} [{a['kappa_ci'][0]:.3f}, {a['kappa_ci'][1]:.3f}] | "
                         f"{a['exact_vs_rest_kappa']:.3f} | {rec} |")
    lines += ["", "Confusion matrices (rows human, columns LLM, order E S C I):", ""]
    for k, a in out["agreement"].items():
        lines.append(f"- {k}: `{a['confusion_rows_human_cols_llm']}`")

    lines += ["", "### Ranking each test query's judged products (NDCG@10)\n",
              "| Locale | Ranker | NDCG@10 [95% CI] | LLM − ranker [95% CI] | p (paired, 2-sided) | Queries |",
              "|---|---|---|---|---|---|"]
    for loc in ("jp", "us"):
        view = df[(df["set"] == "view") & (df["locale"] == loc)]
        if view.empty:
            continue
        full = pairs[(pairs["set"] == "view") & (pairs["locale"] == loc)].groupby("qorder").size()
        done = view.groupby("qorder").size()
        complete = done[done == full.reindex(done.index)].index
        view = view[view["qorder"].isin(complete)]
        qids, llm = view_ndcg(view)
        m, lo, hi = bootstrap_ci(llm)
        res = {"queries": len(qids), "llm_expected_gain": {"ndcg@10": m, "ci": [lo, hi]}}
        lines.append(f"| {loc} | **LLM judge (expected gain)** | {m:.3f} [{lo:.3f}, {hi:.3f}] | | | {len(qids)} |")
        for name, (tag, key) in REFERENCE.items():
            path = SEARCH / f"per_query{'_' + tag if tag else ''}.json"
            if not path.exists():
                continue
            pq = json.loads(path.read_text(encoding="utf-8"))[loc]
            pos = {q: i for i, q in enumerate(pq["query_id"])}
            vals = pq[key]
            ref = np.array([vals[pos[q]] if pos[q] < len(vals) and vals[pos[q]] is not None else np.nan
                            for q in qids], dtype=float)
            rm, rlo, rhi = bootstrap_ci(ref)
            dlt, dci, p, n = paired_delta(ref, llm)
            res[name] = {"ndcg@10": rm, "ci": [rlo, rhi], "llm_minus_ranker": dlt, "delta_ci": dci,
                         "p_two_sided": p, "n_queries": n}
            ptxt = "< 0.001" if p < 0.001 else f"{p:.3f}"
            lines.append(f"| {loc} | {name} | {rm:.3f} [{rlo:.3f}, {rhi:.3f}] | {dlt:+.3f} [{dci[0]:+.3f}, "
                         f"{dci[1]:+.3f}] | {ptxt} | {n} |")
        res["random_order"] = json.loads((SEARCH / "results.json").read_text(encoding="utf-8"))[loc]["rerank_judged"]["random"]
        out["rerank_judged_ndcg10"][loc] = res
    md = "\n".join(lines) + "\n"
    print(md)
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (o / "results.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
