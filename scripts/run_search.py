"""Search evaluation on ESCI, separately for Japanese (jp) and English (us) queries.

For each locale: sample test queries (and dev queries from the train split for tuning the hybrid weight),
build a corpus of their judged products plus distractors, then compare
BM25 (several tokenizers) / multilingual E5 / hybrid / hybrid + cross-encoder rerank.

Two views:
  retrieval  rank the whole corpus: NDCG@10 and recall@100 of Exact products
  rerank     reorder only the judged candidates of each query (the official ESCI task 1 setting)

--encoder / --reranker take a local model directory (e.g. the fine-tuned models from
notebooks/colab_finetune.ipynb) and need --tag, which names the outputs (results_<tag>.json, the
embedding cache) so the off-the-shelf results.json is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps import data  # noqa: E402
from bps.bm25 import BM25, top_k  # noqa: E402
from bps.dense import E5, RERANKER, DenseIndex, Reranker  # noqa: E402
from bps.hybrid import convex, rrf  # noqa: E402
from bps.metrics import bootstrap_ci, ndcg_at_k, paired_bootstrap_p, recall_at_k  # noqa: E402
from bps.text import TOKENIZERS, has_japanese  # noqa: E402

OUT = Path("outputs/search")
TOKENIZERS_BY_LOCALE = {"jp": ["whitespace", "char_bigram", "mecab"], "us": ["whitespace", "word"]}
MAIN_TOKENIZER = {"jp": "mecab", "us": "word"}
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)


def per_query(ranked_ids, split, k_ndcg=10, k_rec=100):
    nd, rc = [], []
    for ranked, qid in zip(ranked_ids, split.queries["query_id"]):
        lab = split.labels[int(qid)]
        nd.append(ndcg_at_k(ranked, lab, k_ndcg))
        rc.append(recall_at_k(ranked, {p for p, g in lab.items() if g == 1.0}, k_rec))
    return np.array(nd), np.array(rc)


def summarise(nd, rc):
    m, lo, hi = bootstrap_ci(nd)
    r, rlo, rhi = bootstrap_ci(rc)
    return {"ndcg@10": m, "ndcg@10_ci": [lo, hi], "recall@100": r, "recall@100_ci": [rlo, rhi]}


def model_fingerprint(name: str) -> str:
    """Hub name as is; for a local directory, a hash of its weight files' names, sizes and mtimes, so a
    retrained model saved to the same path does not reuse stale embeddings."""
    path = Path(name)
    if not path.is_dir():
        return name
    files = sorted(f for f in path.rglob("*") if f.suffix in {".safetensors", ".bin"})
    key = "|".join(f"{f.relative_to(path)}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files)
    return f"{path.resolve()}#{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def run_locale(locale, args, encoder, reranker, per_query_out):
    t0 = time.time()
    test, dev, corpus_ids, docs = data.eval_setup(locale, args.n_test, args.n_dev, args.corpus)
    ids = np.array(corpus_ids)
    pos = {p: i for i, p in enumerate(corpus_ids)}
    print(f"[{locale}] {len(test.queries)} test / {len(dev.queries)} dev queries, corpus {len(docs):,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    res = {"n_test_queries": len(test.queries), "n_dev_queries": len(dev.queries), "corpus_size": len(docs),
           "share_test_queries_with_japanese_chars": float(np.mean([has_japanese(q) for q in test.queries["query"]])),
           "retrieval": {}, "rerank_judged": {}, "timing_s": {}, "index_bytes": {}}
    per_q = {}

    def evaluate(name, rank_fn, split):
        t = time.perf_counter()
        ranked = [rank_fn(q) for q in split.queries["query"]]
        dt = (time.perf_counter() - t) / len(ranked)
        nd, rc = per_query([ids[r].tolist() for r in ranked], split)
        return nd, rc, dt, ranked

    # --- BM25 with each tokenizer
    bm25 = {}
    for tok in TOKENIZERS_BY_LOCALE[locale]:
        t = time.perf_counter()
        bm25[tok] = BM25(TOKENIZERS[tok]).fit(docs)
        res["timing_s"][f"bm25_{tok}_index"] = time.perf_counter() - t
        res["index_bytes"][f"bm25_{tok}"] = bm25[tok].nbytes
        nd, rc, dt, _ = evaluate(f"bm25_{tok}", lambda q, b=bm25[tok]: b.search(q, 100)[0], test)
        res["retrieval"][f"bm25_{tok}"] = summarise(nd, rc) | {"ms_per_query": dt * 1e3}
        per_q[f"bm25_{tok}"] = nd
        print(f"[{locale}] bm25_{tok}: ndcg@10 {np.nanmean(nd):.3f} recall@100 {np.nanmean(rc):.3f}", flush=True)
    sparse = bm25[MAIN_TOKENIZER[locale]]

    # --- dense
    t = time.perf_counter()
    dense = DenseIndex(encoder)
    suffix = "" if args.encoder == E5 else f"_{args.tag}"
    cache = OUT / f"emb_{locale}_{len(docs)}{suffix}.npy"
    stamp = cache.with_suffix(".model.txt")
    fp = model_fingerprint(args.encoder)
    if cache.exists() and (args.encoder == E5 or (stamp.exists() and stamp.read_text(encoding="utf-8") == fp)):
        dense.emb = np.load(cache)
    else:
        dense.fit(docs)
        np.save(cache, dense.emb)
        if args.encoder != E5:
            stamp.write_text(fp, encoding="utf-8")
    res["timing_s"]["dense_index"] = time.perf_counter() - t
    res["index_bytes"]["dense_fp32"] = dense.nbytes
    nd, rc, dt, _ = evaluate("e5", lambda q: dense.search(q, 100)[0], test)
    res["retrieval"]["e5_small"] = summarise(nd, rc) | {"ms_per_query": dt * 1e3}
    per_q["e5_small"] = nd
    print(f"[{locale}] e5: ndcg@10 {np.nanmean(nd):.3f} recall@100 {np.nanmean(rc):.3f}", flush=True)

    # --- hybrid: tune alpha on dev queries, then apply once to test
    def score_pairs(split):
        """(sparse, dense) full score vectors, one query at a time to keep memory flat."""
        qs = list(split.queries["query"])
        qe = dense.encode(qs, "query: ")
        for q, e in zip(qs, qe):
            yield sparse.scores(q), dense.emb @ e

    ranked_by_alpha = {float(a): [] for a in ALPHAS}
    for s, d in score_pairs(dev):
        for a in ALPHAS:
            ranked_by_alpha[float(a)].append(ids[convex(s, d, a)].tolist())
    dev_curve = {a: float(np.nanmean(per_query(r, dev)[0])) for a, r in ranked_by_alpha.items()}
    alpha = max(dev_curve, key=dev_curve.get)
    res["hybrid_alpha_dev_curve"] = dev_curve
    res["hybrid_alpha"] = alpha

    hyb, fused = [], []
    for s, d in score_pairs(test):
        hyb.append(convex(s, d, alpha))
        fused.append(rrf([top_k(s, 100)[0], top_k(d, 100)[0]]))
    nd, rc = per_query([ids[r].tolist() for r in hyb], test)
    res["retrieval"]["hybrid_convex"] = summarise(nd, rc)
    per_q["hybrid_convex"] = nd
    nd, rc = per_query([ids[r].tolist() for r in fused], test)
    res["retrieval"]["hybrid_rrf"] = summarise(nd, rc)
    per_q["hybrid_rrf"] = nd
    print(f"[{locale}] hybrid alpha={alpha}: ndcg@10 {res['retrieval']['hybrid_convex']['ndcg@10']:.3f}; "
          f"rrf {res['retrieval']['hybrid_rrf']['ndcg@10']:.3f}", flush=True)

    # --- rerank the hybrid top-N with the cross-encoder (recall@100 is unchanged inside the top-N)
    t = time.perf_counter()
    rr = []
    for q, cand in zip(test.queries["query"], hyb):
        top = reranker.rerank(q, cand[: args.rerank_n], docs)
        rr.append(np.concatenate([top, cand[args.rerank_n:]]))
    res["timing_s"]["rerank_per_query"] = (time.perf_counter() - t) / len(rr)
    nd, rc = per_query([ids[r].tolist() for r in rr], test)
    res["retrieval"][f"hybrid_rerank{args.rerank_n}"] = summarise(nd, rc)
    per_q[f"hybrid_rerank{args.rerank_n}"] = nd
    print(f"[{locale}] rerank: ndcg@10 {np.nanmean(nd):.3f} "
          f"({res['timing_s']['rerank_per_query'] * 1e3:.0f} ms/query)", flush=True)

    # --- paired significance against the best BM25 and against dense
    best_bm25 = max((k for k in per_q if k.startswith("bm25")), key=lambda k: np.nanmean(per_q[k]))
    res["paired_p_vs_best_bm25"] = {k: paired_bootstrap_p(per_q[best_bm25], v)
                                    for k, v in per_q.items() if k != best_bm25}
    res["paired_p_vs_e5"] = {k: paired_bootstrap_p(per_q["e5_small"], v)
                             for k, v in per_q.items() if not k.startswith("bm25") and k != "e5_small"}

    # --- split by whether the query contains Japanese characters (jp locale has many English/model-number queries)
    if locale == "jp":
        mask = np.array([has_japanese(q) for q in test.queries["query"]])
        res["by_script"] = {name: {"japanese_chars": float(np.nanmean(v[mask])),
                                   "latin_only": float(np.nanmean(v[~mask]))} for name, v in per_q.items()}

    # --- rerank-only view: judged candidates of each test query, as in the ESCI benchmark
    view = {}
    for name, fn in {"bm25": lambda q, c: sparse.scores(q)[c],
                     "e5_small": lambda q, c: dense.emb[c] @ dense.encode([q], "query: ")[0]}.items():
        nds = []
        for q, qid in zip(test.queries["query"], test.queries["query_id"]):
            lab = test.labels[int(qid)]
            cand = np.array([pos[p] for p in lab])
            order = cand[np.argsort(-fn(q, cand), kind="stable")]
            nds.append(ndcg_at_k(ids[order].tolist(), lab, 10))
        view[name] = bootstrap_ci(nds)[0]
        per_q[f"rerank_judged_{name}"] = np.array(nds)
    nds = []
    for q, qid in zip(test.queries["query"][: args.n_rerank_view], test.queries["query_id"][: args.n_rerank_view]):
        lab = test.labels[int(qid)]
        cand = np.array([pos[p] for p in lab])
        nds.append(ndcg_at_k(ids[reranker.rerank(q, cand, docs)].tolist(), lab, 10))
    view["cross_encoder"] = bootstrap_ci(nds)[0]
    view["cross_encoder_ci"] = list(bootstrap_ci(nds)[1:])
    per_q["rerank_judged_cross_encoder"] = np.array(nds)
    view["random"] = float(np.nanmean([_random_ndcg(test.labels[int(q)]) for q in test.queries["query_id"]]))
    view["note"] = (f"cross_encoder on the first {len(nds)} test queries; candidates per query: "
                    f"{np.mean([len(v) for v in test.labels.values()]):.1f}")
    res["rerank_judged"] = view

    # --- examples for the README
    res["examples"] = []
    for q, cand in list(zip(test.queries["query"], rr))[:: max(1, len(rr) // 6)][:6]:
        res["examples"].append({"query": q, "top3": [docs[i][:90] for i in cand[:3]]})
    res["wall_s"] = time.time() - t0
    # per-query NDCG@10 (same query order in every run), for paired comparisons between runs
    per_query_out[locale] = {"query_id": [int(q) for q in test.queries["query_id"]],
                             **{k: [None if np.isnan(x) else float(x) for x in v] for k, v in per_q.items()}}
    return res


def _random_ndcg(lab, n=50, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(lab)
    return float(np.mean([ndcg_at_k(rng.permutation(ids).tolist(), lab, 10) for _ in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locales", nargs="+", default=["jp", "us"])
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--corpus", type=int, default=50_000)
    ap.add_argument("--rerank-n", type=int, default=20)
    ap.add_argument("--n-rerank-view", type=int, default=200)
    ap.add_argument("--encoder", default=E5, help="hub name or local SentenceTransformer directory")
    ap.add_argument("--reranker", default=RERANKER, help="hub name or local CrossEncoder directory")
    ap.add_argument("--tag", default="", help="output name; required with a non-default encoder/reranker")
    args = ap.parse_args()
    if (args.encoder != E5 or args.reranker != RERANKER) and not args.tag:
        ap.error("--tag is required with a custom --encoder or --reranker")
    OUT.mkdir(parents=True, exist_ok=True)
    encoder, reranker = DenseIndex(name=args.encoder).model, Reranker(name=args.reranker)
    suffix = f"_{args.tag}" if args.tag else ""
    path, pq_path = OUT / f"results{suffix}.json", OUT / f"per_query{suffix}.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    per_query_out = json.loads(pq_path.read_text(encoding="utf-8")) if pq_path.exists() else {}
    results["config"] = vars(args) | {"encoder_fingerprint": model_fingerprint(args.encoder),
                                       "reranker_fingerprint": model_fingerprint(args.reranker)}
    for loc in args.locales:
        results[loc] = run_locale(loc, args, encoder, reranker, per_query_out)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        pq_path.write_text(json.dumps(per_query_out), encoding="utf-8")
    print(f"wrote {path} and {pq_path}")


if __name__ == "__main__":
    main()
