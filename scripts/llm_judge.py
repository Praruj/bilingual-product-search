"""LLM relevance judge on Colab (free T4): label ESCI pairs with an open instruction model, then train a
cross-encoder on those labels. Driven by notebooks/colab_llm_judge.ipynb; needs a GPU except for --smoke.

    python scripts/llm_judge.py label --pairs judge_pairs.parquet --out labels/ --set view --hours 1.5
    python scripts/llm_judge.py train --pairs judge_pairs.parquet --labels labels/ --target llm --out models/x

label  One forward pass per pair: the chat prompt (bps.judge.SYSTEM + query + product) ends where the answer
       starts, and the next-token probabilities of the letters E / S / C / I (renormalised over those four) are
       the judgement. No sampling, so it is deterministic. Queries are processed in `qrank` order, alternating
       locales, in chunks written as separate parquet files; existing chunks are skipped, so a disconnected
       session resumes where it stopped and a time budget always leaves a random sample of whole queries.
train  The cross-encoder recipe of notebooks/colab_finetune.ipynb (same base model, loss and hyperparameters)
       on the distill pairs whose queries are fully labelled. --target human uses the ESCI gains of exactly
       the same pairs (the control); --target llm uses the gain of the LLM's argmax label.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps.judge import LABELS, SYSTEM, user_message  # noqa: E402
from bps.metrics import GAIN  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"   # Apache-2.0
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CE = dict(epochs=1, batch_size=64, lr=2e-5, warmup=0.1, max_len=128)   # as in colab_finetune.ipynb
PROB_COLS = [f"p_{l}" for l in LABELS]


def ordered_queries(pairs: pd.DataFrame) -> list[tuple[str, int]]:
    """(locale, query_id) in qrank order, locales alternating."""
    q = pairs.drop_duplicates(["locale", "query_id"])[["locale", "query_id", "qrank"]]
    return [(r.locale, int(r.query_id)) for r in q.sort_values(["qrank", "locale"]).itertuples()]


def chunks(pairs: pd.DataFrame, queries_per_chunk: int):
    order = ordered_queries(pairs)
    key = pairs.set_index(["locale", "query_id"]).sort_index()
    for i in range(0, len(order), queries_per_chunk):
        sel = order[i:i + queries_per_chunk]
        yield i // queries_per_chunk, key.loc[sel].reset_index()


def load_labels(label_dir: Path, set_name: str) -> pd.DataFrame:
    files = sorted(label_dir.glob(f"{set_name}_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


# --- label


def load_llm(name: str, quant: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.float16,
                                                        bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if quant != "4bit" and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    ids = []
    for l in LABELS:
        t = tok.encode(l, add_special_tokens=False)
        assert len(t) == 1, f"label {l!r} is not a single token: {t}"
        ids.append(t[0])
    return tok, model, ids


def prompts(tok, df: pd.DataFrame) -> list[str]:
    return [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": user_message(q, d)}],
                                    tokenize=False, add_generation_prompt=True)
            for q, d in zip(df["query"], df["doc"])]


def common_prefix(seqs: list[list[int]]) -> list[int]:
    first = seqs[0]
    n = len(first)
    for s in seqs[1:]:
        n = min(n, len(s))
        n = next((i for i in range(n) if s[i] != first[i]), n)
    return first[:n]


def judge_probs(tok, model, label_ids, texts: list[str], max_tokens: int,
                share_prefix: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """n x 4 probabilities over E/S/C/I (renormalised) and the probability mass the four letters had.

    share_prefix: the system prompt is the same in every prompt, so its keys/values are computed once and
    reused; each batch then only runs the query + product tokens (about 40% fewer tokens)."""
    import torch

    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    dev = next(model.parameters()).device
    prefix = common_prefix(enc)[:-1] if share_prefix and len(enc) > 1 else []   # keep >= 1 token per prompt
    P = len(prefix)
    if P:
        from transformers import DynamicCache
        with torch.no_grad():
            past = model.model(input_ids=torch.tensor([prefix], device=dev), past_key_values=DynamicCache(),
                               use_cache=True).past_key_values
        enc = [e[P:] for e in enc]
    order = np.argsort([len(e) for e in enc])
    probs, mass = np.zeros((len(texts), 4), np.float32), np.zeros(len(texts), np.float32)
    i = 0
    while i < len(order):   # batches of similar length, at most max_tokens tokens of padded input
        L = len(enc[order[i]])
        j = i
        while j < len(order) and (j - i + 1) * max(L, len(enc[order[j]])) <= max_tokens:
            L = max(L, len(enc[order[j]]))
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        batch = tok.pad({"input_ids": [enc[k] for k in idx]}, return_tensors="pt").to(dev)
        with torch.no_grad():
            pos = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0) + P   # positions as if unpadded
            kw = {}
            if P:
                cache = DynamicCache()
                for layer, (k, v) in enumerate(zip(past.key_cache, past.value_cache)):
                    cache.update(k.expand(len(idx), -1, -1, -1).contiguous(),
                                 v.expand(len(idx), -1, -1, -1).contiguous(), layer)
                ones = torch.ones(len(idx), P, dtype=batch["attention_mask"].dtype, device=dev)
                kw = {"past_key_values": cache, "use_cache": True,
                      "attention_mask": torch.cat([ones, batch["attention_mask"]], dim=1)}
            h = model.model(input_ids=batch["input_ids"], attention_mask=kw.pop("attention_mask", batch["attention_mask"]),
                            position_ids=pos, **kw).last_hidden_state[:, -1]   # left padding: last = answer slot
            logp = torch.log_softmax(model.lm_head(h).float(), dim=-1)[:, label_ids]
        probs[idx] = torch.softmax(logp, dim=-1).cpu().numpy()
        mass[idx] = logp.exp().sum(-1).cpu().numpy()
        i = j
    return probs, mass


def cmd_label(args):
    pairs = pd.read_parquet(args.pairs)
    pairs = pairs[pairs["set"] == args.set]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, label_ids = load_llm(args.model, args.quant)
    t0, done = time.time(), 0
    for n, df in chunks(pairs, args.queries_per_chunk):
        path = out / f"{args.set}_{n:05d}.parquet"
        if path.exists():
            continue
        if time.time() - t0 > args.hours * 3600 or (args.max_chunks and n >= args.max_chunks):
            print(f"stopping before chunk {n} (time budget or --max-chunks)", flush=True)
            break
        t = time.time()
        p, mass = judge_probs(tok, model, label_ids, prompts(tok, df), args.max_tokens, not args.no_shared_prefix)
        res = df[["set", "locale", "query_id", "product_id"]].copy()
        res[PROB_COLS] = p
        res["label_mass"] = mass
        res["model"] = args.model
        res.to_parquet(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").replace(path)   # a chunk file exists only once complete
        done += len(df)
        lab = np.array(LABELS)[p.argmax(1)]
        print(f"chunk {n}: {len(df)} pairs in {time.time() - t:.0f}s ({len(df) / (time.time() - t):.1f}/s), "
              f"agree {np.mean(lab == df['esci_label'].to_numpy()):.2f}, letter mass {mass.mean():.3f}; "
              f"total {done} pairs, {(time.time() - t0) / 60:.0f} min", flush=True)
    labelled = load_labels(out, args.set)
    print(f"{args.set}: {len(labelled):,} of {len(pairs):,} pairs labelled")


# --- train


def training_pairs(pairs: pd.DataFrame, labels: pd.DataFrame, target: str) -> pd.DataFrame:
    """Distill pairs whose query is fully labelled, with the training target `y`."""
    d = pairs[pairs["set"] == "distill"].merge(labels, on=["set", "locale", "query_id", "product_id"], how="left")
    full = d.groupby(["locale", "query_id"])["p_E"].transform(lambda s: s.notna().all())
    d = d[full].reset_index(drop=True)
    if target == "human":
        d["y"] = d["gain"].astype(np.float32)
    elif target == "llm":
        d["y"] = pd.Series(np.array(LABELS)[d[PROB_COLS].to_numpy().argmax(1)]).map(GAIN).astype(np.float32)
    else:
        raise ValueError(target)
    return d


def cmd_train(args):
    import torch
    import torch.nn.functional as F
    from sentence_transformers import CrossEncoder

    pairs = pd.read_parquet(args.pairs)
    d = training_pairs(pairs, load_labels(Path(args.labels), "distill"), args.target)
    if args.max_queries:
        keep = set(d.drop_duplicates(["locale", "query_id"]).sort_values(["qrank", "locale"])
                   ["query_id"].iloc[:args.max_queries])
        d = d[d["query_id"].isin(keep)].reset_index(drop=True)
    print(f"target {args.target}: {d['query_id'].nunique():,} queries, {len(d):,} pairs, "
          f"mean target {d['y'].mean():.3f}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ce = CrossEncoder(args.base, num_labels=1, max_length=CE["max_len"], device=dev)
    net, tok = ce.model.to(dev), ce.tokenizer   # CrossEncoder only moves the model in predict()
    q, doc, y = d["query"].to_numpy(), d["doc"].to_numpy(), d["y"].to_numpy(np.float32)
    bs = CE["batch_size"]
    n_steps = CE["epochs"] * math.ceil(len(y) / bs)
    opt = torch.optim.AdamW(net.parameters(), lr=CE["lr"], weight_decay=0.01)
    w = max(1, int(n_steps * CE["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / w, max(0.0, (n_steps - s) / max(1, n_steps - w))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    net.train()
    step, t0 = 0, time.time()
    for epoch in range(CE["epochs"]):
        order = np.random.default_rng(100 + epoch).permutation(len(y))
        for s in range(0, len(order), bs):
            i = order[s:s + bs]
            feats = tok(list(q[i]), list(doc[i]), truncation=True, max_length=CE["max_len"],
                        padding=True, return_tensors="pt").to(dev)
            target = torch.from_numpy(y[i]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = net(**feats).logits.squeeze(-1).float()
                loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 200 == 0 or step == n_steps:
                print(f"step {step}/{n_steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval()
    net.half()   # fp16 weights halve the download; transformers loads them as fp32 on the laptop
    ce.save(args.out)
    log = {"target": args.target, "base": args.base, "hparams": CE, "queries": int(d["query_id"].nunique()),
           "queries_by_locale": d.drop_duplicates(["locale", "query_id"])["locale"].value_counts().to_dict(),
           "pairs": int(len(d)), "mean_target": float(d["y"].mean()), "train_s": time.time() - t0, "device": dev}
    Path(args.out, "train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("label")
    a.add_argument("--pairs", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--set", choices=["view", "distill"], required=True)
    a.add_argument("--model", default=MODEL)
    a.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    a.add_argument("--hours", type=float, default=1.5)
    a.add_argument("--max-chunks", type=int, default=0)
    a.add_argument("--queries-per-chunk", type=int, default=20)
    a.add_argument("--max-tokens", type=int, default=6144, help="padded tokens per forward pass")
    a.add_argument("--no-shared-prefix", action="store_true", help="recompute the system prompt in every batch")
    t = sub.add_parser("train")
    t.add_argument("--pairs", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--target", choices=["human", "llm"], required=True)
    t.add_argument("--base", default=RERANKER)
    t.add_argument("--max-queries", type=int, default=0)
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    {"label": cmd_label, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
