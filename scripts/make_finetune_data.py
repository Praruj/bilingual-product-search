"""Write the ESCI fine-tuning pairs (jp + us train split, minus the eval dev queries) for the Colab notebook.

Output: data/finetune/esci_train_pairs.parquet, one row per judged (query, product) pair. Upload it to
Google Drive as MyDrive/bps/esci_train_pairs.parquet and run notebooks/colab_finetune.ipynb.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps import data  # noqa: E402
from bps.finetune import assert_no_eval_leak, training_examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locales", nargs="+", default=["jp", "us"])
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--out", default="data/finetune/esci_train_pairs.parquet")
    args = ap.parse_args()

    frames, stats = [], {}
    for loc in args.locales:
        df, dropped = training_examples(loc, args.n_dev)
        # leakage checks against exactly what run_search.py evaluates on
        assert_no_eval_leak(df, data.load_examples(loc), dropped, args.n_test)
        stats[loc] = {"queries": int(df["query_id"].nunique()), "pairs": len(df),
                      "labels": df["esci_label"].value_counts().to_dict(),
                      "excluded_dev_queries": len(dropped)}
        print(loc, stats[loc], flush=True)
        frames.append(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    Path("outputs/finetune").mkdir(parents=True, exist_ok=True)
    Path("outputs/finetune/train_data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
