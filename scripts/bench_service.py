"""Latency of the running API over real HTTP on this laptop's CPU, plus index sizes on disk.

Starts uvicorn as a subprocess, sends held-out ESCI test queries one at a time (no concurrency),
and records server-side and end-to-end latency per mode.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bps import data  # noqa: E402

INDEX = Path("outputs/index")
OUT = Path("outputs/service")
PORT = 8765
N_QUERIES = 200


def dir_mb(p: Path) -> dict:
    return {f.name: round(f.stat().st_size / 1e6, 1) for f in sorted(p.iterdir())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path("outputs/search/results.json").read_text(encoding="utf-8"))["config"]
    queries = {}
    for loc in ("jp", "us"):
        test, *_ = data.eval_setup(loc, cfg["n_test"], cfg["n_dev"], cfg["corpus"])
        queries[loc] = list(test.queries["query"])[:N_QUERIES]

    env = os.environ | {"PYTHONPATH": "src"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "bps.service:app", "--port", str(PORT),
                             "--log-level", "warning"], env=env)
    try:
        t0 = time.time()
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=60) as c:
            while True:
                try:
                    if c.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.time() - t0 > 300:
                    raise RuntimeError("server did not start")
                time.sleep(1)
            startup_s = time.time() - t0
            res = {"cpu": platform.processor(), "torch_threads": "default (all cores)",
                   "startup_s": startup_s, "n_queries_per_locale": N_QUERIES, "latency_ms": {}}
            for loc, qs in queries.items():
                for q in qs[:5]:  # warm-up
                    c.get("/search", params={"q": q, "locale": loc, "mode": "rerank"})
                for mode in ("bm25", "dense", "hybrid", "rerank"):
                    server, e2e = [], []
                    for q in qs:
                        t = time.perf_counter()
                        r = c.get("/search", params={"q": q, "locale": loc, "mode": mode, "k": 10})
                        e2e.append((time.perf_counter() - t) * 1e3)
                        server.append(r.json()["took_ms"])
                    res["latency_ms"][f"{loc}_{mode}"] = {
                        "server_p50": float(np.percentile(server, 50)), "server_p95": float(np.percentile(server, 95)),
                        "http_p50": float(np.percentile(e2e, 50)), "http_p95": float(np.percentile(e2e, 95))}
                    print(loc, mode, res["latency_ms"][f"{loc}_{mode}"], flush=True)
            users = json.loads((INDEX / "recs" / "meta.json").read_text())["users"][:N_QUERIES]
            lat = []
            for u in users:
                t = time.perf_counter()
                c.get(f"/recommend/{u}")
                lat.append((time.perf_counter() - t) * 1e3)
            res["latency_ms"]["recommend"] = {"http_p50": float(np.percentile(lat, 50)),
                                              "http_p95": float(np.percentile(lat, 95))}
            res["examples"] = {
                "ja": c.get("/search", params={"q": "ワイヤレスイヤホン ノイズキャンセリング", "mode": "rerank", "k": 3}).json(),
                "en": c.get("/search", params={"q": "stainless steel water bottle", "mode": "rerank", "k": 3}).json(),
                "recommend": c.get(f"/recommend/{users[0]}", params={"k": 3}).json()}
        try:
            import psutil

            # on Windows the venv python.exe is a launcher; the server is its child
            procs = [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]
            res["server_rss_mb"] = max(p.memory_info().rss for p in procs) / 1e6
        except ImportError:
            pass
    finally:
        proc.terminate()
    res["index_mb"] = {d.name: dir_mb(d) for d in sorted(INDEX.iterdir())}
    (OUT / "benchmark.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT / 'benchmark.json'}")


if __name__ == "__main__":
    main()
