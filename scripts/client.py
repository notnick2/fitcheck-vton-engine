"""Call a deployed RunPod endpoint (single shot or small load test).

    export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
    python scripts/client.py --person me.jpg --garment shirt.jpg --category upper
    python scripts/client.py --person me.jpg --garment shirt.jpg --concurrency 8 --requests 32

Uses /runsync (20 MB payload cap) for single shots and /run + /status polling
for load tests, the pattern a mobile backend should use.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

API = "https://api.runpod.ai/v2"


def _post(url: str, body: dict | None, key: str) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def as_ref(s: str) -> str:
    return s if s.startswith(("http://", "https://")) else base64.b64encode(Path(s).read_bytes()).decode()


def run_async(endpoint: str, key: str, payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    job = _post(f"{API}/{endpoint}/run", {"input": payload}, key)
    while True:
        st = _post(f"{API}/{endpoint}/status/{job['id']}", None, key)
        if st["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return st, time.perf_counter() - t0
        time.sleep(0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--garment", required=True)
    ap.add_argument("--category", default="upper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", default="out/client")
    args = ap.parse_args()
    key, endpoint = os.environ["RUNPOD_API_KEY"], os.environ["RUNPOD_ENDPOINT_ID"]
    payload = {"person_image": as_ref(args.person), "garment_image": as_ref(args.garment),
               "category": args.category, "seed": args.seed, "return_mask": True}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.requests == 1:
        t0 = time.perf_counter()
        res = _post(f"{API}/{endpoint}/runsync", {"input": payload}, key)
        wall = time.perf_counter() - t0
        results = [(res, wall)]
    else:
        with cf.ThreadPoolExecutor(args.concurrency) as pool:
            futs = [pool.submit(run_async, endpoint, key, {**payload, "seed": args.seed + i}) for i in range(args.requests)]
            results = [f.result() for f in futs]

    walls, execs, delays = [], [], []
    for i, (res, wall) in enumerate(results):
        if res.get("status") != "COMPLETED":
            print(f"[{i}] {res.get('status')}: {res.get('error') or res.get('output')}")
            continue
        out = res["output"]
        for k in ("image", "mask"):
            if k in out and out[k].startswith("data:"):
                (out_dir / f"{i:02d}_{k}.webp").write_bytes(base64.b64decode(out[k].split(",", 1)[1]))
        walls.append(wall)
        execs.append(res.get("executionTime", 0) / 1000)
        delays.append(res.get("delayTime", 0) / 1000)
        print(f"[{i}] wall {wall:.2f}s  exec {execs[-1]:.2f}s  queue/cold {delays[-1]:.2f}s  stages {out['timings_ms']}")

    if walls:
        print(f"\nok {len(walls)}/{len(results)}  wall p50 {statistics.median(walls):.2f}s  "
              f"exec p50 {statistics.median(execs):.2f}s  max queue/cold {max(delays):.2f}s  -> {out_dir}")


if __name__ == "__main__":
    main()
