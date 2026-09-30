"""In-worker benchmark: latency per stage, peak VRAM, $/try-on.

Run inside the worker image on a RunPod GPU pod (same GPU type as the endpoint):

    python scripts/bench.py --engine qwen_edit --gpu-price-hr 1.75 --runs 10 \
        --people .assets/person --garments .assets/garment --out out/bench_qwen.json

Env switches (VTON_FP8, VTON_FBCACHE, VTON_COMPILE, VTON_TRT_ENGINE) select the preset,
so the same script produces every row of the optimisation table.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.environ.get("VTON_ENGINE", "qwen_edit"))
    ap.add_argument("--people", default=str(ROOT / ".assets/person"))
    ap.add_argument("--garments", default=str(ROOT / ".assets/garment"))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--gpu-price-hr", type=float, default=1.75, help="RunPod flex $/hr for this GPU")
    ap.add_argument("--save-images", default="out/bench_images")
    ap.add_argument("--out", default="out/bench.json")
    args = ap.parse_args()
    os.environ["VTON_ENGINE"] = args.engine

    import torch

    from vton.config import get_settings
    from vton.engines import create_engine
    from vton.masking import MediaPipeMasker
    from vton.service import TryOnService

    s = get_settings()
    t0 = time.perf_counter()
    engine = create_engine(s)
    engine.load()
    load_s = time.perf_counter() - t0
    svc = TryOnService(s, engine, MediaPipeMasker(s.mediapipe_dir))
    t0 = time.perf_counter()
    svc.warmup()
    warm_s = time.perf_counter() - t0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    enc = lambda p: base64.b64encode(Path(p).read_bytes()).decode()  # noqa: E731
    people = sorted(Path(args.people).glob("*.*"))
    garments = sorted(Path(args.garments).glob("*.*"))
    pairs = list(itertools.islice(itertools.cycle(itertools.product(people, garments)), args.runs))
    save = Path(args.save_images)
    save.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (p, g) in enumerate(pairs):
        cat = "overall" if g.name.startswith("overall") else "lower" if g.name.startswith("lower") else "upper"
        req = {"person_image": enc(p), "garment_image": enc(g), "category": cat, "seed": i, "output_format": "jpeg"}
        if args.steps:
            req["steps"] = args.steps
        try:
            out = svc.run(req)
        except Exception as e:  # keep benchmarking past a bad photo
            print(f"[{i}] {p.name} x {g.name}: {e}")
            continue
        rows.append(out["timings_ms"])
        (save / f"{i:02d}_{p.stem}__{g.stem}.jpg").write_bytes(base64.b64decode(out["image"].split(",", 1)[1]))
        print(f"[{i}] {p.name} x {g.name} ({cat}): {out['timings_ms']['total']:.0f} ms")

    stages = sorted({k for r in rows for k in r})
    summary = {k: {"p50": round(statistics.median([r[k] for r in rows if k in r]), 1),
                   "p95": round(pct([r[k] for r in rows if k in r], 95), 1)} for k in stages}
    p50_s = summary["total"]["p50"] / 1000
    report = {
        "engine": args.engine,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "preset": {k: os.environ.get(k) for k in ("VTON_FP8", "VTON_FBCACHE", "VTON_COMPILE", "VTON_TRT_ENGINE")},
        "steps": args.steps or getattr(engine, "default_steps", None),
        "runs": len(rows),
        "load_s": round(load_s, 1),
        "warmup_s": round(warm_s, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "latency_ms": summary,
        "usd_per_tryon_at_p50": round(p50_s * args.gpu_price_hr / 3600, 5),
        "tryons_per_gpu_hour": round(3600 / p50_s) if p50_s else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
