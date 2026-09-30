"""Download exactly the weight files an engine loads (image bake / volume warm).

    python scripts/fetch_models.py --engine qwen_edit
    python scripts/fetch_models.py --engine catvton_flux     # needs HF_TOKEN (gated FLUX repo)
    python scripts/fetch_models.py --engine catvton_flux --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vton.config import Settings  # noqa: E402


def plan(engine: str, s: Settings) -> list[tuple[str, list[str] | None]]:
    """(repo_id, allow_patterns) pairs; None = whole repo."""
    if engine == "catvton_flux":
        return [
            # transformer + vae + scheduler only; no CLIP/T5, no duplicate single-file checkpoint
            (s.flux_fill_repo, ["transformer/*", "vae/*", "scheduler/*", "model_index.json"]),
            (s.catvton_repo, ["flux-lora/*"]),
        ]
    if engine == "qwen_edit":
        return [
            (s.qwen_edit_repo, ["*.json", "transformer/*", "vae/*", "text_encoder/*", "tokenizer/*", "processor/*", "scheduler/*"]),
            (s.qwen_lightning_repo, [s.qwen_lightning_file]),
        ]
    if engine == "mock":
        return []
    raise SystemExit(f"unknown engine {engine}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dry-run", action="store_true", help="list files and sizes without downloading")
    args = ap.parse_args()
    s = Settings()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    total = 0
    for repo, patterns in plan(args.engine, s):
        if args.dry_run:
            import fnmatch

            info = api.model_info(repo, files_metadata=True)
            files = [f for f in info.siblings if patterns is None or any(fnmatch.fnmatch(f.rfilename, p) for p in patterns)]
            size = sum(f.size or 0 for f in files)
            total += size
            print(f"{repo}: {len(files)} files, {size / 1e9:.2f} GB")
        else:
            path = snapshot_download(repo, allow_patterns=patterns, cache_dir=s.hf_cache)
            print(f"{repo} -> {path}")
    if args.dry_run:
        print(f"TOTAL {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
