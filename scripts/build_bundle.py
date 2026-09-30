"""Merge the Lightning LoRA into Qwen-Image-Edit-2511 and publish one repo.

RunPod "cached models" allow exactly one HF repo per endpoint and don't bill
for the download. Shipping base + LoRA as a single pre-fused repo means:
no LoRA fetch or fuse at boot, and one cache entry.

    python scripts/build_bundle.py --out /workspace/qwen-edit-2511-lightning \
        --push fitcheck/qwen-edit-2511-lightning-4step --private

Then on the endpoint:
    QWEN_EDIT_REPO=fitcheck/qwen-edit-2511-lightning-4step  QWEN_LIGHTNING_FUSED=1
Needs ~64 GB of CPU RAM (runs fine on any RunPod pod; no GPU required).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    import torch
    from diffusers import QwenImageEditPlusPipeline

    from vton.config import Settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", help="HF repo id to upload to")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    s = Settings()

    pipe = QwenImageEditPlusPipeline.from_pretrained(s.qwen_edit_repo, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(s.qwen_lightning_repo, weight_name=s.qwen_lightning_file, adapter_name="lightning")
    pipe.fuse_lora(lora_scale=1.0, adapter_names=["lightning"])
    pipe.unload_lora_weights()
    pipe.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    Path(args.out, "BUNDLE.md").write_text(
        f"Base: {s.qwen_edit_repo} (Apache-2.0)\nFused: {s.qwen_lightning_repo}/{s.qwen_lightning_file}\n"
        "Serve with QWEN_LIGHTNING_FUSED=1 (4 steps, CFG 1.0).\n"
    )
    print("saved", args.out)

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, private=args.private, exist_ok=True)
        api.upload_large_folder(repo_id=args.push, folder_path=args.out, repo_type="model")
        print("pushed", args.push)


if __name__ == "__main__":
    main()
