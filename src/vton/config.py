"""Runtime settings, read once from the environment.

Every knob that changes cost/latency/quality lives here so a RunPod endpoint
can be re-tuned from its env-var panel without rebuilding the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# RunPod "cached models" mount their HF cache here; prefer it when present.
RUNPOD_HF_CACHE = Path("/runpod-volume/huggingface-cache/hub")


def resolve_hf_cache() -> str | None:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if RUNPOD_HF_CACHE.is_dir():
        return str(RUNPOD_HF_CACHE)
    return None


@dataclass(frozen=True)
class Settings:
    # Which diffusion backend to load: "qwen_edit" | "catvton_flux" | "mock"
    engine: str = field(default_factory=lambda: _env("VTON_ENGINE", "qwen_edit"))

    # Working resolution (portrait 3:4, the aspect of every mainstream VTON dataset).
    width: int = field(default_factory=lambda: _env_int("VTON_WIDTH", 768))
    height: int = field(default_factory=lambda: _env_int("VTON_HEIGHT", 1024))

    # Acceleration switches (see vton/accel.py).
    fp8: bool = field(default_factory=lambda: _env_bool("VTON_FP8", True))
    compile: bool = field(default_factory=lambda: _env_bool("VTON_COMPILE", False))
    first_block_cache: float = field(default_factory=lambda: _env_float("VTON_FBCACHE", 0.0))
    trt_engine_path: str = field(default_factory=lambda: _env("VTON_TRT_ENGINE", ""))

    # Run one throwaway inference at boot so compile/autotune cost is paid
    # before the worker reports ready (FlashBoot then snapshots a warm worker).
    warmup: bool = field(default_factory=lambda: _env_bool("VTON_WARMUP", True))

    # Model locations (overridable to point at private mirrors / bundles).
    flux_fill_repo: str = field(default_factory=lambda: _env("FLUX_FILL_REPO", "black-forest-labs/FLUX.1-Fill-dev"))
    catvton_repo: str = field(default_factory=lambda: _env("CATVTON_REPO", "zhengchong/CatVTON"))
    qwen_edit_repo: str = field(default_factory=lambda: _env("QWEN_EDIT_REPO", "Qwen/Qwen-Image-Edit-2511"))
    qwen_lightning_repo: str = field(
        default_factory=lambda: _env("QWEN_LIGHTNING_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")
    )
    qwen_lightning_file: str = field(
        default_factory=lambda: _env(
            "QWEN_LIGHTNING_FILE", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
        )
    )

    # Set when QWEN_EDIT_REPO points at a bundle with Lightning already merged
    # (scripts/build_bundle.py): 4-step schedule, no LoRA download.
    qwen_lightning_fused: bool = field(default_factory=lambda: _env_bool("QWEN_LIGHTNING_FUSED", False))

    # MediaPipe assets for the commercial-safe masker.
    mediapipe_dir: str = field(default_factory=lambda: _env("MEDIAPIPE_DIR", "/models/mediapipe"))

    # Guard rails.
    max_input_bytes: int = field(default_factory=lambda: _env_int("VTON_MAX_INPUT_BYTES", 15 * 1024 * 1024))
    max_input_pixels: int = field(default_factory=lambda: _env_int("VTON_MAX_INPUT_PIXELS", 40_000_000))
    fetch_timeout_s: float = field(default_factory=lambda: _env_float("VTON_FETCH_TIMEOUT", 15.0))

    hf_cache: str | None = field(default_factory=resolve_hf_cache)


def get_settings() -> Settings:
    return Settings()
