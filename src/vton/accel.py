"""Inference accelerators, each independently switchable per endpoint.

Order matters: fuse LoRA -> quantize -> cache hooks -> compile.

| switch            | what                                   | needs            |
|-------------------|----------------------------------------|------------------|
| VTON_FP8=1        | torchao FP8 dyn-act/FP8-weight, per-row| sm_89+ (L4, L40S, 4090, H100) |
| VTON_FBCACHE=0.08 | First-Block-Cache (skip redundant steps)| any              |
| VTON_COMPILE=1    | torch.compile (max-autotune, no cudagraph)| any, +boot time |
| VTON_TRT_ENGINE   | serialized TensorRT engine for the DiT | engine built on same GPU arch |
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def gpu_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()


def apply_fp8(module: torch.nn.Module) -> bool:
    """Weight+activation FP8 on Linear layers. Returns False when skipped."""
    if gpu_capability() < (8, 9):
        log.warning("FP8 requested but GPU capability %s < 8.9; staying in bf16", gpu_capability())
        return False
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_
    except ImportError:
        log.warning("torchao not installed; FP8 disabled")
        return False
    quantize_(module, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    log.info("FP8 (torchao, per-row) applied to %s", type(module).__name__)
    return True


def apply_first_block_cache(module: torch.nn.Module, threshold: float) -> bool:
    if threshold <= 0:
        return False
    from diffusers.hooks import FirstBlockCacheConfig

    module.enable_cache(FirstBlockCacheConfig(threshold=threshold))
    log.info("First-Block-Cache enabled (threshold=%s)", threshold)
    return True


def apply_compile(module: torch.nn.Module) -> torch.nn.Module:
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True
    # Static shapes: every request runs at the same canvas, so compile once.
    module.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
    log.info("torch.compile scheduled for %s", type(module).__name__)
    return module


def fuse_lora_file(transformer: torch.nn.Module, lora_path: str, scale: float = 1.0) -> None:
    """Load a diffusers/PEFT LoRA into `transformer`, bake it in, drop PEFT.

    Fusing removes the per-layer LoRA matmuls (free speed) and is a
    prerequisite for FP8 quantization and TensorRT export.
    """
    from safetensors.torch import load_file

    state = load_file(lora_path)
    prefix = "transformer" if any(k.startswith("transformer.") for k in state) else None
    transformer.load_lora_adapter(state, prefix=prefix, adapter_name="tryon")
    transformer.fuse_lora(lora_scale=scale, adapter_names=["tryon"])
    transformer.unload_lora()
    log.info("fused LoRA %s (%d tensors)", lora_path, len(state))
