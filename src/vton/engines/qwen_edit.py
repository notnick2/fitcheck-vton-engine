"""Qwen-Image-Edit-2511 try-on: the commercially clean production engine.

* Qwen-Image-Edit-2511: Apache-2.0. Multi-image editing (person + garment)
  without any try-on LoRA.
* Lightning 4-step distillation LoRA (lightx2v), fused -> 4 NFEs at CFG 1
  instead of 40 steps x 2 (CFG) = ~20x fewer transformer calls.
* FP8 transformer via torchao on sm_89+ (L40S / L4 / 4090 / H100).
* Optional Nunchaku SVDQuant INT4/FP4 transformer (QWEN_NUNCHAKU_TRANSFORMER)
  for 24 GB cards.

The model edits the whole frame; our mask is then used for the identity-lock
composite (vton.compose), so face/hair/background come from the user's real
photo pixel-for-pixel.
"""

from __future__ import annotations

import logging
import math
import os

import torch
from PIL import Image

from .. import accel
from ..config import Settings
from .base import GenerateParams, TryOnEngine

log = logging.getLogger(__name__)

PROMPTS = {
    "upper": (
        "Dress the person in image 1 in the top from image 2. Replace only their upper-body garment. "
        "Reproduce the garment from image 2 exactly: colour, print, logos, text, fabric texture, neckline, "
        "sleeve length and fit. Keep the person's face, identity, hair, skin tone, body shape, pose, hands, "
        "lower-body clothing, background and lighting from image 1 unchanged."
    ),
    "lower": (
        "Dress the person in image 1 in the bottoms from image 2. Replace only their lower-body garment. "
        "Reproduce the garment from image 2 exactly: colour, pattern, fabric texture, length and fit. "
        "Keep the person's face, identity, hair, skin tone, body shape, pose, top, shoes, background and "
        "lighting from image 1 unchanged."
    ),
    "overall": (
        "Dress the person in image 1 in the full outfit from image 2. Replace their clothing with it. "
        "Reproduce the outfit from image 2 exactly: colour, print, fabric texture, silhouette and length. "
        "Keep the person's face, identity, hair, skin tone, body shape, pose, background and lighting from "
        "image 1 unchanged."
    ),
}

# Scheduler config published with the Qwen-Image Lightning LoRAs.
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class QwenEditEngine(TryOnEngine):
    name = "qwen_edit"
    commercial_ok = True  # Apache-2.0 base; verify the Lightning LoRA license before launch
    needs_mask = False

    def __init__(self, settings: Settings, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        self.settings = settings
        self.device = torch.device(device)
        self.dtype = dtype
        self.pipe = None
        self.lightning = bool(settings.qwen_lightning_repo) or settings.qwen_lightning_fused
        self.default_steps = 4 if self.lightning else 40
        self.default_guidance = 1.0 if self.lightning else 4.0

    def load(self) -> None:
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

        s = self.settings
        kw = {"cache_dir": s.hf_cache} if s.hf_cache else {}
        extra = {}
        nunchaku = os.environ.get("QWEN_NUNCHAKU_TRANSFORMER")
        if nunchaku:
            from nunchaku import NunchakuQwenImageTransformer2DModel

            extra["transformer"] = NunchakuQwenImageTransformer2DModel.from_pretrained(nunchaku)
        if self.lightning:
            extra["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)

        pipe = QwenImageEditPlusPipeline.from_pretrained(s.qwen_edit_repo, torch_dtype=self.dtype, **extra, **kw)
        # Skip when weights already have Lightning merged (our bundle / nunchaku builds).
        if self.lightning and not nunchaku and not s.qwen_lightning_fused:
            pipe.load_lora_weights(s.qwen_lightning_repo, weight_name=s.qwen_lightning_file, adapter_name="lightning", **kw)
            pipe.fuse_lora(lora_scale=1.0, adapter_names=["lightning"])
            pipe.unload_lora_weights()
        pipe.set_progress_bar_config(disable=True)

        # VRAM choreography: bf16 transformer (~41 GB) + text encoder (~17 GB)
        # would not fit a 48 GB L40S together. Quantize the transformer on
        # the GPU first (-> ~20 GB), *then* bring the rest over.
        pipe.transformer.to(self.device)
        if s.fp8 and not nunchaku:
            accel.apply_fp8(pipe.transformer)
            torch.cuda.empty_cache()
        pipe.to(self.device)

        accel.apply_first_block_cache(pipe.transformer, s.first_block_cache)
        if s.compile:
            accel.apply_compile(pipe.transformer)
        self.pipe = pipe

    @torch.inference_mode()
    def generate(self, person, garment, mask, params: GenerateParams) -> Image.Image:
        w, h = person.size
        guidance = params.guidance if params.guidance is not None else self.default_guidance
        out = self.pipe(
            image=[person, garment],
            prompt=PROMPTS[params.category],
            negative_prompt=" ",
            true_cfg_scale=guidance,
            num_inference_steps=params.steps or self.default_steps,
            height=h,
            width=w,
            generator=torch.Generator(device="cpu").manual_seed(params.seed),
        ).images[0]
        return out if out.size == (w, h) else out.resize((w, h), Image.LANCZOS)
