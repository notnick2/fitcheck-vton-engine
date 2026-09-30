"""CatVTON on FLUX.1-Fill-dev, rebuilt for serving.

Changes vs. the upstream CatVTON-FLUX implementation:

* No text encoders. CatVTON-FLUX never uses a prompt, yet
  `FluxTryOnPipeline.from_pretrained` still pulls the whole repo. Loading only
  `transformer/`, `vae/` and `scheduler/` skips CLIP + T5 (~10 GB of download
  and ~10 GB of VRAM) and shrinks the image / cold start accordingly.
* Works on current diffusers. The upstream loop passes
  `encoder_hidden_states=None`, which crashes in `context_embedder` on any
  recent diffusers.
  We pass a zero-length text sequence instead: mathematically the same
  "no text tokens" attention, and valid for every diffusers version.
* LoRA fused into the weights (no per-layer LoRA matmuls, FP8/TRT-ready).
* Pooled-projection width read from the model config instead of a
  hard-coded 768.

LICENSE WARNING: FLUX.1-Fill-dev (FLUX.1 [dev] non-commercial) and the
CatVTON LoRA (CC BY-NC-SA 4.0) are NOT licensed for a paid app. This engine
exists for parity/benchmarking and for use under a BFL commercial license.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from .. import accel
from ..config import Settings
from .base import GenerateParams, TryOnEngine

log = logging.getLogger(__name__)


class CatVTONFluxEngine(TryOnEngine):
    name = "catvton_flux"
    commercial_ok = False
    needs_mask = True
    default_steps = 28
    default_guidance = 30.0

    def __init__(self, settings: Settings, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        self.settings = settings
        self.device = torch.device(device)
        self.dtype = dtype
        self.pipe = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
        from huggingface_hub import hf_hub_download

        s = self.settings
        kw = {"cache_dir": s.hf_cache} if s.hf_cache else {}
        transformer = FluxTransformer2DModel.from_pretrained(
            s.flux_fill_repo, subfolder="transformer", torch_dtype=self.dtype, **kw
        )
        vae = AutoencoderKL.from_pretrained(s.flux_fill_repo, subfolder="vae", torch_dtype=self.dtype, **kw)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(s.flux_fill_repo, subfolder="scheduler", **kw)
        lora = hf_hub_download(s.catvton_repo, "flux-lora/pytorch_lora_weights.safetensors", **kw)
        accel.fuse_lora_file(transformer, lora)

        self._assemble(transformer, vae, scheduler)
        transformer.to(self.device)
        vae.to(self.device)

        if s.fp8:
            accel.apply_fp8(transformer)
        accel.apply_first_block_cache(transformer, s.first_block_cache)
        if s.compile:
            accel.apply_compile(transformer)
        if s.trt_engine_path:
            from ..trt import TRTTransformer

            self.pipe.transformer = TRTTransformer(s.trt_engine_path, transformer.config, self.device)

    @classmethod
    def from_components(cls, transformer, vae, scheduler, settings: Settings, device="cpu", dtype=torch.float32):
        """Build around already-constructed modules (tests, custom checkpoints)."""
        eng = cls(settings, device=device, dtype=dtype)
        eng._assemble(transformer, vae, scheduler)
        return eng

    def _assemble(self, transformer, vae, scheduler) -> None:
        from diffusers import FluxFillPipeline

        # FluxFillPipeline supplies the latent packing/mask helpers; we never
        # call its __call__, so the text-encoder slots stay empty.
        self.pipe = FluxFillPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            transformer=transformer,
        )

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def generate(self, person, garment, mask, params: GenerateParams) -> Image.Image:
        from diffusers.pipelines.flux.pipeline_flux_fill import calculate_shift, retrieve_timesteps

        pipe = self.pipe
        tr = pipe.transformer
        steps = params.steps or self.default_steps
        guidance_scale = params.guidance if params.guidance is not None else self.default_guidance
        width, height = person.size
        device, dtype = self.device, self.dtype
        generator = torch.Generator(device="cpu").manual_seed(params.seed)

        image = pipe.image_processor.preprocess(person, height=height, width=width)
        cond = pipe.image_processor.preprocess(garment, height=height, width=width)
        m = pipe.mask_processor.preprocess(mask, height=height, width=width)

        # CatVTON: [masked person | garment] side by side, garment half unmasked.
        masked = torch.cat((image * (1 - m), cond), dim=-1).to(device=device, dtype=dtype)
        m2 = torch.cat((m, torch.zeros_like(m)), dim=-1).to(device=device, dtype=dtype)

        nc = pipe.vae.config.latent_channels
        vsf = pipe.vae_scale_factor
        lh, lw = 2 * (height // (vsf * 2)), 2 * (width * 2 // (vsf * 2))
        noise = torch.randn((1, nc, lh, lw), generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
        latents = pipe._pack_latents(noise, 1, nc, lh, lw)
        img_ids = pipe._prepare_latent_image_ids(1, lh // 2, lw // 2, device, dtype)
        mask_lat, masked_lat = pipe.prepare_mask_latents(
            m2, masked, 1, nc, 1, height, width * 2, dtype, device, generator
        )
        cond_lat = torch.cat((masked_lat, mask_lat), dim=-1)

        sigmas = np.linspace(1.0, 1 / steps, steps)
        sched = pipe.scheduler
        mu = calculate_shift(
            latents.shape[1],
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(sched, steps, device, sigmas=sigmas, mu=mu)

        cfg = tr.config
        guidance = (
            torch.full([1], guidance_scale, device=device, dtype=torch.float32) if cfg.guidance_embeds else None
        )
        pooled = torch.zeros(1, cfg.pooled_projection_dim, device=device, dtype=dtype)
        txt = torch.zeros(1, 0, cfg.joint_attention_dim, device=device, dtype=dtype)
        txt_ids = torch.zeros(0, 3, device=device, dtype=img_ids.dtype)

        try:
            for t in timesteps:
                with _cache_ctx(tr):
                    noise = tr(
                        hidden_states=torch.cat((latents, cond_lat), dim=2),
                        timestep=(t.expand(1) / 1000).to(dtype),
                        guidance=guidance,
                        pooled_projections=pooled,
                        encoder_hidden_states=txt,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]
                latents = sched.step(noise, t, latents, return_dict=False)[0]
        finally:
            if hasattr(tr, "_reset_stateful_cache"):
                tr._reset_stateful_cache()

        latents = pipe._unpack_latents(latents, height, width * 2, vsf)
        latents = latents.split(latents.shape[-1] // 2, dim=-1)[0]  # keep the person half
        latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        out = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
        return pipe.image_processor.postprocess(out, output_type="pil")[0]


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _cache_ctx(module):
    # Cache hooks (FBCache) need a named context; plain modules don't.
    if getattr(module, "is_cache_enabled", False):
        return module.cache_context("cond")
    return _NullCtx()
