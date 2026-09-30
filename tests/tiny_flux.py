"""Randomly initialised, structurally faithful FLUX-Fill components (CPU-sized)."""

from __future__ import annotations

import torch


def tiny_flux_components(seed: int = 0):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel

    torch.manual_seed(seed)
    latent_ch = 16
    # Fill input = packed noise (16*4) + packed masked image (16*4) + packed 8x8 mask (64*4)
    in_ch = latent_ch * 4 * 2 + 64 * 4
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=in_ch,
        out_channels=latent_ch * 4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=24,
        guidance_embeds=True,
        axes_dims_rope=(4, 6, 6),
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=latent_ch,
        norm_num_groups=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
        shift_factor=0.0609,
        scaling_factor=1.5035,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256, max_image_seq_len=4096, base_shift=0.5, max_shift=1.15, use_dynamic_shifting=True
    )
    return transformer.eval(), vae.eval(), scheduler
