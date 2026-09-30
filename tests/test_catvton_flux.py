"""CatVTON-FLUX loop on a tiny random model: shapes, determinism, text-free path."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from vton.config import Settings
from vton.engines.base import GenerateParams
from vton.engines.catvton_flux import CatVTONFluxEngine

from tiny_flux import tiny_flux_components

W, H = 64, 96


def _inputs():
    person = Image.new("RGB", (W, H), (180, 150, 130))
    garment = Image.new("RGB", (W, H), (30, 60, 200))
    mask = Image.new("L", (W, H), 0)
    mask.paste(255, (16, 24, 48, 72))
    return person, garment, mask


@pytest.fixture(scope="module")
def engine():
    tr, vae, sched = tiny_flux_components()
    return CatVTONFluxEngine.from_components(tr, vae, sched, Settings(engine="catvton_flux"))


def test_generates_working_resolution_image(engine):
    out = engine.generate(*_inputs(), GenerateParams(category="upper", steps=3, seed=1))
    assert isinstance(out, Image.Image)
    assert out.size == (W, H)
    assert out.mode == "RGB"


def test_seed_is_deterministic(engine):
    a = engine.generate(*_inputs(), GenerateParams(category="upper", steps=2, seed=7))
    b = engine.generate(*_inputs(), GenerateParams(category="upper", steps=2, seed=7))
    c = engine.generate(*_inputs(), GenerateParams(category="upper", steps=2, seed=8))
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()


def test_upstream_none_text_path_is_broken_on_current_diffusers():
    """Documents *why* we pass a zero-length text sequence."""
    tr, _, _ = tiny_flux_components()
    seq = 12
    hidden = torch.randn(1, seq, tr.config.in_channels)
    img_ids = torch.zeros(seq, 3)
    with pytest.raises((TypeError, AttributeError)):
        tr(
            hidden_states=hidden,
            timestep=torch.tensor([0.5]),
            guidance=torch.tensor([30.0]),
            pooled_projections=torch.zeros(1, tr.config.pooled_projection_dim),
            encoder_hidden_states=None,
            txt_ids=None,
            img_ids=img_ids,
            return_dict=False,
        )


def test_fbcache_path_runs_and_resets(engine):
    from vton import accel

    tr, vae, sched = tiny_flux_components()
    eng = CatVTONFluxEngine.from_components(tr, vae, sched, Settings(engine="catvton_flux"))
    assert accel.apply_first_block_cache(eng.pipe.transformer, 0.5)
    for _ in range(2):  # second call proves state was reset between requests
        out = eng.generate(*_inputs(), GenerateParams(category="upper", steps=4, seed=3))
        assert out.size == (W, H)


def test_lora_fuse_roundtrip(tmp_path):
    """A PEFT LoRA saved in diffusers format fuses and changes the weights."""
    from peft import LoraConfig
    from safetensors.torch import save_file

    from vton import accel

    tr, _, _ = tiny_flux_components()
    donor, _, _ = tiny_flux_components()
    donor.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v"], init_lora_weights=False))
    from peft.utils import get_peft_model_state_dict

    sd = {f"transformer.{k}": v.contiguous() for k, v in get_peft_model_state_dict(donor).items()}
    path = tmp_path / "lora.safetensors"
    save_file(sd, str(path))

    before = tr.transformer_blocks[0].attn.to_q.weight.detach().clone()
    accel.fuse_lora_file(tr, str(path))
    after = tr.transformer_blocks[0].attn.to_q.weight
    assert not torch.equal(before, after)
    # PEFT layers are gone -> plain Linear, ready for FP8 / ONNX export
    assert type(tr.transformer_blocks[0].attn.to_q).__name__ == "Linear"
