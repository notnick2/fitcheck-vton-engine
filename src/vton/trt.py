"""TensorRT path for the CatVTON-FLUX transformer.

Pipeline:  fuse LoRA -> TextFreeFlux wrapper -> ONNX (static shapes)
           -> [optional ModelOpt FP8 PTQ] -> trtexec -> .plan -> TRTTransformer

CatVTON-FLUX uses no prompt, so the wrapper bakes the empty text stream and
zero pooled vector into the graph. The exported model has four inputs, all
static for a fixed canvas (768x1024 -> 6144 image tokens), which is ideal
for TensorRT: no optimisation profiles, one tactic set, no zero-length tensors.
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_NAMES = ("hidden_states", "timestep", "guidance", "img_ids")
OUTPUT_NAME = "noise_pred"


class TextFreeFlux(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.tr = transformer
        cfg = transformer.config
        self.joint_dim = cfg.joint_attention_dim
        self.pooled_dim = cfg.pooled_projection_dim

    def forward(self, hidden_states, timestep, guidance, img_ids):
        b = hidden_states.shape[0]
        return self.tr(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=hidden_states.new_zeros(b, self.pooled_dim),
            encoder_hidden_states=hidden_states.new_zeros(b, 0, self.joint_dim),
            txt_ids=img_ids.new_zeros(0, 3),
            img_ids=img_ids,
            return_dict=False,
        )[0]


def example_inputs(transformer: nn.Module, width: int, height: int, vae_scale: int = 8, dtype=torch.float32):
    """Inputs matching CatVTON's [person | garment] canvas at (width, height)."""
    lh, lw = 2 * (height // (vae_scale * 2)), 2 * (width * 2 // (vae_scale * 2))
    seq = (lh // 2) * (lw // 2)
    cfg = transformer.config
    ids = torch.zeros(lh // 2, lw // 2, 3)
    ids[..., 1] += torch.arange(lh // 2)[:, None]
    ids[..., 2] += torch.arange(lw // 2)[None, :]
    return (
        torch.randn(1, seq, cfg.in_channels, dtype=dtype),
        torch.full((1,), 0.5, dtype=dtype),
        torch.full((1,), 30.0, dtype=torch.float32),
        ids.reshape(-1, 3).to(dtype),
    )


class TRTTransformer:
    """Drop-in for FluxTransformer2DModel.__call__ inside CatVTONFluxEngine.

    Requires `tensorrt>=10` and an engine built for *this* GPU architecture
    (build on the same RunPod GPU type the endpoint uses).
    """

    is_cache_enabled = False

    def __init__(self, engine_path: str, config, device: torch.device) -> None:
        import tensorrt as trt

        self.config = config
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)
        out_shape = tuple(self.engine.get_tensor_shape(OUTPUT_NAME))
        out_dtype = _torch_dtype(self.engine.get_tensor_dtype(OUTPUT_NAME))
        self._out = torch.empty(out_shape, dtype=out_dtype, device=device)
        self.dtype = out_dtype

    def __call__(self, *, hidden_states, timestep, guidance, img_ids, **_ignored):
        feeds = {"hidden_states": hidden_states, "timestep": timestep, "guidance": guidance, "img_ids": img_ids}
        keep = []
        for name, t in feeds.items():
            want = _torch_dtype(self.engine.get_tensor_dtype(name))
            t = t.to(device=self.device, dtype=want).contiguous()
            keep.append(t)
            self.context.set_tensor_address(name, t.data_ptr())
        self.context.set_tensor_address(OUTPUT_NAME, self._out.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        return (self._out.clone(),)


def _torch_dtype(trt_dtype):
    import tensorrt as trt

    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
    }[trt_dtype]
