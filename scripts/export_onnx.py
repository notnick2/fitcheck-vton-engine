"""Export the CatVTON-FLUX transformer to a static-shape, text-free ONNX graph.

    # real model (run on a GPU pod; needs HF_TOKEN for the gated FLUX repo)
    python scripts/export_onnx.py --out /workspace/onnx/catvton_flux.onnx --width 768 --height 1024

    # then on the SAME GPU type as the endpoint:
    trtexec --onnx=/workspace/onnx/catvton_flux.onnx --bf16 --saveEngine=/workspace/catvton_flux.plan
    # FP8 (Ada/Hopper): quantize with NVIDIA ModelOpt first (see docs/OPTIMIZATION.md),
    # then trtexec --fp8 --bf16 ...

    # CPU self-check on a tiny random model (what CI runs):
    python scripts/export_onnx.py --tiny --out out/tiny.onnx --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import torch  # noqa: E402

from vton.trt import INPUT_NAMES, OUTPUT_NAME, TextFreeFlux, example_inputs  # noqa: E402


def load_transformer(args):
    if args.tiny:
        from tiny_flux import tiny_flux_components

        return tiny_flux_components()[0], torch.float32
    from diffusers import FluxTransformer2DModel
    from huggingface_hub import hf_hub_download

    from vton import accel
    from vton.config import Settings

    s = Settings()
    dtype = torch.bfloat16
    tr = FluxTransformer2DModel.from_pretrained(s.flux_fill_repo, subfolder="transformer", torch_dtype=dtype)
    accel.fuse_lora_file(tr, hf_hub_download(s.catvton_repo, "flux-lora/pytorch_lora_weights.safetensors"))
    return tr.to("cuda" if torch.cuda.is_available() else "cpu"), dtype


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--tiny", action="store_true", help="random tiny model (CPU self-test)")
    ap.add_argument("--verify", action="store_true", help="compare onnxruntime vs torch outputs")
    args = ap.parse_args()
    if args.tiny:
        args.width, args.height = 64, 96

    tr, dtype = load_transformer(args)
    model = TextFreeFlux(tr).eval()
    device = next(tr.parameters()).device
    inputs = tuple(t.to(device) for t in example_inputs(tr, args.width, args.height, dtype=dtype))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        ref = model(*inputs)
    print(f"tokens={inputs[0].shape[1]} in_ch={inputs[0].shape[2]} out={tuple(ref.shape)}")

    torch.onnx.export(
        model,
        inputs,
        str(out_path),
        input_names=list(INPUT_NAMES),
        output_names=[OUTPUT_NAME],
        dynamo=True,
        opset_version=18,
        optimize=True,
        external_data=not args.tiny,  # >2 GB protobuf limit for the real 12B model
    )
    print("exported", out_path)

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        feeds = {n: t.float().cpu().numpy() for n, t in zip(INPUT_NAMES, inputs)}
        feeds = {i.name: feeds[i.name] for i in sess.get_inputs()}  # constant-folded inputs may vanish
        got = sess.run([OUTPUT_NAME], feeds)[0]
        err = float(np.abs(got - ref.float().cpu().numpy()).max())
        print(f"onnxruntime vs torch max abs err = {err:.2e}  inputs={[i.name for i in sess.get_inputs()]}")
        if err > 1e-3:
            raise SystemExit("ONNX output mismatch")


if __name__ == "__main__":
    main()
