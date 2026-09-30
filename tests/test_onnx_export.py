"""The text-free FLUX graph exports to ONNX and matches PyTorch (TensorRT prerequisite)."""

import numpy as np
import pytest
import torch

from vton.trt import INPUT_NAMES, OUTPUT_NAME, TextFreeFlux, example_inputs

from tiny_flux import tiny_flux_components

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def test_text_free_flux_onnx_roundtrip(tmp_path):
    tr = tiny_flux_components()[0]
    model = TextFreeFlux(tr).eval()
    inputs = example_inputs(tr, 64, 96)
    with torch.inference_mode():
        ref = model(*inputs).numpy()

    path = tmp_path / "flux.onnx"
    torch.onnx.export(
        model, inputs, str(path), input_names=list(INPUT_NAMES), output_names=[OUTPUT_NAME],
        dynamo=True, opset_version=18, optimize=True, verbose=False,
    )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == list(INPUT_NAMES)  # no text inputs survive
    got = sess.run([OUTPUT_NAME], {n: t.numpy() for n, t in zip(INPUT_NAMES, inputs)})[0]
    assert np.abs(got - ref).max() < 1e-3
