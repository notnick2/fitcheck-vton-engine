"""End-to-end service path on real photos with the mock engine + real masker."""

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vton.config import Settings
from vton.engines.mock import MockEngine
from vton.imageio import InputError
from vton.masking import MaskingError, MediaPipeMasker
from vton.service import TryOnService

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / ".assets"
pytestmark = pytest.mark.skipif(not (ASSETS / "models").is_dir(), reason="run scripts/fetch_assets first")


def b64(path_or_img) -> str:
    if isinstance(path_or_img, Image.Image):
        buf = io.BytesIO()
        path_or_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    return base64.b64encode(Path(path_or_img).read_bytes()).decode()


def decode(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")


@pytest.fixture(scope="module")
def service():
    s = Settings(engine="mock", mediapipe_dir=str(ASSETS / "models"), warmup=False)
    masker = MediaPipeMasker(s.mediapipe_dir)
    yield TryOnService(s, MockEngine(), masker)
    masker.close()


@pytest.fixture(scope="module")
def person_path():
    return ASSETS / "person" / "049713_0.jpg"


@pytest.fixture(scope="module")
def garment_path():
    return next((ASSETS / "garment").glob("upper_*"))


def test_full_request_returns_native_resolution_and_timings(service, person_path, garment_path):
    out = service.run(
        {"person_image": b64(person_path), "garment_image": b64(garment_path), "category": "upper", "seed": 5,
         "return_mask": True}
    )
    person = Image.open(person_path)
    assert (out["width"], out["height"]) == person.size
    assert out["image"].startswith("data:image/webp;base64,")
    assert out["mask"].startswith("data:image/webp;base64,")
    assert out["seed"] == 5 and out["engine"] == "mock"
    for stage in ("decode", "analyze", "mask", "preprocess", "diffusion", "compose", "encode", "total"):
        assert stage in out["timings_ms"]


def test_identity_lock_keeps_face_region(service, person_path, garment_path):
    out = service.run(
        {"person_image": b64(person_path), "garment_image": b64(garment_path), "category": "upper",
         "output_format": "png"}
    )
    import cv2

    from vton.masking import FACE_SKIN, HAIR

    person = Image.open(person_path).convert("RGB")
    src = np.asarray(person, dtype=np.int16)
    res = np.asarray(decode(out["image"]), dtype=np.int16)
    # Pixels the segmenter calls face/hair (shrunk past the feather) must be untouched.
    a = service.masker.analyze(person)
    face = np.isin(a.classes, (FACE_SKIN, HAIR)).astype(np.uint8)
    face = cv2.resize(face, person.size, interpolation=cv2.INTER_NEAREST)
    face = cv2.erode(face, np.ones((25, 25), np.uint8)).astype(bool)
    assert face.sum() > 1000
    assert np.abs(src - res).max(axis=2)[face].max() == 0
    torso = (slice(int(0.25 * src.shape[0]), int(0.35 * src.shape[0])), slice(src.shape[1] // 2 - 20, src.shape[1] // 2 + 20))
    assert np.abs(src[torso] - res[torso]).mean() > 10  # garment region actually changed


def test_blank_photo_rejected_before_gpu(service, garment_path):
    blank = Image.new("RGB", (768, 1024), (240, 240, 240))
    with pytest.raises(MaskingError) as e:
        service.run({"person_image": b64(blank), "garment_image": b64(garment_path)})
    assert e.value.code == "no_person_detected"


def test_custom_mask_is_honoured(service, person_path, garment_path):
    person = Image.open(person_path)
    mask = Image.new("L", person.size, 0)
    mask.paste(255, (0, person.height - 100, person.width, person.height))
    out = service.run(
        {"person_image": b64(person_path), "garment_image": b64(garment_path), "mask_image": b64(mask),
         "output_format": "png"}
    )
    src = np.asarray(person.convert("RGB"), dtype=np.int16)
    res = np.asarray(decode(out["image"]), dtype=np.int16)
    assert np.abs(src[: person.height // 2] - res[: person.height // 2]).max() == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"garment_image": "x"},
        {"person_image": "x", "garment_image": "x", "category": "hat"},
        {"person_image": "x", "garment_image": "x", "steps": 500},
        {"person_image": "x", "garment_image": "x", "surprise": 1},
    ],
)
def test_bad_requests_rejected(service, payload):
    with pytest.raises(InputError):
        service.run(payload)
