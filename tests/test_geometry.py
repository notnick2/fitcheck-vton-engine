import base64
import io

import numpy as np
import pytest
from PIL import Image

from vton import compose, imageio, preprocess
from vton.config import Settings


def test_plan_crop_centre_when_no_subject():
    box = preprocess.plan_crop((1000, 1000), None, aspect=0.75)
    assert box.size == (750, 1000)
    assert box.x0 == 125 and box.y0 == 0


def test_plan_crop_tightens_on_small_subject():
    # 4000x3000 landscape photo, person occupies a thin strip in the middle.
    box = preprocess.plan_crop((4000, 3000), (1900, 1000, 2100, 2000), aspect=0.75)
    w, h = box.size
    assert abs(w / h - 0.75) < 0.01
    assert h < 1300  # far tighter than the 3000px full-height crop
    assert box.x0 <= 1900 and box.x1 >= 2100 and box.y0 <= 1000 and box.y1 >= 2000


def test_plan_crop_pads_when_subject_exceeds_frame():
    box = preprocess.plan_crop((600, 400), (0, 0, 600, 400), aspect=0.75)
    assert box.y0 < 0 and box.y1 > 400  # vertical padding needed for 3:4


def test_crop_to_pads_outside_image():
    img = Image.new("RGB", (10, 10), (255, 0, 0))
    out = preprocess.crop_to(img, preprocess.CropBox(-5, 0, 15, 10), fill=(0, 255, 0))
    assert out.size == (20, 10)
    assert out.getpixel((0, 5)) == (0, 255, 0)
    assert out.getpixel((10, 5)) == (255, 0, 0)


def test_fit_garment_trims_studio_background():
    img = Image.new("RGB", (1000, 1334), (255, 255, 255))
    img.paste((20, 40, 160), (450, 600, 550, 750))  # small garment in a big sweep
    out = preprocess.fit_garment(img, (768, 1024))
    arr = np.asarray(out)
    fg = np.abs(arr.astype(int) - 255).max(axis=2) > 18
    # garment went from 13% of the frame width to most of the canvas width
    xs = np.nonzero(fg.any(axis=0))[0]
    assert xs.max() - xs.min() > 0.7 * 768


def test_identity_lock_preserves_outside_mask_exactly():
    rng = np.random.default_rng(0)
    original = Image.fromarray(rng.integers(0, 255, (1600, 1200, 3), dtype=np.uint8))
    box = preprocess.CropBox(100, 100, 1100, 1433)  # 3:4 crop inside the photo
    generated = Image.new("RGB", (768, 1024), (0, 255, 0))
    mask = Image.new("L", (768, 1024), 0)
    mask.paste(255, (300, 300, 500, 600))

    out = compose.paste_back(original, generated, mask, box)
    assert out.size == original.size
    a, b = np.asarray(original), np.asarray(out)
    # Far from the mask -> bit-identical to the user's photo.
    assert np.array_equal(a[:80], b[:80])
    assert np.array_equal(a[:, 1150:], b[:, 1150:])
    # Mask centre -> generated pixels.
    cy, cx = 100 + int(450 * 1333 / 1024), 100 + int(400 * 1000 / 768)
    assert tuple(b[cy, cx]) == (0, 255, 0)


def test_load_image_handles_data_uri_exif_and_alpha():
    img = Image.new("RGBA", (40, 20), (255, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    out = imageio.load_image(uri, Settings(), "person_image")
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255)

    # EXIF orientation 6 (rotate 90 CW) must be applied.
    jpg = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20), "blue").save(jpg, format="JPEG", exif=exif)
    out = imageio.load_image(base64.b64encode(jpg.getvalue()).decode(), Settings(), "person_image")
    assert out.size == (20, 40)


@pytest.mark.parametrize("bad", ["", "not base64!!", "aGVsbG8="])
def test_load_image_rejects_garbage(bad):
    with pytest.raises(imageio.InputError):
        imageio.load_image(bad, Settings(), "person_image")
