"""Identity lock: only pixels inside the try-on mask may change.

Diffusion models re-render the whole canvas, which subtly alters faces, hair,
tattoos and backgrounds (and drops everything to the 768x1024 working
resolution). Pasting the generated garment back into the *original* photo
through a feathered mask keeps the user looking exactly like themselves, at
their camera's native resolution.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .preprocess import CropBox


def feather(mask: Image.Image, radius: float) -> Image.Image:
    """Grow the mask slightly, then blur, so seams land on generated pixels.

    OpenCV's separable dilate/blur: ~20x faster than PIL's MaxFilter at 12 MP.
    """
    m = mask.convert("L")
    if radius <= 0:
        return m
    grow = max(3, int(radius) | 1)
    arr = cv2.dilate(np.asarray(m), cv2.getStructuringElement(cv2.MORPH_RECT, (grow, grow)))
    k = int(radius * 3) | 1
    return Image.fromarray(cv2.GaussianBlur(arr, (k, k), radius))


def paste_back(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    box: CropBox,
    feather_frac: float = 0.006,
) -> Image.Image:
    """Composite `generated` (working-res crop) into `original` (full-res).

    `mask` is at working resolution, white = region the model may change.
    """
    cw, ch = box.size
    gen = generated.resize((cw, ch), Image.LANCZOS)
    radius = max(1.0, feather_frac * ch)
    alpha = feather(mask.resize((cw, ch), Image.BILINEAR), radius)

    # Intersection of the crop box with the real image (crop may be padded).
    ix0, iy0 = max(box.x0, 0), max(box.y0, 0)
    ix1, iy1 = min(box.x1, original.width), min(box.y1, original.height)
    if ix1 <= ix0 or iy1 <= iy0:
        return original.copy()

    local = (ix0 - box.x0, iy0 - box.y0, ix1 - box.x0, iy1 - box.y0)
    out = original.copy()
    region = Image.composite(gen.crop(local), original.crop((ix0, iy0, ix1, iy1)), alpha.crop(local))
    out.paste(region, (ix0, iy0))
    return out


def outside_mask_delta(a: Image.Image, b: Image.Image, mask: Image.Image) -> float:
    """Mean abs pixel difference outside `mask` (0 = perfect identity lock)."""
    keep = np.asarray(mask.convert("L").resize(a.size)) < 8
    if not keep.any():
        return 0.0
    da = np.asarray(a.convert("RGB"), dtype=np.int16)
    db = np.asarray(b.convert("RGB").resize(a.size), dtype=np.int16)
    return float(np.abs(da - db)[keep].mean())
