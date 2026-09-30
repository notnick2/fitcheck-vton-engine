"""Geometry: getting arbitrary phone photos into the model's 3:4 canvas and back.

The crop box is kept in *original* pixel coordinates so compose.paste_back can
put the generated garment back into the untouched full-resolution photo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropBox:
    """Region of the source image (may extend past its borders -> padded)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def size(self) -> tuple[int, int]:
        return self.x1 - self.x0, self.y1 - self.y0


def plan_crop(
    image_size: tuple[int, int],
    subject_bbox: tuple[float, float, float, float] | None,
    aspect: float,
    margin: float = 0.08,
) -> CropBox:
    """Smallest `aspect` (w/h) box that holds the subject plus a margin.

    With no subject box we fall back to the largest centred crop, i.e. the
    behaviour of CatVTON's resize_and_crop.
    """
    W, H = image_size
    if subject_bbox is None:
        if W / H > aspect:
            cw, ch = H * aspect, float(H)
        else:
            cw, ch = float(W), W / aspect
        cx, cy = W / 2, H / 2
    else:
        x0, y0, x1, y1 = subject_bbox
        pad = margin * max(y1 - y0, x1 - x0)
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        bw, bh = x1 - x0, y1 - y0
        ch = max(bh, bw / aspect)
        cw = ch * aspect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    # Slide the box back inside the image where possible; only pad when the
    # subject genuinely needs more room than the photo offers.
    left = cx - cw / 2
    top = cy - ch / 2
    left = _slide(left, cw, W)
    top = _slide(top, ch, H)
    return CropBox(round(left), round(top), round(left + cw), round(top + ch))


def _slide(start: float, length: float, limit: int) -> float:
    if length >= limit:
        return (limit - length) / 2  # centre, pad both sides
    return min(max(start, 0.0), limit - length)


def crop_to(img: Image.Image, box: CropBox, fill=(255, 255, 255)) -> Image.Image:
    """Crop that pads with `fill` where the box leaves the image."""
    canvas = Image.new("RGB", box.size, fill)
    ix0, iy0 = max(box.x0, 0), max(box.y0, 0)
    ix1, iy1 = min(box.x1, img.width), min(box.y1, img.height)
    if ix1 > ix0 and iy1 > iy0:
        canvas.paste(img.crop((ix0, iy0, ix1, iy1)), (ix0 - box.x0, iy0 - box.y0))
    return canvas


def fit_person(
    img: Image.Image,
    size: tuple[int, int],
    subject_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[Image.Image, CropBox]:
    w, h = size
    box = plan_crop(img.size, subject_bbox, aspect=w / h)
    return crop_to(img, box).resize(size, Image.LANCZOS), box


def fit_garment(img: Image.Image, size: tuple[int, int], margin: float = 0.06, tol: int = 18) -> Image.Image:
    """Trim flat studio background around a garment, then pad to the canvas.

    Product shots are usually a small garment on a big white sweep; trimming
    first gives the model many more pixels of fabric/print detail.
    """
    # Find the garment on a thumbnail; the bbox only needs ~1% precision.
    s = min(1.0, 256 / max(img.size))
    thumb = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR) if s < 1 else img
    arr = np.asarray(thumb, dtype=np.int16)
    corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    bg = np.median(corners, axis=0)
    fg = np.abs(arr - bg).max(axis=2) > tol
    ys, xs = np.nonzero(fg)
    bg_rgb = tuple(int(c) for c in bg)

    if len(xs) < 0.01 * fg.size:  # nothing to trim against; keep whole frame
        box = (0, 0, img.width, img.height)
    else:
        box = (int(xs.min() / s), int(ys.min() / s), int((xs.max() + 1) / s), int((ys.max() + 1) / s))

    x0, y0, x1, y1 = box
    pad = int(margin * max(x1 - x0, y1 - y0))
    cropped = crop_to(img, CropBox(x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=bg_rgb)

    w, h = size
    scale = min(w / cropped.width, h / cropped.height)
    resized = cropped.resize((max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", size, bg_rgb)
    canvas.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2))
    return canvas
