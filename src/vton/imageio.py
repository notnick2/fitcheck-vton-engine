"""Input decoding / output encoding for the serving layer.

Mobile uploads are messy: EXIF-rotated JPEGs, HEIC-converted PNGs with alpha,
20MP photos. Everything is normalised here so the model never sees it.
"""

from __future__ import annotations

import base64
import binascii
import io
import urllib.request

from PIL import Image, ImageOps

from .config import Settings


class InputError(ValueError):
    """Client-side problem; surfaced to the caller with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def load_image(ref: str, settings: Settings, field_name: str) -> Image.Image:
    if not isinstance(ref, str) or not ref:
        raise InputError("invalid_input", f"'{field_name}' must be a URL, data URI or base64 string")

    if ref.startswith(("http://", "https://")):
        raw = _fetch(ref, settings, field_name)
    else:
        payload = ref.split(",", 1)[1] if ref.startswith("data:") else ref
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InputError("invalid_input", f"'{field_name}' is not valid base64") from exc

    if len(raw) > settings.max_input_bytes:
        raise InputError("image_too_large", f"'{field_name}' exceeds {settings.max_input_bytes} bytes")

    try:
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > settings.max_input_pixels:
            raise InputError("image_too_large", f"'{field_name}' exceeds {settings.max_input_pixels} pixels")
        img.load()
    except InputError:
        raise
    except Exception as exc:  # PIL raises a zoo of exception types
        raise InputError("invalid_image", f"'{field_name}' could not be decoded") from exc

    return normalize(img)


def normalize(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation and flatten alpha onto white."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def _fetch(url: str, settings: Settings, field_name: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fitcheck-vton/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=settings.fetch_timeout_s) as resp:
            raw = resp.read(settings.max_input_bytes + 1)
    except Exception as exc:
        raise InputError("fetch_failed", f"could not download '{field_name}': {exc}") from exc
    return raw


def encode_image(img: Image.Image, fmt: str = "webp", quality: int = 90) -> tuple[bytes, str]:
    fmt = fmt.lower()
    buf = io.BytesIO()
    if fmt == "webp":
        img.save(buf, format="WEBP", quality=quality, method=4)
        mime = "image/webp"
    elif fmt in ("jpg", "jpeg"):
        img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        mime = "image/jpeg"
    elif fmt == "png":
        img.save(buf, format="PNG", optimize=False)
        mime = "image/png"
    else:
        raise InputError("invalid_input", f"unsupported output format '{fmt}'")
    return buf.getvalue(), mime


def to_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"
