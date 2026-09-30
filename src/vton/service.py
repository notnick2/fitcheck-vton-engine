"""Request -> response orchestration, independent of RunPod.

    decode -> analyse body (QC, pre-GPU) -> smart crop -> mask -> diffuse
           -> identity-lock paste-back at native resolution -> encode

Anything that can reject a request runs before the GPU is touched, so bad
uploads cost milliseconds of CPU instead of seconds of GPU.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from PIL import Image

from . import compose, imageio, preprocess
from .config import Settings
from .engines import GenerateParams, TryOnEngine
from .imageio import InputError
from .masking import CATEGORIES, MaskingError, overlay
from .timing import StageTimer

log = logging.getLogger(__name__)

OUTPUT_FORMATS = ("webp", "jpeg", "png")


@dataclass
class TryOnRequest:
    person_image: str
    garment_image: str
    category: str = "upper"
    mask_image: str | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    identity_lock: bool = True
    output_format: str = "webp"
    output_quality: int = 90
    max_output_side: int = 1536
    return_mask: bool = False

    @classmethod
    def parse(cls, payload: dict) -> "TryOnRequest":
        if not isinstance(payload, dict):
            raise InputError("invalid_input", "input must be a JSON object")
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise InputError("invalid_input", f"unknown fields: {sorted(unknown)}")
        for req in ("person_image", "garment_image"):
            if not payload.get(req):
                raise InputError("invalid_input", f"'{req}' is required")
        r = cls(**payload)
        if r.category not in CATEGORIES:
            raise InputError("invalid_input", f"category must be one of {CATEGORIES}")
        if r.steps is not None and not (1 <= int(r.steps) <= 60):
            raise InputError("invalid_input", "steps must be in [1, 60]")
        if r.guidance is not None and not (0 <= float(r.guidance) <= 50):
            raise InputError("invalid_input", "guidance must be in [0, 50]")
        if r.seed is not None and not (0 <= int(r.seed) < 2**63):
            raise InputError("invalid_input", "seed must be a non-negative int64")
        if r.output_format.lower() not in OUTPUT_FORMATS and r.output_format.lower() != "jpg":
            raise InputError("invalid_input", f"output_format must be one of {OUTPUT_FORMATS}")
        if not (1 <= int(r.output_quality) <= 100):
            raise InputError("invalid_input", "output_quality must be in [1, 100]")
        if not (256 <= int(r.max_output_side) <= 4096):
            raise InputError("invalid_input", "max_output_side must be in [256, 4096]")
        return r


class TryOnService:
    def __init__(self, settings: Settings, engine: TryOnEngine, masker) -> None:
        self.settings = settings
        self.engine = engine
        self.masker = masker
        self.size = (settings.width, settings.height)

    def warmup(self) -> None:
        if self.settings.warmup:
            self.engine.warmup(self.size)

    def run(self, payload: dict) -> dict:
        timer = StageTimer()
        req = TryOnRequest.parse(payload)
        seed = req.seed if req.seed is not None else random.randrange(2**31)
        warnings: list[str] = []

        with timer.stage("decode"):
            person = imageio.load_image(req.person_image, self.settings, "person_image")
            garment = imageio.load_image(req.garment_image, self.settings, "garment_image")
            custom_mask = (
                imageio.load_image(req.mask_image, self.settings, "mask_image").convert("L")
                if req.mask_image
                else None
            )

        with timer.stage("analyze"):
            analysis = self.masker.analyze(person)
            warnings += analysis.warnings
            # Scale the subject box from analysis resolution to the original.
            sx = person.width / analysis.size[0]
            sy = person.height / analysis.size[1]
            bbox = analysis.subject_bbox()
            bbox = None if bbox is None else (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)

        with timer.stage("mask"):
            if custom_mask is not None:
                full_mask = custom_mask.resize(person.size, Image.NEAREST)
            else:
                full_mask = self.masker.mask(person, req.category, analysis)
                warnings += [w for w in analysis.warnings if w not in warnings]
            if not full_mask.getbbox():
                raise MaskingError("empty_mask", "nothing to replace was found for this category")

        with timer.stage("preprocess"):
            person_w, box = preprocess.fit_person(person, self.size, bbox)
            mask_w = preprocess.crop_to(full_mask.convert("RGB"), box, fill=(0, 0, 0)).convert("L")
            mask_w = mask_w.resize(self.size, Image.NEAREST)
            garment_w = preprocess.fit_garment(garment, self.size)

        with timer.stage("diffusion", sync_cuda=True):
            generated = self.engine.generate(
                person_w,
                garment_w,
                mask_w,
                GenerateParams(category=req.category, steps=req.steps, guidance=req.guidance, seed=seed),
            )

        with timer.stage("compose"):
            if req.identity_lock:
                result = compose.paste_back(person, generated, mask_w, box)
            else:
                result = generated
            result = _limit_side(result, req.max_output_side)

        with timer.stage("encode"):
            data, mime = imageio.encode_image(result, req.output_format, req.output_quality)
            out = {
                "image": imageio.to_data_uri(data, mime),
                "width": result.width,
                "height": result.height,
                "seed": seed,
                "category": req.category,
                **self.engine.describe(),
            }
            if req.return_mask:
                vis = _limit_side(overlay(person, full_mask), 768)
                mdata, mmime = imageio.encode_image(vis, "webp", 80)
                out["mask"] = imageio.to_data_uri(mdata, mmime)

        out["warnings"] = warnings
        out["timings_ms"] = timer.as_dict()
        return out


def _limit_side(img: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
