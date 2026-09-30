"""CPU stand-in engine: exercises the whole serving path without a GPU.

It paints the garment's dominant colour (with the person's shading) into the
mask, which is enough to eyeball masks/compositing and to run CI and the
container smoke test.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import GenerateParams, TryOnEngine


class MockEngine(TryOnEngine):
    name = "mock"
    commercial_ok = True

    def load(self) -> None:
        pass

    def generate(self, person, garment, mask, params: GenerateParams) -> Image.Image:
        g = np.asarray(garment.convert("RGB"), dtype=np.float32)
        bg = np.median(np.stack([g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]]), axis=0)
        fabric = g[np.abs(g - bg).max(axis=2) > 18]
        colour = np.median(fabric, axis=0) if len(fabric) else np.array([120, 120, 200], np.float32)

        p = np.asarray(person.convert("RGB"), dtype=np.float32)
        shade = p.mean(axis=2, keepdims=True) / 255.0
        painted = np.clip(colour * (0.35 + 0.9 * shade), 0, 255)
        m = (np.asarray(mask.convert("L"), dtype=np.float32) / 255.0)[..., None]
        return Image.fromarray((p * (1 - m) + painted * m).astype(np.uint8))

    def warmup(self, size) -> None:
        pass
