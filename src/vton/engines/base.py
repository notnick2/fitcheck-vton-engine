"""Engine contract shared by every diffusion backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class GenerateParams:
    category: str  # upper | lower | overall
    steps: int | None = None  # None -> engine default
    guidance: float | None = None
    seed: int = 0


class TryOnEngine(ABC):
    name: str = "base"
    # Whether every weight this engine loads is licensed for a paid product.
    commercial_ok: bool = False
    # Whether the engine needs an inpainting mask as model input (vs. mask-free
    # editing where the mask is only used for the identity-lock composite).
    needs_mask: bool = True

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(
        self, person: Image.Image, garment: Image.Image, mask: Image.Image, params: GenerateParams
    ) -> Image.Image:
        """All images at working resolution; returns working-resolution RGB."""

    def warmup(self, size: tuple[int, int]) -> None:
        blank = Image.new("RGB", size, (200, 200, 200))
        mask = Image.new("L", size, 0)
        mask.paste(255, (size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 2))
        self.generate(blank, blank, mask, GenerateParams(category="upper", steps=2, seed=0))

    def describe(self) -> dict:
        return {"engine": self.name, "commercial_ok": self.commercial_ok}
