"""Engine registry. Heavy imports stay lazy so the mock engine runs torch-free."""

from __future__ import annotations

from ..config import Settings
from .base import GenerateParams, TryOnEngine

ENGINES = ("qwen_edit", "catvton_flux", "mock")


def create_engine(settings: Settings) -> TryOnEngine:
    name = settings.engine
    if name == "mock":
        from .mock import MockEngine

        return MockEngine()
    if name == "catvton_flux":
        from .catvton_flux import CatVTONFluxEngine

        return CatVTONFluxEngine(settings)
    if name == "qwen_edit":
        from .qwen_edit import QwenEditEngine

        return QwenEditEngine(settings)
    raise ValueError(f"unknown VTON_ENGINE '{name}', expected one of {ENGINES}")


__all__ = ["ENGINES", "GenerateParams", "TryOnEngine", "create_engine"]
