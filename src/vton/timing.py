"""Tiny stage timer so every response carries a latency breakdown."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str, sync_cuda: bool = False):
        if sync_cuda:
            _cuda_sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda:
                _cuda_sync()
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1000

    def as_dict(self) -> dict[str, float]:
        out = {k: round(v, 1) for k, v in self.stages.items()}
        out["total"] = round((time.perf_counter() - self._t0) * 1000, 1)
        return out


def _cuda_sync() -> None:
    # Only if an engine already imported torch; never pay a cold import here.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
