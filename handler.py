"""RunPod Serverless entrypoint.

Models load at import time (outside the handler) and a warm-up inference runs
before `runpod.serverless.start`, so the first real request never pays for
weight loading, CUDA context creation or kernel autotuning, and FlashBoot
snapshots an already-hot worker.

Local run (CPU, mock engine):
    VTON_ENGINE=mock MEDIAPIPE_DIR=.assets/models python handler.py --test_input "$(cat test_input.json)"
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import runpod  # noqa: E402

from vton.config import get_settings  # noqa: E402
from vton.engines import create_engine  # noqa: E402
from vton.imageio import InputError  # noqa: E402
from vton.masking import MaskingError, MediaPipeMasker  # noqa: E402
from vton.service import TryOnService  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("handler")

_boot = time.perf_counter()
settings = get_settings()
engine = create_engine(settings)
engine.load()
service = TryOnService(settings, engine, MediaPipeMasker(settings.mediapipe_dir))
service.warmup()
BOOT_MS = round((time.perf_counter() - _boot) * 1000)
log.info("worker ready: engine=%s boot=%dms", settings.engine, BOOT_MS)


def _maybe_upload(job_id: str, out: dict) -> dict:
    """With BUCKET_ENDPOINT_URL set, return a URL instead of inline base64."""
    if not os.environ.get("BUCKET_ENDPOINT_URL"):
        return out
    import base64
    import tempfile

    from runpod.serverless.utils import rp_upload

    header, b64 = out["image"].split(",", 1)
    ext = header.split("/")[1].split(";")[0]
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(base64.b64decode(b64))
        path = f.name
    try:
        out["image"] = rp_upload.upload_image(job_id, path)
    finally:
        os.unlink(path)
    return out


def handler(job: dict) -> dict:
    try:
        out = service.run(job.get("input") or {})
        out["boot_ms"] = BOOT_MS
        return _maybe_upload(job["id"], out)
    except (InputError, MaskingError) as e:
        # Client-fixable: stable machine-readable code for the app to map to UX copy.
        return {"error": f"{e.code}: {e.message}"}
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job.get("id"))
        oom = "out of memory" in str(e).lower()
        # An OOM can leave the CUDA context fragmented; recycle this worker.
        return {"error": f"internal_error: {type(e).__name__}", "refresh_worker": oom}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
