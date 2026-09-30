"""Fetch dev/test assets: MediaPipe models (Apache-2.0) + CatVTON demo photos.

The demo photos belong to the CatVTON project (CC BY-NC-SA 4.0): they are
downloaded for local testing only and are never committed or shipped.

    python scripts/fetch_assets.py            # -> .assets/
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".assets"
MP = "https://storage.googleapis.com/mediapipe-models"
DEMO = "https://huggingface.co/spaces/zhengchong/CatVTON/resolve/main/resource/demo/example"

FILES = {
    "models/pose_landmarker_full.task": f"{MP}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "models/selfie_multiclass_256x256.tflite": f"{MP}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
    "person/049713_0.jpg": f"{DEMO}/person/women/049713_0.jpg",
    "person/1-model_3.png": f"{DEMO}/person/women/1-model_3.png",
    "person/model_5.png": f"{DEMO}/person/men/model_5.png",
    "person/Yifeng_0.png": f"{DEMO}/person/men/Yifeng_0.png",
    "garment/upper_21514384_52353349_1000.jpg": f"{DEMO}/condition/upper/21514384_52353349_1000.jpg",
    "garment/upper_23255574_53383833_1000.jpg": f"{DEMO}/condition/upper/23255574_53383833_1000.jpg",
    "garment/overall_21744571_51588794_1000.jpg": f"{DEMO}/condition/overall/21744571_51588794_1000.jpg",
    "garment/overall_22153949_52376342_1000.jpg": f"{DEMO}/condition/overall/22153949_52376342_1000.jpg",
}


def main() -> None:
    for rel, url in FILES.items():
        dst = ROOT / rel
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fitcheck-vton"})) as r:
            dst.write_bytes(r.read())
        print("fetched", rel)
    print("assets ready in", ROOT)


if __name__ == "__main__":
    main()
