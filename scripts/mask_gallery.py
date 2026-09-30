"""Render a contact sheet of masks for every person x category (CPU only).

    python scripts/mask_gallery.py --people .assets/person --out out/mask_gallery.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vton.masking import CATEGORIES, MaskingError, MediaPipeMasker, overlay  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", default=".assets/person")
    ap.add_argument("--models", default=".assets/models")
    ap.add_argument("--out", default="out/mask_gallery.jpg")
    ap.add_argument("--thumb", type=int, default=300)
    args = ap.parse_args()

    masker = MediaPipeMasker(args.models)
    people = sorted(p for p in Path(args.people).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    tw, th = args.thumb, args.thumb * 4 // 3
    sheet = Image.new("RGB", (tw * (len(CATEGORIES) + 1), (th + 22) * len(people)), "white")
    draw = ImageDraw.Draw(sheet)

    for row, path in enumerate(people):
        img = Image.open(path).convert("RGB")
        t = time.perf_counter()
        analysis = masker.analyze(img)
        t_an = (time.perf_counter() - t) * 1000
        y = row * (th + 22)
        sheet.paste(img.resize((tw, th)), (0, y))
        draw.text((4, y + th + 4), f"{path.name[:22]}  analyze {t_an:.0f}ms", fill="black")
        for col, cat in enumerate(CATEGORIES, start=1):
            try:
                t = time.perf_counter()
                m = masker.mask(img, cat, analysis)
                label = f"{cat}  {(time.perf_counter() - t) * 1000:.0f}ms"
                tile = overlay(img, m)
            except MaskingError as e:
                label, tile = f"{cat}: {e.code}", img.convert("L").convert("RGB")
            sheet.paste(tile.resize((tw, th)), (col * tw, y))
            draw.text((col * tw + 4, y + th + 4), label, fill="black")
        print(f"{path.name}: analyze {t_an:.0f} ms, people={analysis.num_people}, warnings={analysis.warnings}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=88)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
