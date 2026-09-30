"""Garment-agnostic try-on masks from commercially licensed models only.

Replaces CatVTON's AutoMasker (DensePose + SCHP). Those checkpoints are
trained on research-only data (DensePose-COCO, LIP/ATR) and drag in a
detectron2 source build. This version uses two Apache-2.0 MediaPipe models
that run on CPU in tens of ms, so masking never competes with diffusion for
the GPU:

  * PoseLandmarker (33 keypoints + visibility) -> limb geometry, QC checks
  * Selfie multiclass segmenter -> hair / body-skin / face-skin / clothes / other

Mask semantics follow CatVTON (upper | lower | overall), with one tweak:
an "overall" try-on may also replace shoes.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CATEGORIES = ("upper", "lower", "overall")

# Selfie multiclass classes
BG, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)

# BlazePose landmark ids
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HAND = (17, 19, 21)
R_HAND = (18, 20, 22)
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

VISIBLE = 0.5


class MaskingError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class BodyAnalysis:
    """Everything the masker learned about one photo (at analysis resolution)."""

    size: tuple[int, int]  # (w, h) of the analysed image
    classes: np.ndarray  # (h, w) uint8 class map
    points: np.ndarray  # (33, 2) float pixel coords
    visibility: np.ndarray  # (33,)
    num_people: int
    warnings: list[str] = field(default_factory=list)

    def vis(self, *ids: int) -> bool:
        w, h = self.size
        for i in ids:
            x, y = self.points[i]
            if self.visibility[i] < VISIBLE or not (-0.02 * w <= x <= 1.02 * w and -0.02 * h <= y <= 1.02 * h):
                return False
        return True

    def subject_bbox(self) -> tuple[float, float, float, float] | None:
        ys, xs = np.nonzero(self.classes != BG)
        if len(xs) == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


class MediaPipeMasker:
    def __init__(self, model_dir: str | Path, analysis_max_side: int = 512) -> None:
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        model_dir = Path(model_dir)
        self._vision = vision
        self._max_side = analysis_max_side
        # Each MediaPipe task graph is not re-entrant, but the two graphs are
        # independent: run them side by side and pay max(seg, pose), not the sum.
        self._seg_lock = threading.Lock()
        self._pose_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mp-pose")
        self._seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(model_dir / "selfie_multiclass_256x256.tflite")),
                output_confidence_masks=False,
                output_category_mask=True,
            )
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(model_dir / "pose_landmarker_full.task")),
                num_poses=3,
            )
        )

    def close(self) -> None:
        """Stop MediaPipe's graph threads (otherwise interpreter exit can hang)."""
        with self._seg_lock, self._pose_lock:
            for task in (self._seg, self._pose):
                if task is not None:
                    task.close()
            self._seg = self._pose = None
        self._pool.shutdown(wait=True)

    # ------------------------------------------------------------------ analysis
    def analyze(self, img: Image.Image) -> BodyAnalysis:
        import mediapipe as mp

        scale = min(1.0, self._max_side / max(img.size))
        work = img if scale == 1.0 else img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
        arr = np.ascontiguousarray(np.asarray(work.convert("RGB")))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

        pose_future = self._pool.submit(self._detect_pose, mp_img)
        with self._seg_lock:
            seg = self._seg.segment(mp_img)
        pose = pose_future.result()

        classes = seg.category_mask.numpy_view().squeeze().astype(np.uint8).copy()
        if not pose.pose_landmarks:
            raise MaskingError("no_person_detected", "could not find a person in the photo")

        w, h = work.size
        # Several detections: keep the biggest person, tell the caller.
        people = [np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in p]) for p in pose.pose_landmarks]
        main = max(people, key=_pose_extent)
        warnings = []
        if len(people) > 1:
            warnings.append("multiple_people")
        return BodyAnalysis(
            size=(w, h),
            classes=classes,
            points=main[:, :2],
            visibility=main[:, 2],
            num_people=len(people),
            warnings=warnings,
        )

    def _detect_pose(self, mp_img):
        with self._pose_lock:
            return self._pose.detect(mp_img)

    # ------------------------------------------------------------------ masks
    def mask(self, img: Image.Image, category: str, analysis: BodyAnalysis | None = None) -> Image.Image:
        """Binary L-mode mask at `img` size; white = region to re-generate."""
        a = analysis or self.analyze(img)
        m = build_mask(a, category)
        return Image.fromarray(m).resize(img.size, Image.BILINEAR).point(lambda v: 255 if v >= 128 else 0)


def _pose_extent(p: np.ndarray) -> float:
    vis = p[p[:, 2] > VISIBLE]
    if len(vis) < 2:
        return 0.0
    return float(np.ptp(vis[:, 0]) * np.ptp(vis[:, 1]))


def check_framing(a: BodyAnalysis, category: str) -> None:
    """Refuse photos that cannot produce a good result *before* paying for GPU."""
    need = {
        "upper": (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP),
        "lower": (L_HIP, R_HIP),
        "overall": (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP),
    }[category]
    if not a.vis(*need):
        region = {"upper": "shoulders to hips", "lower": "hips and legs", "overall": "shoulders to legs"}[category]
        raise MaskingError("person_cut_off", f"the photo must show the {region} for a '{category}' try-on")
    # Shorts/skirts work from mid-thigh photos; long garments will be cropped.
    if category in ("lower", "overall") and not a.vis(L_KNEE, R_KNEE):
        if "legs_partially_visible" not in a.warnings:
            a.warnings.append("legs_partially_visible")


def build_mask(a: BodyAnalysis, category: str) -> np.ndarray:
    if category not in CATEGORIES:
        raise MaskingError("invalid_input", f"category must be one of {CATEGORIES}")
    check_framing(a, category)

    w, h = a.size
    P = a.points
    cls = a.classes
    person = cls != BG
    # "other" = accessories (bags, hats, shoes): protected unless doing a full look.
    clothes = (cls == CLOTHES) | ((cls == OTHER) if category == "overall" else False)
    ys = np.arange(h)[:, None]

    shoulder_w = float(np.linalg.norm(P[L_SHOULDER] - P[R_SHOULDER]))
    torso_h = float(np.mean([P[L_HIP][1], P[R_HIP][1]]) - np.mean([P[L_SHOULDER][1], P[R_SHOULDER][1]]))
    torso_h = max(torso_h, 0.5 * shoulder_w)
    waist_y = float(np.mean([P[L_HIP][1], P[R_HIP][1]]))
    arm_t = max(4, int(0.42 * shoulder_w))
    leg_t = max(4, int(0.55 * shoulder_w))

    def canvas() -> np.ndarray:
        return np.zeros((h, w), np.uint8)

    def poly(ids) -> np.ndarray:
        c = canvas()
        cv2.fillPoly(c, [P[list(ids)].astype(np.int32)], 1)
        return c.astype(bool)

    def limbs(chains, thickness) -> np.ndarray:
        c = canvas()
        for chain in chains:
            pts = [P[i] for i in chain if a.visibility[i] >= VISIBLE * 0.6]
            for p0, p1 in zip(pts, pts[1:]):
                cv2.line(c, tuple(map(int, p0)), tuple(map(int, p1)), 1, thickness)
        return c.astype(bool)

    def discs(ids, radius) -> np.ndarray:
        c = canvas()
        for i in ids:
            cv2.circle(c, tuple(map(int, P[i])), radius, 1, -1)
        return c.astype(bool)

    face_hair = (cls == FACE_SKIN) | (cls == HAIR)
    hands = discs((L_WRIST, R_WRIST, *L_HAND, *R_HAND), max(3, int(0.20 * shoulder_w))) & ~clothes
    feet = discs((L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_FOOT, R_FOOT), max(3, int(0.28 * shoulder_w)))

    arms = limbs([(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)], arm_t) & person
    torso = poly((R_SHOULDER, L_SHOULDER, L_HIP, R_HIP))
    legs = limbs([(L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE), (L_HIP, R_HIP)], leg_t)

    if category == "upper":
        # MediaPipe's hip point sits on the joint, below the waistband.
        waistband = waist_y - 0.10 * torso_h
        body = (clothes & (ys < waistband)) | torso
        region = _hull(body & (ys < waist_y)) | arms
        protect = face_hair | hands | (ys > waist_y)
    elif category == "lower":
        top = waist_y - 0.12 * torso_h
        body = (clothes & (ys > top)) | (legs & person)
        region = _hull(body & (ys > top))
        protect = face_hair | hands | feet | (ys < top)
    else:  # overall: dresses, jumpsuits, full looks (+ shoes)
        body = clothes | torso | (legs & person)
        region = _hull(body & ~face_hair) | arms | (feet & person)
        protect = face_hair | hands

    k = max(3, int(0.018 * h)) | 1
    region = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    mask = region & ~protect
    return (mask * 255).astype(np.uint8)


def _hull(m: np.ndarray) -> np.ndarray:
    m8 = m.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return m
    # Drop specks (stray "clothes" pixels on the background) before hulling.
    biggest = max(cv2.contourArea(c) for c in contours)
    keep = [c for c in contours if cv2.contourArea(c) >= 0.02 * biggest]
    out = np.zeros_like(m8)
    cv2.fillPoly(out, [cv2.convexHull(np.vstack(keep))], 1)
    return out.astype(bool)


def overlay(img: Image.Image, mask: Image.Image, color=(255, 64, 160), alpha: float = 0.5) -> Image.Image:
    """Debug visualisation used by the mask gallery and `return_mask`."""
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    m = (np.asarray(mask.convert("L").resize(img.size)) > 127)[..., None]
    tint = base * (1 - alpha) + np.array(color, np.float32) * alpha
    return Image.fromarray(np.where(m, tint, base).astype(np.uint8))
