"""Golden detections from dlib's frontal face detector.

For every test image and configuration, stores what ``dlib.get_frontal_face_detector().run``
returns: rectangles, scores and filter (weight) indices, in dlib's order. Rectangles and scores
are not image data, so ``tests/golden/detections.npz`` is committed.
"""

from __future__ import annotations

import sys

import numpy as np
from golden_common import GOLDEN_DIR, TestImage, dlib, meta

UPSAMPLES = (0, 1, 2)
ADJUST_THRESHOLDS = (0.0, -0.5)
# Upsampling twice multiplies the pixel count by ~16: skip it on the very large images.
MAX_PIXELS_FOR_UPSAMPLE_2 = 1_000_000


def generate(images: list[TestImage]) -> None:
    detector = dlib.get_frontal_face_detector()
    case_image, case_upsample, case_adjust, case_start, case_count = [], [], [], [], []
    rects, scores, indices = [], [], []
    for img in images:
        arr = img.contiguous()
        for up in UPSAMPLES:
            if up == 2 and arr.shape[0] * arr.shape[1] > MAX_PIXELS_FOR_UPSAMPLE_2:
                continue
            for adjust in ADJUST_THRESHOLDS:
                dets, det_scores, det_idx = detector.run(arr, up, adjust)
                if adjust == 0.0:
                    # detector(img, upsample) is run(img, upsample, 0.0) without the scores.
                    plain = detector(arr, up)
                    assert [tuple(_r(d)) for d in plain] == [tuple(_r(d)) for d in dets]
                case_image.append(img.id)
                case_upsample.append(up)
                case_adjust.append(adjust)
                case_start.append(len(rects))
                case_count.append(len(dets))
                rects += [_r(d) for d in dets]
                scores += list(det_scores)
                indices += list(det_idx)
    out = GOLDEN_DIR / "detections.npz"
    np.savez_compressed(
        out,
        **meta(images),
        case_image=np.array(case_image),
        case_upsample=np.array(case_upsample, dtype=np.int64),
        case_adjust=np.array(case_adjust, dtype=np.float64),
        case_start=np.array(case_start, dtype=np.int64),
        case_count=np.array(case_count, dtype=np.int64),
        rects=np.array(rects, dtype=np.int64).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float64),
        weight_indices=np.array(indices, dtype=np.int64),
    )
    print(
        f"{out.relative_to(GOLDEN_DIR.parent.parent)}: {len(case_image)} cases, "
        f"{len(rects)} detections",
        file=sys.stderr,
    )


def _r(d: dlib.rectangle) -> list[int]:
    return [d.left(), d.top(), d.right(), d.bottom()]
