#!/usr/bin/env python3
"""Generate reference outputs from the original dlib into tests/golden/*.npz.

Run with the development venv, which has the original dlib installed
(``dlib-bin``, see docs/internals.md). dlib is only used here, never by manyfaced.

    python tools/gen_golden.py               # every component
    python tools/gen_golden.py landmarks     # only some components

Every golden file stores ``image_ids`` and ``image_digests`` for the images it
used, so the tests can verify they feed exactly the same pixels to manyfaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import golden_clusters  # noqa: E402
import golden_descriptors  # noqa: E402
import golden_detections  # noqa: E402
from golden_common import (  # noqa: E402
    GOLDEN_DIR,
    MODELS_DIR,
    TestImage,
    dlib,
    landmark_cases,
    testimages,
)
from golden_common import meta as _meta  # noqa: E402


def gen_landmarks(images: list[TestImage]) -> None:
    by_id = {img.id: img for img in images}
    cases = landmark_cases(images)
    for model in ("shape_predictor_5_face_landmarks", "shape_predictor_68_face_landmarks"):
        sp = dlib.shape_predictor(str(MODELS_DIR / f"{model}.dat"))
        parts = []
        for image_id, _, box in cases:
            det = sp(by_id[image_id].contiguous(), dlib.rectangle(*box))
            parts.append([[p.x, p.y] for p in det.parts()])
        num_parts = len(parts[0])
        out = GOLDEN_DIR / f"landmarks_{num_parts}.npz"
        np.savez_compressed(
            out,
            **_meta(images),
            model=np.array(f"{model}.dat"),
            case_image=np.array([c[0] for c in cases]),
            case_source=np.array([c[1] for c in cases]),
            rects=np.array([c[2] for c in cases], dtype=np.int64),
            parts=np.array(parts, dtype=np.int64),
        )
        print(f"{out.relative_to(GOLDEN_DIR.parent.parent)}: {len(cases)} cases", file=sys.stderr)


# Chip configurations: (size, padding). Paddings are exact in float32, so dlib.get_face_chip
# (which takes padding as a C++ float) and get_face_chip_details (double) agree.
CHIP_CONFIGS_ALL = [(150, 0.25)]
CHIP_CONFIGS_NATURAL = [(80, 0.5), (300, 0.0)]
# A tiny chip of a big face exercises the image pyramid (strong downscale) branch.
CHIP_CONFIG_DOWNSCALE = (32, 0.25)
CHIP_DOWNSCALE_IMAGES = (
    "large/tom_cruise_x3",
    "faces/bald_guys.jpg",
    "faces/Tom_Cruise_avp_2014_4.jpg",
)
CHIP_SYNTHETIC_SOURCES = ("partly_outside", "tiny_center")
LOCAL_GOLDEN_DIR = GOLDEN_DIR / "local"


def gen_chips(images: list[TestImage]) -> None:
    """Face chip details (committed) and chip pixels (local only, see docs/internals.md).

    Landmarks come from dlib's own shape predictors so these goldens do not depend on manyfaced's
    landmark parity. Chips are generated both one at a time (dlib.get_face_chip) and, for
    images with several detected faces, together (dlib.get_face_chips), because a batch shares
    one image pyramid and can differ slightly from single extractions.
    """
    by_id = {img.id: img for img in images}
    cases = [
        c
        for c in landmark_cases(images)
        if c[1] in ("xml", "hog_up1") or c[1] in CHIP_SYNTHETIC_SOURCES
    ]
    lm_image, lm_source, lm_parts, lm_points, lm_det = [], [], [], [], []
    for model in ("shape_predictor_5_face_landmarks", "shape_predictor_68_face_landmarks"):
        sp = dlib.shape_predictor(str(MODELS_DIR / f"{model}.dat"))
        for image_id, source, box in cases:
            det = sp(by_id[image_id].contiguous(), dlib.rectangle(*box))
            pts = np.zeros((68, 2), dtype=np.int64)
            parts = [[p.x, p.y] for p in det.parts()]
            pts[: len(parts)] = parts
            lm_image.append(image_id)
            lm_source.append(source)
            lm_parts.append(len(parts))
            lm_points.append(pts)
            lm_det.append(det)

    records = []  # (landmark index, size, padding, batch id)
    for i, source in enumerate(lm_source):
        natural = source in ("xml", "hog_up1")
        configs = list(CHIP_CONFIGS_ALL)
        if natural:
            configs += CHIP_CONFIGS_NATURAL
            if lm_image[i] in CHIP_DOWNSCALE_IMAGES:
                configs.append(CHIP_CONFIG_DOWNSCALE)
        records += [(i, size, pad, -1) for size, pad in configs]
    # Batches: every HOG detection of an image with several faces, per model.
    groups: dict[tuple[str, int], list[int]] = {}
    for i, source in enumerate(lm_source):
        if source == "hog_up1":
            groups.setdefault((lm_image[i], lm_parts[i]), []).append(i)
    batches = [members for members in groups.values() if len(members) >= 2]
    for b, members in enumerate(batches):
        records += [(i, 150, 0.25, b) for i in members]

    rects, angles, rows, cols, digests, chips = [], [], [], [], [], {}
    batch_chips = {}
    for b, members in enumerate(batches):
        arr = by_id[lm_image[members[0]]].contiguous()
        dets = dlib.full_object_detections()
        for i in members:
            dets.append(lm_det[i])
        batch_chips[b] = dict(zip(members, dlib.get_face_chips(arr, dets, 150, 0.25), strict=True))
    for k, (i, size, pad, b) in enumerate(records):
        d = dlib.get_face_chip_details(lm_det[i], size, pad)
        r = d.rect
        rects.append([r.left(), r.top(), r.right(), r.bottom()])
        angles.append(d.angle)
        rows.append(d.rows)
        cols.append(d.cols)
        if b >= 0:
            chip = batch_chips[b][i]
        else:
            chip = dlib.get_face_chip(by_id[lm_image[i]].contiguous(), lm_det[i], size, pad)
        chip = np.asarray(chip, dtype=np.uint8)
        chips[f"chip_{k:05d}"] = chip
        digests.append(testimages.digest(chip))

    details = {
        **_meta(images),
        "lm_image": np.array(lm_image),
        "lm_source": np.array(lm_source),
        "lm_num_parts": np.array(lm_parts, dtype=np.int64),
        "lm_points": np.array(lm_points, dtype=np.int64),
        "rec_landmarks": np.array([r[0] for r in records], dtype=np.int64),
        "rec_size": np.array([r[1] for r in records], dtype=np.int64),
        "rec_padding": np.array([r[2] for r in records], dtype=np.float64),
        "rec_batch": np.array([r[3] for r in records], dtype=np.int64),
        "rec_rect": np.array(rects, dtype=np.float64),
        "rec_angle": np.array(angles, dtype=np.float64),
        "rec_rows": np.array(rows, dtype=np.int64),
        "rec_cols": np.array(cols, dtype=np.int64),
        "rec_chip_digest": np.array(digests),
    }
    out = GOLDEN_DIR / "face_chip_details.npz"
    np.savez_compressed(out, **details)
    LOCAL_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    local = LOCAL_GOLDEN_DIR / "face_chips.npz"
    # The chip digests tie the local pixels to the committed details file.
    np.savez_compressed(local, rec_chip_digest=details["rec_chip_digest"], **chips)
    print(
        f"{out.relative_to(GOLDEN_DIR.parent.parent)}: {len(records)} chips "
        f"({len(batches)} batches); pixels in {local.relative_to(GOLDEN_DIR.parent.parent)}",
        file=sys.stderr,
    )


GENERATORS = {
    "landmarks": gen_landmarks,
    "chips": gen_chips,
    "descriptors": golden_descriptors.generate,
    "detections": golden_detections.generate,
    "clusters": golden_clusters.generate,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "components",
        nargs="*",
        choices=[[], *GENERATORS],
        help="components to generate (default: all)",
    )
    args = parser.parse_args()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    images = testimages.load_images()
    for name in args.components or GENERATORS:
        GENERATORS[name](images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
