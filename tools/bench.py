#!/usr/bin/env python3
"""Benchmark manyfaced against the original dlib, component by component, single-threaded.

    VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python tools/bench.py [--markdown]

Each row runs the same operation on the same inputs with both libraries and reports the best
mean time over several repeats (best-of-R of N calls), plus the manyfaced/dlib ratio. Needs the
original dlib (dlib-bin), the models in models/ and the dlib example images in third_party/dlib.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import manyfaced  # noqa: E402
import testimages  # noqa: E402

try:
    import dlib
except ImportError:
    sys.exit("bench.py needs the original dlib: uv sync --group golden")

M = testimages.MODELS_DIR
SP5 = M / "shape_predictor_5_face_landmarks.dat"
SP68 = M / "shape_predictor_68_face_landmarks.dat"
RESNET = M / "dlib_face_recognition_resnet_model_v1.dat"


def best(f: Callable[[], object], min_time: float = 0.2, repeats: int = 5) -> float:
    """Best mean seconds per call over `repeats` runs of at least `min_time` seconds each."""
    f()
    t0 = time.perf_counter()
    f()
    once = max(time.perf_counter() - t0, 1e-7)
    n = max(1, int(min_time / once))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        times.append((time.perf_counter() - t0) / n)
    return min(times)


def first_call(statement: str, repeats: int = 5) -> float:
    """Best time of `statement` run as the first call of a fresh interpreter."""
    code = (
        "import time, dlib, manyfaced\n"
        f"t = time.perf_counter(); {statement}; print(time.perf_counter() - t)"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(repeats)
    ]
    return min(float(r) for r in runs)


def landmarks(det: dlib.full_object_detection) -> np.ndarray:
    return np.array([[p.x, p.y] for p in det.parts()], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = parser.parse_args()

    images = {img.id: img.contiguous() for img in testimages.load_images()}
    small = images["faces/2008_001322.jpg"]  # 500x375
    small_gray = images["gray/2008_001322"]
    big = images["faces/bald_guys.jpg"]  # 1986x1545, many faces
    huge_face = images["large/tom_cruise_x3"]  # 2400x1695, one face of ~1500 px

    d_det = dlib.get_frontal_face_detector()
    f_det = manyfaced.FhogDetector()
    d_sp5, f_sp5 = dlib.shape_predictor(str(SP5)), manyfaced.ShapePredictor(SP5)
    d_sp68, f_sp68 = dlib.shape_predictor(str(SP68)), manyfaced.ShapePredictor(SP68)
    d_rec = dlib.face_recognition_model_v1(str(RESNET))
    f_rec = manyfaced.FaceRecognizer(RESNET)

    small_faces = d_det(small, 1)
    face = small_faces[0]
    box = (face.left(), face.top(), face.right(), face.bottom())
    shape5 = d_sp5(small, face)
    lm5 = landmarks(shape5)
    shape68 = d_sp68(small, face)
    lm68 = landmarks(shape68)
    shapes = dlib.full_object_detections()
    for f in small_faces:
        shapes.append(d_sp5(small, f))
    lms = [landmarks(s) for s in shapes]
    huge_det = d_det(huge_face, 0)[0]
    huge_shape = d_sp5(huge_face, huge_det)
    huge_lm = landmarks(huge_shape)
    chip = np.asarray(dlib.get_face_chip(small, shape5, 150, 0.25))
    chips = [np.asarray(c) for c in dlib.get_face_chips(small, shapes, 150, 0.25)]

    def pipeline_dlib(img: np.ndarray) -> list:
        return [d_rec.compute_face_descriptor(img, d_sp5(img, r)) for r in d_det(img, 1)]

    def pipeline_manyfaced(img: np.ndarray) -> list:
        rects = f_det(img, 1)
        if not rects:
            return []
        return f_rec.embed_many(img, [f_sp5(img, r) for r in rects])

    n_small = len(small_faces)
    n_big = len(d_det(big, 1))
    rows: list[tuple[str, str, Callable[[], object], Callable[[], object]]] = [
        (
            "Loading",
            "shape_predictor 5 points",
            lambda: dlib.shape_predictor(str(SP5)),
            lambda: manyfaced.ShapePredictor(SP5),
        ),
        (
            "Loading",
            "shape_predictor 68 points",
            lambda: dlib.shape_predictor(str(SP68)),
            lambda: manyfaced.ShapePredictor(SP68),
        ),
        (
            "Loading",
            "face_recognition_model_v1",
            lambda: dlib.face_recognition_model_v1(str(RESNET)),
            lambda: manyfaced.FaceRecognizer(RESNET),
        ),
        (
            "Detector",
            "detector(img, 0), 500×375 RGB",
            lambda: d_det(small, 0),
            lambda: f_det(small, 0),
        ),
        (
            "Detector",
            "detector(img, 1), 500×375 RGB",
            lambda: d_det(small, 1),
            lambda: f_det(small, 1),
        ),
        (
            "Detector",
            "detector(img, 0), 500×375 gray",
            lambda: d_det(small_gray, 0),
            lambda: f_det(small_gray, 0),
        ),
        (
            "Detector",
            "detector(img, 0), 1986×1545 RGB",
            lambda: d_det(big, 0),
            lambda: f_det(big, 0),
        ),
        (
            "Detector",
            "run(img, 0, -0.5), 500×375 RGB",
            lambda: d_det.run(small, 0, -0.5),
            lambda: f_det.run(small, 0, -0.5),
        ),
        (
            "Landmarks",
            "shape_predictor 5 points, RGB",
            lambda: d_sp5(small, face),
            lambda: f_sp5(small, box),
        ),
        (
            "Landmarks",
            "shape_predictor 5 points, gray",
            lambda: d_sp5(small_gray, face),
            lambda: f_sp5(small_gray, box),
        ),
        (
            "Landmarks",
            "shape_predictor 68 points, RGB",
            lambda: d_sp68(small, face),
            lambda: f_sp68(small, box),
        ),
        (
            "Face chip",
            "get_face_chip_details (5 points)",
            lambda: dlib.get_face_chip_details(shape5, 150, 0.25),
            lambda: manyfaced.face_chip_details(lm5, 150, 0.25),
        ),
        (
            "Face chip",
            "get_face_chip_details (68 points)",
            lambda: dlib.get_face_chip_details(shape68, 150, 0.25),
            lambda: manyfaced.face_chip_details(lm68, 150, 0.25),
        ),
        (
            "Face chip",
            "get_face_chip 150×150",
            lambda: dlib.get_face_chip(small, shape5),
            lambda: manyfaced.face_chip(small, lm5),
        ),
        (
            "Face chip",
            "get_face_chip, ~1500 px face (pyramid)",
            lambda: dlib.get_face_chip(huge_face, huge_shape),
            lambda: manyfaced.face_chip(huge_face, huge_lm),
        ),
        (
            "Face chip",
            f"get_face_chips, {n_small} faces",
            lambda: dlib.get_face_chips(small, shapes),
            lambda: manyfaced.face_chips(small, lms),
        ),
        (
            "Descriptor",
            "compute_face_descriptor(chip)",
            lambda: d_rec.compute_face_descriptor(chip),
            lambda: f_rec.embed_chip(chip),
        ),
        (
            "Descriptor",
            "compute_face_descriptor(img, shape)",
            lambda: d_rec.compute_face_descriptor(small, shape5),
            lambda: f_rec.embed(small, lm5),
        ),
        (
            "Descriptor",
            f"compute_face_descriptor(img, {n_small} shapes)",
            lambda: d_rec.compute_face_descriptor(small, shapes),
            lambda: f_rec.embed_many(small, lms),
        ),
        (
            "Descriptor",
            f"compute_face_descriptor({len(chips)} chips)",
            lambda: d_rec.compute_face_descriptor(chips),
            lambda: f_rec.embed_chips(chips),
        ),
        (
            "Pipeline",
            f"detect(up 1) + 5 pts + descriptor, 500×375 ({n_small} faces)",
            lambda: pipeline_dlib(small),
            lambda: pipeline_manyfaced(small),
        ),
        (
            "Pipeline",
            f"detect(up 1) + 5 pts + descriptor, 1986×1545 ({n_big} faces)",
            lambda: pipeline_dlib(big),
            lambda: pipeline_manyfaced(big),
        ),
    ]

    # manyfaced parses the embedded detector once per process and then clones it, so its steady
    # state is free; compare the first call of a fresh interpreter instead.
    results = [
        (
            "Loading",
            "get_frontal_face_detector (first call)",
            first_call("dlib.get_frontal_face_detector()"),
            first_call("manyfaced.FhogDetector()"),
        )
    ]
    for group, name, d, f in rows:
        td, tf = best(d), best(f)
        results.append((group, name, td, tf))
        if not args.markdown:
            print(
                f"{group:10} {name:58} dlib {td * 1e3:9.3f} ms  manyfaced {tf * 1e3:9.3f} ms  "
                f"{tf / td:5.2f}x",
                flush=True,
            )

    if args.markdown:
        print(
            f"{platform.platform()}, {platform.processor() or platform.machine()}, "
            f"Python {platform.python_version()}, dlib {dlib.__version__}, manyfaced "
            f"{manyfaced.__version__}, VECLIB_MAXIMUM_THREADS="
            f"{os.environ.get('VECLIB_MAXIMUM_THREADS', 'unset')}\n"
        )
        print("| Area | Operation | dlib | manyfaced | manyfaced / dlib |")
        print("|---|---|---:|---:|---:|")
        for group, name, td, tf in results:
            print(f"| {group} | {name} | {fmt(td)} | {fmt(tf)} | {tf / td:.2f}× |")
    return 0


def fmt(seconds: float) -> str:
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} µs"


if __name__ == "__main__":
    sys.exit(main())
