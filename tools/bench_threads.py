#!/usr/bin/env python3
"""Multi-threaded throughput (images per second) of manyfaced and dlib on the same images.

    VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python tools/bench_threads.py [--threads 1,2,4,8] [--markdown]

N Python threads take images from a shared queue. manyfaced shares one set of models between all
threads (they are immutable and release the GIL). dlib gets one set of models per thread, since
its detector is not thread-safe; its bindings release the GIL in the HOG detector only, not in
shape_predictor or compute_face_descriptor. BLAS threads are pinned to 1 for both libraries so
that the only parallelism is the Python threads.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import manyfaced  # noqa: E402
import testimages  # noqa: E402

try:
    import dlib
except ImportError:
    sys.exit("bench_threads.py needs the original dlib: uv sync --group golden")

M = testimages.MODELS_DIR
SP5 = M / "shape_predictor_5_face_landmarks.dat"
RESNET = M / "dlib_face_recognition_resnet_model_v1.dat"
UPSAMPLE = 1


def dlib_worker_factory():
    local = threading.local()

    def models():
        if not hasattr(local, "det"):
            local.det = dlib.get_frontal_face_detector()
            local.sp = dlib.shape_predictor(str(SP5))
            local.rec = dlib.face_recognition_model_v1(str(RESNET))
        return local

    def detect(img):
        return len(models().det(img, UPSAMPLE))

    def pipeline(img):
        m = models()
        shapes = dlib.full_object_detections()
        for r in m.det(img, UPSAMPLE):
            shapes.append(m.sp(img, r))
        return len(m.rec.compute_face_descriptor(img, shapes)) if len(shapes) else 0

    return models, detect, pipeline


def manyfaced_worker_factory():
    det = manyfaced.FhogDetector()
    sp = manyfaced.ShapePredictor(SP5)
    rec = manyfaced.FaceRecognizer(RESNET)

    def models():
        return None

    def detect(img):
        return len(det(img, UPSAMPLE))

    def pipeline(img):
        rects = det(img, UPSAMPLE)
        return len(rec.embed_many(img, [sp(img, r) for r in rects])) if rects else 0

    return models, detect, pipeline


def throughput(fn, images, threads: int, warm, repeats: int = 3) -> float:
    """Best images/second over `repeats` runs of all `images` through `threads` workers."""
    best = 0.0
    with ThreadPoolExecutor(threads) as pool:
        # Start every worker and load its models (dlib) outside the timed region. The barrier
        # holds each task until all `threads` are running, so each lands on its own thread
        # (the pool only reuses idle threads).
        barrier = threading.Barrier(threads)

        def start(_: int) -> None:
            warm()
            barrier.wait()

        list(pool.map(start, range(threads)))
        list(pool.map(fn, images[:threads]))
        for _ in range(repeats):
            t0 = time.perf_counter()
            list(pool.map(fn, images))
            best = max(best, len(images) / (time.perf_counter() - t0))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=8, help="passes over the image set")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    counts = [int(t) for t in args.threads.split(",")]

    base = [
        img.contiguous()
        for img in testimages.load_images()
        if img.id.startswith("faces/") and img.id != "faces/bald_guys.jpg"
    ]
    images = base * args.rounds
    libs = {"dlib": dlib_worker_factory(), "manyfaced": manyfaced_worker_factory()}

    results: dict[tuple[str, str], list[float]] = {}
    for task in ("detect", "pipeline"):
        for name, (warm, detect, pipeline) in libs.items():
            fn = detect if task == "detect" else pipeline
            results[(task, name)] = [throughput(fn, images, n, warm) for n in counts]
            if not args.markdown:
                print(task, name, [f"{r:.1f}" for r in results[(task, name)]], flush=True)

    if args.markdown:
        by_area = sorted(base, key=lambda a: a.shape[0] * a.shape[1])
        small, large = (f"{a.shape[1]}×{a.shape[0]}" for a in (by_area[0], by_area[-1]))
        print(
            f"{platform.platform()}, {os.cpu_count()} logical CPUs, Python "
            f"{platform.python_version()}, dlib {dlib.__version__}, "
            f"manyfaced {manyfaced.__version__}; "
            f"{len(base)} images of {small} to {large} px, each {args.rounds} times, "
            f"upsample {UPSAMPLE}, images/s\n"
        )
        head = " | ".join(f"{n} thr" for n in counts)
        print(f"| Task | Library | {head} | scaling {counts[0]}→{counts[-1]} |")
        print("|---|---|" + "---:|" * (len(counts) + 1))
        labels = {"detect": "detector", "pipeline": "detect + 5 pts + descriptor"}
        for (task, name), r in results.items():
            cells = " | ".join(f"{v:.1f}" for v in r)
            print(f"| {labels[task]} | {name} | {cells} | {r[-1] / r[0]:.1f}× |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
