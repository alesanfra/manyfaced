"""Golden face descriptors from dlib's face_recognition_model_v1.

Inputs are dlib's own 5- and 68-point landmarks on the XML ground truth boxes and the HOG
detections (upsample 1) of every RGB test image (dlib's compute_face_descriptor only accepts
RGB). Three of dlib's overloads are recorded:

- ``single``: ``compute_face_descriptor(img, face)``, one face at a time;
- ``batch``: ``compute_face_descriptor(img, faces)`` for images with several detections, whose
  chips are extracted together and can differ slightly from single extractions;
- ``chip``: ``compute_face_descriptor(chip)`` on ``dlib.get_face_chip(img, face)``. The chip
  pixels are not stored (image licenses), only their digest: the tests rebuild them with
  manyfaced's bit-exact chip extraction and check the digest.

Descriptors are stored as float32: dlib computes them in float and only widens them to double.
"""

from __future__ import annotations

import sys

import numpy as np
from golden_common import GOLDEN_DIR, MODELS_DIR, TestImage, dlib, landmark_cases, meta, testimages

MODEL = "dlib_face_recognition_resnet_model_v1.dat"
PREDICTORS = ("shape_predictor_5_face_landmarks", "shape_predictor_68_face_landmarks")


def generate(images: list[TestImage]) -> None:
    by_id = {img.id: img for img in images}
    rgb = [img for img in images if img.array.ndim == 3]
    cases = [c for c in landmark_cases(rgb) if c[1] in ("xml", "hog_up1")]
    fr = dlib.face_recognition_model_v1(str(MODELS_DIR / MODEL))

    lm_image, lm_source, lm_parts, lm_points, dets = [], [], [], [], []
    for model in PREDICTORS:
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
            dets.append(det)

    def as_array(v) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64)
        assert a.shape == (128,)
        # dlib converts a float descriptor to double: the values are exact float32.
        assert np.array_equal(a, a.astype(np.float32).astype(np.float64))
        return a

    single = np.array(
        [
            as_array(fr.compute_face_descriptor(by_id[lm_image[i]].contiguous(), d))
            for i, d in enumerate(dets)
        ]
    )

    # Batches: all HOG detections of an image (per landmark model) in one call.
    groups: dict[tuple[str, int], list[int]] = {}
    for i, source in enumerate(lm_source):
        if source == "hog_up1":
            groups.setdefault((lm_image[i], lm_parts[i]), []).append(i)
    batch_landmarks, batch_ids, batch_desc = [], [], []
    for b, members in enumerate(m for m in groups.values() if len(m) >= 2):
        faces = dlib.full_object_detections()
        for i in members:
            faces.append(dets[i])
        out = fr.compute_face_descriptor(by_id[lm_image[members[0]]].contiguous(), faces)
        assert len(out) == len(members)
        for i, d in zip(members, out, strict=True):
            batch_landmarks.append(i)
            batch_ids.append(b)
            batch_desc.append(as_array(d))

    chip_desc, chip_digests = [], []
    for i, d in enumerate(dets):
        chip = np.asarray(dlib.get_face_chip(by_id[lm_image[i]].contiguous(), d), np.uint8)
        chip_digests.append(testimages.digest(chip))
        chip_desc.append(as_array(fr.compute_face_descriptor(chip)))

    out = GOLDEN_DIR / "descriptors.npz"
    np.savez_compressed(
        out,
        **meta(images),
        model=np.array(MODEL),
        lm_image=np.array(lm_image),
        lm_source=np.array(lm_source),
        lm_num_parts=np.array(lm_parts, dtype=np.int64),
        lm_points=np.array(lm_points, dtype=np.int64),
        single=single.astype(np.float32),
        batch_landmarks=np.array(batch_landmarks, dtype=np.int64),
        batch_ids=np.array(batch_ids, dtype=np.int64),
        batch=np.array(batch_desc).reshape(-1, 128).astype(np.float32),
        chip_digests=np.array(chip_digests),
        chip=np.array(chip_desc).astype(np.float32),
    )
    print(
        f"{out.relative_to(GOLDEN_DIR.parent.parent)}: {len(dets)} faces, "
        f"{len(batch_desc)} in {len(set(batch_ids))} batches",
        file=sys.stderr,
    )
