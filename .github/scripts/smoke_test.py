"""Smoke test for an installed manyfaced wheel: import, bundled models, one pipeline run.

Run it from a directory that does not contain the repository's python/ folder.
"""

import sys

import numpy as np

import manyfaced

print("manyfaced", manyfaced.__version__, "from", manyfaced.__file__)
assert "site-packages" in manyfaced.__file__, "the source tree was imported, not the wheel"

for name in ("shape_predictor_5_face_landmarks", "dlib_face_recognition_resnet_model_v1"):
    path = manyfaced.model_path(name)
    print(name, "->", path)

# Loads the bundled 5-point model and the ResNet, then runs the full pipeline.
manyfaced.ShapePredictor()
manyfaced.FaceRecognizer()
rng = np.random.default_rng(0)
image = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
faces = manyfaced.find_faces(image)
print("faces in noise:", len(faces))
print("ok on", sys.platform, sys.version.split()[0])
