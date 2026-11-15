# manyfaced

manyfaced is a Rust port of the face pipeline of [dlib](https://github.com/davisking/dlib). It
is packaged for Python as one abi3 wheel for each platform. It reads the original dlib `.dat`
models and gives the same results as dlib.

The name comes from the [Many-Faced God](https://gameofthrones.fandom.com/wiki/Many-Faced_God) in
*A Song of Ice and Fire* by George R. R. Martin. The temple of this god keeps a hall of faces.

## Why this project exists

I maintain [dlib-bin](https://pypi.org/project/dlib-bin/), a binary distribution of dlib. dlib is
a very capable library, but it is difficult to build:

- The official dlib release on PyPI is a source distribution. `pip install dlib` compiles it with
  CMake and a C++ compiler. This takes a long time.
- The build frequently fails on Windows, on new Python versions, on arm64 and on musl.
- dlib-bin solves the build problem, but it needs one build for each Python version and platform.
  The dlib bindings do not support abi3.

Many users cannot compile dlib on each CI run. I think that most users need only a small part of
dlib: face detection, face landmarks and a descriptor to compare faces. I have no official data
for this. One indication is the download count in October 2026
([pypistats](https://pypistats.org/packages/dlib)): approximately 93k for
[face_recognition](https://github.com/ageitgey/face_recognition), which wraps that pipeline, and
172k for dlib.

manyfaced is a port of that subset. It is not a fork of dlib and it does not compete with dlib.
The algorithms and the models are the work and the intellectual property of the dlib authors.
dlib stays the reference. I am not an expert in these algorithms. [Claude
Code](https://claude.com/claude-code) generated 100% of the code. It used the dlib source code as
the specification and the dlib outputs as the test oracle.

## Ported features

| Feature | dlib | manyfaced | Result compared with dlib |
|---|---|---|---|
| HOG frontal face detector | `get_frontal_face_detector`, `fhog_object_detector` | `FhogDetector` | bit-exact |
| Facial landmarks, 5 and 68 points | `shape_predictor` | `ShapePredictor` | identical |
| Aligned face chips | `get_face_chip(s)`, `get_face_chip_details`, `extract_image_chip(s)` | `face_chip(s)`, `face_chip_details`, `extract_chip(s)` | bit-exact |
| 128-D face descriptors (ResNet) | `face_recognition_model_v1` | `FaceRecognizer` | L2 ≤ 1e-6 |
| Face clustering | `chinese_whispers_clustering`, `chinese_whispers` | same names | identical labels |

These dlib features are not ported:

- The CNN (MMOD) face detector and the correlation tracker.
- All training, the SVMs and the other machine learning tools.
- Optimization (`find_min_global`), the GUI and most of the image processing tools.

Most of these tools are useful in C++, where dlib is a toolkit with no dependencies. In Python,
other libraries supply them: scikit-learn, PyTorch, OpenCV, Optuna and the standard library. Most
Python users install dlib for face detection and recognition. For this reason, manyfaced ports
only that part.

## Installation

```bash
pip install manyfaced
```

manyfaced is not on PyPI at this time. To build it from the source code, refer to
[docs/development.md](docs/development.md).

Wheels are available for these platforms:

- Linux: manylinux 2.28 and musllinux 1.2, x86_64 and aarch64.
- macOS: arm64.
- Windows: x64 and arm64.

Requirements:

- CPython 3.11 or later, numpy and Pillow.
- On x86_64, an x86-64-v3 CPU (AVX2 and FMA: Intel 2013 or later, AMD Zen 2017 or later).
- On Windows on Arm, a native arm64 Python. An x64 Python gets the x64 wheel and runs it in
  emulation.

The wheels contain the 5-point landmark model and the face recognition ResNet. The wheels do not
contain the 68-point model, because it is 100 MB and its training data does not permit commercial
use. For more data, refer to [docs/models.md](docs/models.md).

## Usage

```python
import manyfaced

faces = manyfaced.find_faces("photo.jpg")  # list[manyfaced.Face]: rect, landmarks, descriptor
labels = manyfaced.chinese_whispers_clustering([f.descriptor for f in faces], 0.5)
```

`find_faces` accepts a numpy array, the path of an image file (`str` or `os.PathLike`) or a
Pillow image. For a path or a Pillow image, `find_faces` applies the EXIF orientation, so that
phone photos come out upright. The first call makes the bundled models with
the default options, and the next calls use them again.

To control each step, use the models directly:

```python
import numpy as np
from PIL import Image

import manyfaced

image = np.asarray(Image.open("photo.jpg").convert("RGB"))  # uint8 HxWx3 or HxW

detector = manyfaced.FhogDetector()
predictor = manyfaced.ShapePredictor()  # bundled 5-point model
recognizer = manyfaced.FaceRecognizer()  # bundled ResNet

faces = detector(image, upsample=1)  # list[manyfaced.Rect]
landmarks = [predictor(image, face) for face in faces]  # (5, 2) int64 arrays
descriptors = recognizer.embed_many(image, landmarks)  # (N, 128) float64
labels = manyfaced.chinese_whispers_clustering(descriptors, 0.5)
```

- For the full API, refer to [docs/api.md](docs/api.md).
- To keep the dlib API in existing code, use `manyfaced.compat.dlib`.
- To run face_recognition code on manyfaced, call `manyfaced.compat.install()`.

For the two compatibility layers, refer to [docs/compat.md](docs/compat.md).

## Parity and performance

The tests compare manyfaced with dlib 20.0.1 on the dlib example images and on edge cases. For the
results, refer to [docs/parity.md](docs/parity.md).

On one thread, the time of each operation is 0.5× to 1.1× the dlib time on macOS arm64. On Linux
aarch64 it is 0.1× to 1.4× the dlib time. The full pipeline (detection, landmarks and
descriptor) takes 0.85× the dlib time on macOS arm64 and 0.27× to 0.38× on Linux aarch64. These
values do not include the model load times. There are no measurements on native x86_64 at this
time. For the full data, refer to [docs/performance.md](docs/performance.md).

manyfaced releases the GIL in each step, and many threads can use the same models. On a Mac with
15 cores and 15 Python threads, the full pipeline does 142 images/s. dlib does 52 images/s,
because its bindings release the GIL only in the detector.

## License

manyfaced is available under the MIT license or the Apache-2.0 license, as you choose
([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)). It contains code ported from
dlib and the weights of the dlib frontal face detector. These are under the Boost Software
License 1.0 ([THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)). For the model licenses, refer to
[docs/models.md](docs/models.md).
