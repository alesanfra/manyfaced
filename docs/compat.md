# Migration from dlib and face_recognition

## dlib

`manyfaced.compat.dlib` has the names, signatures, defaults, `repr` values and exception types of
the dlib Python bindings for the features in the table. To use it in existing code, change only
the import.

```python
import manyfaced
import manyfaced.compat.dlib as dlib

img = dlib.load_rgb_image("photo.jpg")
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(str(manyfaced.model_path("shape_predictor_5_face_landmarks")))
facerec = dlib.face_recognition_model_v1(
    str(manyfaced.model_path("dlib_face_recognition_resnet_model_v1"))
)

for rect in detector(img, 1):
    shape = sp(img, rect)
    descriptor = facerec.compute_face_descriptor(img, shape)
```

| dlib (Python) | Status | Notes |
|---|---|---|
| `get_frontal_face_detector`, `fhog_object_detector` (`__call__`, `run`, `num_detectors`, window size) | ✓ | Bit-exact. No `run_multiple`. |
| `shape_predictor` | ✓ | Exact. |
| `full_object_detection(s)`, `rectangle(s)`, `drectangle`, `point(s)`, `dpoint(s)`, `vector(s)`, `chip_dims`, `chip_details` | ✓ | Same `repr` and `str`. `point` and `dpoint` compare by value. |
| `get_face_chip(s)`, `get_face_chip_details`, `extract_image_chip(s)` | ✓ | Bit-exact. 8-bit images only. |
| `face_recognition_model_v1.compute_face_descriptor` (all 5 overloads) | ✓ | RGB only, as in dlib. The jitter uses the manyfaced random generator. |
| `chinese_whispers_clustering`, `chinese_whispers` | ✓ | Identical labels. |
| `load_rgb_image` | ✓ | Pillow decodes the image. The EXIF orientation is ignored, as in dlib. |
| Pickling of `shape_predictor`, `fhog_object_detector` and `face_recognition_model_v1` | ✗ | Make the models again in each process. |
| `cnn_face_detection_model_v1` | Placeholder | You can make an instance. A call to it causes `NotImplementedError`. |
| Training, SVMs, correlation tracker, selective search, GUI, `find_min_global`, other image tools | ✗ | Not in the scope of manyfaced. The `AttributeError` gives the name of the feature. |

## face_recognition

This section is only for existing code that uses
[face_recognition](https://github.com/ageitgey/face_recognition). manyfaced does not need
face_recognition.

face_recognition depends on two packages:

- `dlib`, which pip compiles from the source code.
- `face_recognition_models`, which contains 100 MB of the same dlib models.

`manyfaced.compat.install()` supplies the two modules. For this reason, install face_recognition
without its dependencies:

```bash
pip install manyfaced Click
pip install --no-deps face_recognition
```

Call `manyfaced.compat.install()` before you import face_recognition:

```python
import manyfaced.compat

manyfaced.compat.install()

import face_recognition

image = face_recognition.load_image_file("photo.jpg")
locations = face_recognition.face_locations(image)  # model="hog"; "cnn" is not supported
encodings = face_recognition.face_encodings(image, locations)
```

face_recognition loads the 68-point predictor when you import it. For this reason, the first
import downloads the model (approximately 100 MB) one time. It also shows a warning about the
non-commercial license of the model.
