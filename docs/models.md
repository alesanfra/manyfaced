# Models

manyfaced uses the original dlib model files.

| Model | In the wheel | Used by |
|---|---|---|
| HOG frontal face detector | ✓ (embedded in the library) | `manyfaced.FhogDetector()` |
| `shape_predictor_5_face_landmarks.dat` | ✓ | `manyfaced.ShapePredictor()` |
| `dlib_face_recognition_resnet_model_v1.dat` | ✓ | `manyfaced.FaceRecognizer()` |
| `shape_predictor_68_face_landmarks.dat` | ✗ (100 MB, non-commercial) | `manyfaced.download_model("shape_predictor_68_face_landmarks")` |

- `manyfaced.model_path(name)` gives the path of a bundled model. Use it, for example, with an
  API in the dlib style.
- `manyfaced.download_model(name, directory=None)` downloads a model from
  <https://dlib.net/files/> into `manyfaced.model_cache_dir()` (or `$MANYFACED_MODEL_DIR`). It
  makes sure that the SHA-256 of the file is the pinned value. The next calls use the same file.
- For development, `python tools/download_models.py` downloads the three models into `models/`.

## Licenses

The models are the work of the dlib authors, as are the algorithms.

- The training data of `shape_predictor_68_face_landmarks.dat` is iBUG 300-W. Its annotations are
  "for research purposes ONLY (NO commercial products)"
  ([iBUG](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)). For this reason, the
  dlib model page says that you cannot use the model in a commercial product
  ([dlib-models](https://github.com/davisking/dlib-models)). The 5-point model is sufficient for
  face chips and descriptors.
- The dlib author released `shape_predictor_5_face_landmarks.dat` and
  `dlib_face_recognition_resnet_model_v1.dat` into the public domain
  ([dlib-models](https://github.com/davisking/dlib-models)). If the terms of their training data
  are important for your use, examine them yourself.
- The embedded HOG detector weights come from the dlib source code (Boost Software License 1.0,
  refer to [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES)).
