# Native API

```python
import manyfaced

detector = manyfaced.FhogDetector()  # embedded frontal face detector, or a .svm path
predictor = manyfaced.ShapePredictor()  # bundled 5-point model, or a .dat path
recognizer = manyfaced.FaceRecognizer()  # bundled ResNet, or a .dat path
```

| Entry point | Use |
|---|---|
| `manyfaced.find_faces(image, *, upsample=1, num_jitters=0)` | The full pipeline with the default models. The first call makes the models, and all threads use them. Gives `list[Face]` in detector order. `image` is an array, a path (`str` or `os.PathLike`) or a Pillow image. Paths and Pillow images get the EXIF orientation. |
| `manyfaced.Face` | A frozen dataclass: `rect: Rect`, `landmarks` int64 `(5, 2)`, `descriptor` float64 `(128,)`. The arrays from `find_faces` are read-only. Faces compare by identity. |
| `manyfaced.FhogDetector(path=None)` | The dlib frontal face detector (one shared instance), or a HOG detector trained with dlib (`.svm`). `det(image, upsample=0) -> list[Rect]`. `det.run(image, upsample=0, adjust_threshold=0.0) -> (rects, scores, filter_indices)`. |
| `manyfaced.ShapePredictor(path=None)` | `sp(image, rect) -> int64 (N, 2)` of `(x, y)`. `rect` is a `manyfaced.Rect` or `(left, top, right, bottom)`. |
| `manyfaced.face_chip_details(landmarks, size=150, padding=0.25)` | Gives a `ChipDetails`. For aligned crops, use `manyfaced.extract_chip(s)` and `manyfaced.face_chip(s)`. |
| `manyfaced.FaceRecognizer(path=None)` | `embed`, `embed_many`, `embed_chip` and `embed_chips` give float64 `(128,)` or `(N, 128)`. Options: `num_jitters`, `padding`, `seed`. |
| `manyfaced.chinese_whispers_clustering(descriptors, threshold)`, `manyfaced.chinese_whispers(edges)` | Labels, identical to the dlib labels. |
| `manyfaced.model_path(name)`, `manyfaced.download_model(name, directory=None)`, `manyfaced.model_cache_dir()` | Model files. Refer to [models.md](models.md). |

## Models

- Make each model one time and use it again. Each constructor reads and parses the model file.
- A missing file causes `FileNotFoundError`. A file that is not a valid model causes
  `manyfaced.ModelFormatError`.
- Models do not change after they load. Many threads can use the same model. manyfaced releases
  the GIL during each computation. The dlib detector is not thread-safe.
- You cannot pickle a model. To use a process pool, make the models in each worker process, for
  example in the `initializer` of the pool.

## Images

- An image is a numpy `uint8` array: `HxW` (grayscale) or `HxWx3` (RGB).
- manyfaced accepts views that are not contiguous. It copies them one time.

## Descriptors

- Two descriptors are of the same person when their Euclidean distance is 0.6 or less. This is the
  dlib threshold.
- When `num_jitters > 1`, the descriptor is the mean of the descriptors of chips with random
  jitter, as in dlib. manyfaced uses its own random generator with a seed. The results are
  deterministic, but they are not bit-identical to the dlib results.

## Errors

All errors are exceptions:

- `FileNotFoundError`: the model file does not exist.
- `manyfaced.ModelFormatError`: the model file is damaged or is the wrong type of model. This
  class is a subclass of `ValueError`.
- `TypeError` or `ValueError`: an input is not correct.
- `MemoryError`: an input needs more memory than is available. Examples are a very large view
  made with `np.broadcast_to`, and a `chinese_whispers` edge with a very large node index.

These inputs also cause `ValueError`:

- A face chip of more than 2^28 pixels (16384 × 16384).
- Descriptors with values that are not finite, in `chinese_whispers_clustering`.
