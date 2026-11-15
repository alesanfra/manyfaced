# Performance

## One thread

Conditions of the measurement:

- One thread: `VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
- The best mean of 5 runs. The two libraries get the same inputs.
- dlib 20.0.1 from the `dlib-bin` wheels.
- Apple M-series. "Linux aarch64" is a manylinux wheel in an OrbStack Linux VM on the same
  computer.
- The script is `tools/bench.py --markdown`.

A ratio less than 1 means that manyfaced is faster.

| Operation | macOS arm64: dlib | manyfaced | ratio | Linux aarch64: dlib | manyfaced | ratio |
|---|---:|---:|---:|---:|---:|---:|
| load: get_frontal_face_detector (first call) | 110.43 ms | 406.9 µs | 0.004× | 117.92 ms | 352.5 µs | 0.003× |
| load: shape_predictor 5 points | 19.01 ms | 9.68 ms | 0.51× | 23.26 ms | 8.97 ms | 0.39× |
| load: shape_predictor 68 points | 177.51 ms | 84.06 ms | 0.47× | 212.62 ms | 89.23 ms | 0.42× |
| load: face_recognition_model_v1 | 18.52 ms | 1.70 ms | 0.09× | 21.08 ms | 2.49 ms | 0.12× |
| detector(img, 0), 500×375 RGB | 9.03 ms | 8.56 ms | 0.95× | 7.68 ms | 9.26 ms | 1.21× |
| detector(img, 1), 500×375 RGB | 35.70 ms | 33.41 ms | 0.94× | 31.04 ms | 34.89 ms | 1.12× |
| detector(img, 0), 500×375 gray | 7.70 ms | 7.39 ms | 0.96× | 6.52 ms | 8.08 ms | 1.24× |
| detector(img, 0), 1986×1545 RGB | 141.05 ms | 123.43 ms | 0.88× | 119.69 ms | 129.19 ms | 1.08× |
| run(img, 0, -0.5), 500×375 RGB | 9.04 ms | 8.57 ms | 0.95× | 7.71 ms | 9.30 ms | 1.21× |
| shape_predictor 5 points, RGB | 148.3 µs | 80.6 µs | 0.54× | 94.9 µs | 72.9 µs | 0.77× |
| shape_predictor 5 points, gray | 146.9 µs | 80.0 µs | 0.54× | 92.6 µs | 71.1 µs | 0.77× |
| shape_predictor 68 points, RGB | 311.4 µs | 201.3 µs | 0.65× | 383.5 µs | 256.7 µs | 0.67× |
| get_face_chip_details (5 points) | 0.7 µs | 0.6 µs | 0.86× | 0.3 µs | 0.5 µs | 1.33× |
| get_face_chip_details (68 points) | 0.9 µs | 0.9 µs | 0.97× | 0.5 µs | 0.7 µs | 1.42× |
| get_face_chip 150×150 | 84.4 µs | 85.3 µs | 1.01× | 72.9 µs | 84.9 µs | 1.16× |
| get_face_chip, ~1500 px face (pyramid) | 2.82 ms | 3.07 ms | 1.09× | 2.69 ms | 3.12 ms | 1.16× |
| get_face_chips, 3 faces | 248.4 µs | 256.9 µs | 1.03× | 214.3 µs | 254.6 µs | 1.19× |
| compute_face_descriptor(chip) | 3.87 ms | 2.38 ms | 0.61× | 57.04 ms | 6.63 ms | 0.12× |
| compute_face_descriptor(img, shape) | 3.96 ms | 2.44 ms | 0.62× | 57.16 ms | 6.70 ms | 0.12× |
| compute_face_descriptor(img, 3 shapes) | 11.67 ms | 7.33 ms | 0.63× | 170.02 ms | 20.09 ms | 0.12× |
| compute_face_descriptor(3 chips) | 11.38 ms | 7.09 ms | 0.62× | 169.95 ms | 19.90 ms | 0.12× |
| detect(up 1) + 5 pts + descriptor, 500×375 (3 faces) | 48.54 ms | 41.31 ms | 0.85× | 203.10 ms | 55.77 ms | 0.27× |
| detect(up 1) + 5 pts + descriptor, 1986×1545 (24 faces) | 679.74 ms | 576.35 ms | 0.85× | 1872.15 ms | 705.92 ms | 0.38× |

- On macOS, the face ResNet uses Apple Accelerate, as dlib-bin does. On other platforms it uses
  the pure-Rust `gemm` crate. The dlib-bin Linux wheels use no BLAS. This is the cause of the 8×
  difference on Linux.
- `get_face_chip_details` takes less than a microsecond. For this reason, its ratio shows only
  the overhead of the Python call.
- The detector load is the first call in a new process. dlib decompresses its embedded detector
  on each call to `get_frontal_face_detector()`. manyfaced does this one time in each process.
- There are no measurements on native x86_64 hardware at this time. The x86_64 build has only
  correctness tests, in Rosetta 2.

To do the measurement again, run this command:

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/bench.py --markdown
```

## Multi-threaded throughput

Conditions of the measurement:

- N Python threads send the same 11 dlib example images (500×332 px to 900×916 px, each image 8
  times, upsample 1) through a thread pool.
- The value is the best of 3 timed runs, in images per second.
- The BLAS threads are pinned to 1, as in the single-thread measurement.
- All manyfaced threads use one set of models. Each dlib thread gets its own set of models,
  because the dlib detector is not thread-safe.
- Apple M5 Pro (15 cores), macOS 26.7.1, Python 3.14.6.
- The script is `tools/bench_threads.py --markdown`.

| Task | Library | 1 thr | 2 thr | 4 thr | 5 thr | 8 thr | 15 thr | scaling 1→15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| detector | dlib | 19.8 | 37.9 | 70.9 | 82.6 | 120.1 | 185.2 | 9.4× |
| detector | manyfaced | 21.1 | 41.3 | 73.4 | 85.3 | 118.8 | 180.5 | 8.6× |
| detect + 5 pts + descriptor | dlib | 14.6 | 24.9 | 39.2 | 42.9 | 48.1 | 52.2 | 3.6× |
| detect + 5 pts + descriptor | manyfaced | 17.0 | 33.1 | 57.4 | 68.0 | 97.5 | 142.4 | 8.4× |

- The dlib bindings release the GIL only in the HOG detector. For the detector, the two
  libraries have approximately the same scaling.
- In the full dlib pipeline, only one thread at a time can run `shape_predictor` or
  `compute_face_descriptor`. The dlib throughput stops at approximately 50 images/s, from 5 to 8
  threads.
- manyfaced releases the GIL in all three steps, and the pipeline continues to scale. At 15
  threads, the manyfaced throughput is approximately 2.7× the dlib throughput.
- A second run gave values within 3.5% of the first run in each cell. There are no measurements
  on Linux or x86_64 at this time.
- The Accelerate and BLAS threads were pinned to 1, so that the Python threads are the only
  parallelism. If you do not pin them and you run many Python threads, there can be more threads
  than CPU cores.

To do the measurement again, run this command:

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/bench_threads.py --markdown
```
