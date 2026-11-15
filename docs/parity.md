# Parity with dlib

The reference is dlib 20.0.1 (the `dlib-bin` wheel for macOS arm64). The test inputs are the dlib
example images and these edge cases:

- A grayscale image.
- Images of 13×17 px and 1×1 px.
- An image with no face.
- An image of 2400×1695 px.
- Views that are not contiguous.

`tools/gen_golden.py` makes the reference outputs. The tests are in `tests/`.

| Component | Cases | Result |
|---|---|---|
| HOG detector (`run`, upsample 0/1/2, adjust_threshold 0/−0.5) | 110 runs, 363 detections | Identical rectangles, order and filter indices. Score difference 0. |
| Landmarks, 5 and 68 points | 428 | Identical. |
| Face chips (with pyramid downscaling and batches) | 988 | Identical pixels. |
| Face descriptors | 232 faces (single, chip) + 140 in batches | Maximum L2 1.0e-6 (threshold 1e-4). Identical `compare_faces` decisions on 26 796 pairs. |
| chinese_whispers | 18 clusterings + 3 graphs | Identical labels. |
| face_recognition 1.3.0 on `manyfaced.compat` | 13 images, 69 faces | Identical `face_locations`, `face_landmarks` and `compare_faces` (2 346 pairs). |

The detector, the landmarks and the chips are bit-identical on arm64 and on x86_64. The
descriptors have float rounding differences (approximately 1e-6) between the Accelerate path
(macOS) and the pure-Rust path (Linux, Windows).

dlib does not give bit-identical results across its own builds. FMA contraction, BLAS, LAPACK and
the SIMD width change the low-order bits. dlib-bin for Linux aarch64 and dlib-bin for macOS arm64
give different results: in 6 of 110 detector cases, a rectangle moves by 1 px. manyfaced gives
the results of the macOS arm64 build on all platforms. For more data, refer to
[internals.md](internals.md).
