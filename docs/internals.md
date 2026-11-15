# Internals

This document gives the data that a maintainer needs and that the code does not show:

- The dlib details that control parity.
- The differences between dlib builds.
- The reasons for the design decisions that are not obvious.

All of the data was checked against dlib v20.0.1 (`third_party/dlib`, commit `9b09738b`) and
against `dlib-bin==20.0.1.post1`.

## The reference dlib build

dlib does not give bit-identical results across its own builds. FMA contraction, BLAS, LAPACK and
the SIMD path change the low-order bits, and sometimes more:

- dlib-bin for macOS arm64 is compiled with clang, which enables FMA contraction by default on
  arm64. It links Accelerate. dlib-bin for Linux aarch64 is compiled with GCC, without BLAS or
  LAPACK. Compared with the macOS detector goldens, the Linux wheel moves a rectangle by 1 px in 6
  of 110 cases. Its scores are different by up to 3e-2.
- On ARM, the dlib `simd4f` division and square root are not IEEE operations. They are
  `vrecpe`/`vrsqrte` estimates, with two Newton–Raphson steps (`simd4f.h`). On x86, dlib uses
  `_mm_div_ps` and `_mm_sqrt_ps`.
- Landmarks: a dlib build with `-ffp-contract=on` and without LAPACK is different from dlib-bin
  on one degenerate case. The case is 68 points, `Tom_Cruise_avp_2014_4.jpg`, box
  `[-283,-400,847,1200]`. There, a feature pixel is on a rounding tie, and all 68 points move by
  up to 38 px.
- Chip details: only a build with Accelerate BLAS+LAPACK **and** FMA contraction gives the
  dlib-bin values bit for bit. manyfaced is equal to a dlib build without BLAS on 988 of 988
  cases. Compared with dlib-bin, manyfaced is bit-exact on 74 of 988 cases. On the other cases,
  the relative difference is 2e-14 or less. The chip pixels are bit-exact on all 988 cases.

manyfaced gives the results of **dlib-bin for macOS arm64** on all platforms:

- The goldens come from that build.
- The live comparisons with dlib (`test_compat.py`, `test_face_recognition.py`) run only on macOS
  arm64, in the `parity-macos` CI job. That job also makes the chip pixels again.
- `parity-linux` compares manyfaced with those references. It never compares manyfaced with the
  Linux dlib-bin.

These methods give the arm64 results:

- `detect/neon.rs` emulates the NEON estimate and Newton–Raphson sequences in software. One test
  compared it with the hardware instructions on all 2^32 floats. The committed test examines
  approximately 1M samples.
- The code uses `mul_add` only where clang fuses operations in the reference build. Rust never
  contracts `a * b + c` automatically, and `mul_add` is correctly rounded on all targets. As a
  result, the output does not change with the CPU. The x86-64-v3 build passes all golden tests
  in Rosetta, with and without Accelerate.
- The code keeps the dlib order of floating-point operations. This includes the order in which
  dlib calculates its matrix expressions:
  - `matrix / s` is `matrix * (1/s)` (`matrix.h`, `operator/`).
  - dlib moves scalar factors out of matrix products. As a result, `c * r * m` is `(r * m) * c`.
- `linalg.rs` is a literal port of the dlib `svd4` (Golub–Reinsch) and `pinv` that do not use
  LAPACK.

dlib-bin also has a bug. On strided numpy views (`img[::2, ::2]`), its detector sometimes gives
different scores after different earlier calls in the same process. On
`noncontig/tom_cruise_step2` it gave 0.634 instead of 1.906. The probable cause is a read of
stale memory. For this reason, `gen_golden.py` and the side-by-side tests always give contiguous
copies to dlib. manyfaced gets the original views.

## Serialization (`serialize/`)

- Integers: a control byte, then `size` little-endian bytes of the absolute value.
  - For signed types, `0x80` means negative and `size = ctrl & 0x0F`.
  - For unsigned types, `size = ctrl & 0x8F`. As a result, a sign bit makes the size too large.
  - `size == 0` and `size > sizeof(T)` are errors.
- Floats (`float_details`): an int64 mantissa **then** an int16 exponent.
  - The comment at `serialize.h:173` gives the opposite order. The code at line 645 is correct.
  - The exponents 32000, 32001 and 32002 are +inf, -inf and NaN.
  - The value is `ldexp((T)mantissa, e)`, so the target type controls the rounding. For this
    reason, the reader has one typed read for each C++ type.
  - The reader does not accept the legacy ASCII float format.
- `bool` is ASCII `'1'` or `'0'`. `std::string` and `vector<char>` are a length, then raw bytes.
- `matrix<T>` keeps its dimensions as **negative** longs (positive values mean the legacy format).
  The elements follow in row-major order.
- The reader compares each container length with the remaining bytes before it allocates memory.
  Each error gives the byte offset and a context path, for example
  `shape_predictor > forests > [7] > [319] > leaf_values`.

DNN files do not describe their own structure. `add_layer`, `add_tag_layer` and `add_skip_layer`
are different only in their C++ type. For this reason, a `NetSpec` controls the parser. The
`NetSpec` is a copy of the template type: `face::recognition::resnet_v1_spec()` is a copy of
`anet_type`.

- In the stream, the details of each layer come after its subnet.
- `resizable_tensor` contains raw little-endian `f32`, not `float_details`.
- An `affine_` slot also accepts `bn_con2` and `bn_fc2`. The parser converts them in `f32`:
  `g = gamma / sqrt(var + (float)eps)` and `b = beta - g * mean`. The ResNet model contains
  `affine_`, so only synthetic tests examine this conversion.

The library contains the frontal face detector as the original dlib serialized bytes
(`data/frontal_face_detector.dat`). `tools/cpp_probe/extract_frontal_detector.cpp` extracted these
bytes. The dlib source code keeps the detector in base64, compressed with
`compress_stream::kernel_1ea`. With the extracted bytes, it is not necessary to port that
compressor. When the detector loads, the dims must be 10·10·31 = 3100, and each weight vector
must be longer than that.

## shape_predictor (`face/shape_predictor.rs`)

- The conversion from RGB to gray is the unweighted integer mean `(r + g + b) / 3`
  (`pixel.h:968`).
- `unnormalizing_tform(rect)` uses `find_affine_transform` to map (0,0), (1,0) and (1,1) to the
  corners of the rectangle. dlib keeps those corners as `vector<float,2>`, so they go through an
  `f32` round trip.
- In each cascade, `find_tform_between_shapes` runs in `f64` on the `f32` shapes. Then its 2×2
  matrix is cast to `float`.
- A feature pixel is `round(tform_to_img(tform * delta + current[anchor]))`. The inner part is in
  `f32`, the mapping is in `f64`, and the rounding is `floor(x + 0.5)`. Pixels outside the image
  are 0.
- A tree goes left if and only if `px[idx1] - px[idx2] > thresh` (in `f32`). The leaf deltas are
  added in the order of the trees.
- Landmarks are integers, so parity here means exact equality.

## Face chips (`face/chip.rs`, `image/`)

- `get_face_chip_details`, 5 points: the mean 5-point face is mapped with
  `(padding + p) / (2 * padding + 1)`. This is a matrix expression, so the division is a
  multiplication by the reciprocal. Then the result is multiplied by `size`.
- `get_face_chip_details`, 68 points: the calculation uses parts 17–67, **without** the eyebrows
  (17–26) and the lower lip (55–59 and 65–67). It uses a true division.
- `chip_details` keeps `angle = atan2(p.y, p.x)` and `scale = |p|` for `p = m * (1, 0)`.
- In Python, `get_face_chip(s)` takes `padding` as a C++ `float`, and `get_face_chip_details`
  takes a `double`. The defaults are 150 / 0.25 for `get_face_chip` and 200 / 0.2 for
  `get_face_chip_details`.
- When `angle == 0` and the size is the same, `extract_image_chip` is a plain copy of the
  `lround` rectangle. In all other cases, it uses `extract_image_chips`, also for one chip.
- **Strong downscaling:** dlib makes a `pyramid_down<2>`:
  - It uses the integer Gaussian `[1 4 6 4 1]` with a truncating `/256`.
  - The output is `((nr-3)/2) × ((nc-3)/2)`, and `point_down = p/2 - (1.25, 0.75)`.
  - The pyramid covers **one bounding box for all chips in the batch**. As a result,
    `get_face_chips` on a batch can give pixels that are different from `get_face_chip` on each
    face. manyfaced does
    the same.
- `interpolate_bilinear` gives black when `floor(x) + 1 >= nc` or `floor(y) + 1 >= nr`. It
  calculates in `double` and **truncates** to uint8. It does not round.
- A chip can have a maximum of 2^28 pixels (16384 × 16384). A larger chip causes `ValueError`.
- Rectangles or angles that are not finite cause `ValueError`. In dlib, they cause an infinite
  loop in the search for the pyramid level. Chips with 0 rows or 0 columns are empty. In dlib,
  they cause an assertion failure.

## Face recognition (`face/recognition.rs`, `dnn/`)

- Input: planar R, G and B planes, calculated as `(pixel - avg) / 256`. The averages are
  (122.782, 117.001, 104.298). dlib divides in `double`. A division by a power of two is exact,
  so `f32` gives the same result.
- `con_` is im2col with taps in (k, y, x) order, and `filters × trans(cols)` through `sgemm`. The
  default padding is 0 for stride 2 and size/2 for stride 1.
- **`add_prev_` with shapes that are not the same** uses the elementwise maximum shape. It fills
  the missing values with zeros. This occurs in each `ares_down` block: at size 8, the stride-2
  path gives 3×3 and the pooling path gives 4×4.
- The pooling windows are clipped to the input. An average is the float sum in row order, divided
  by the number of clipped values.
- All Python overloads accept only RGB. A gray image causes `TypeError`. The chips of all faces in
  an image are extracted together.
- Jitter (`num_jitters > 1`) uses the stateful `dlib::rand` member. For this reason, the dlib model
  is not thread-safe. The manyfaced model has no state. Its jitter follows `jitter_image`, but it
  uses its own random generator with a seed. The jitter results are deterministic, but they are
  not bit-identical to the dlib results.
- Descriptors are `float64` arrays that contain the exact `float32` values, as with the dlib
  `matrix_cast<double>`. The Accelerate path (macOS) and the `gemm` path are different by
  approximately 1e-6.

## HOG detector (`detect/`)

- FHOG has 31 planes, 8-pixel cells and a 0.2 clamp. It **has padding of the filter size**
  (rows + 9, cols + 9), with a zero border.
- dlib calculates 8 columns at a time with SIMD, and then completes each row in scalar code. The
  two paths are different:
  - RGB gradient ties go to the later channel in the SIMD path, and to the earlier channel in the
    scalar path.
  - The bilinear products have a different order.
  - NEON lane sums are `(v2+v0)+(v3+v1)`.
- `pyramid_down<6>` gives `5*n/6` with bilinear resizing. The gray path rounds (+0.5) and the RGB
  SIMD path truncates.
- `pyramid_up` (`upsample`) uses the `pyramid_down<2>` geometry and gives (2h+1)×(2w+2).
  `rect_down` maps the rectangles back.
- The detector uses separable filters, unless the rank is more than `31·min(h,w)/3`. The frontal
  detector always uses the separable path. The manyfaced `svd4` gives the same components as
  LAPACK, but the signs can be different. The signs cancel exactly.
- The score is `saliency - w[dims]`.
- NMS (`test_box_overlap`) uses `inner / area(bounding box of a ∪ b)`, not the true union, with
  a strict `>`. The final sort is stable for ties. The dlib `std::sort` is not stable, but the
  goldens have no ties.
- One 1-ulp difference has no known cause. It occurs in 13 of 3780 saliency values of one pyramid
  level on `faces/2008_001322.jpg`. It does not change the outputs: the score difference is 0.
- The library does not accept detectors with `cell_size == 1`, because dlib has a different
  implementation for them.
- Limits for detector files that dlib does not have:
  - A minimum pyramid layer size of 0 is an error. dlib asserts this in
    `set_min_pyramid_layer_size`.
  - The FHOG window can have a maximum of 4096 cells on each side.
  - The pyramid stops at the first level that is the same as the level before it (for example,
    1×1). dlib makes copies of that level up to `max_pyramid_levels`. Real detectors stop much
    earlier, at their minimum layer size.

## Clustering (`cluster/`)

- `dlib::rand` is MT19937 with seed 5489. The constructor advances it by 10 000 draws.
  `get_random_64bit_number() = (a << 32) | b`.
- `chinese_whispers`:
  - The edges are made symmetric (self-loops one time) and sorted.
  - It does `num_nodes * iterations` steps. Each step selects node `rnd % num_nodes` and takes the
    neighbour label with the largest sum of weights. A tie goes to the smallest label
    (`std::map`, strict `>`).
  - The labels are numbered again in the order in which they first occur.
- `chinese_whispers_clustering(descriptors, t)` adds an edge for each `j >= i`, self-pairs
  included, when `length(a - b) < t` in `double`. `t` is a C++ `float`.

## dlib binding behaviour in `manyfaced.compat.dlib`

These behaviours are the same as in dlib:

- `rectangle(drectangle)` uses `lround`.
- `dcenter()` gives a `point` rounded with `floor(x + 0.5)`.
- `rectangle + point/rectangle` is the bounding union.
- Containers print the C++ `operator<<` of their elements.
- A model that does not load causes `RuntimeError`.
- face_recognition makes a `cnn_face_detection_model_v1` when you import it. For this reason, there
  is a placeholder.

These behaviours are different from dlib, intentionally:

- Only 8-bit images.
- `point` compares by value. dlib compares by identity.
- Lists are accepted where dlib requires `full_object_detections`.
- You cannot pickle the models (`shape_predictor`, `fhog_object_detector`,
  `face_recognition_model_v1`). dlib can pickle them.
- `load_rgb_image` uses Pillow to decode the image. As a result, the pixels can be slightly
  different from the pixels of the dlib JPEG decoder.

## Design decisions and their reasons

- **x86-64-v3 baseline** (`.cargo/config.toml`):
  - On the plain x86-64 baseline, `f32::mul_add` compiles to a call to the libm `fmaf`. This call
    takes 14–30% of the detector time. With v3, it is one `vfmadd` instruction, with no
    multiversioning and no fallback.
  - On older CPUs, `python/manyfaced/_cpu.py` raises `ImportError`, so that the CPU does not get
    an illegal instruction.
  - `MANYFACED_SKIP_CPU_CHECK=1` disables the check. This is necessary in Rosetta, because its
    CPUID does not show AVX2 and FMA.
- **Accelerate on macOS** (cargo feature `accelerate`, on by default, forwarded by
  `manyfaced-py`):
  - `gemm` runs at almost the NEON peak of one core. The Accelerate `sgemm` uses the AMX units
    and is approximately 2.7× faster on the ResNet.
  - On other platforms, `gemm` 0.19 is the only path. `--no-default-features` tests it on macOS
    too.
  - If a dimension is too large for a C `int`, the code uses `gemm`.
  - With many Python threads, the Accelerate threads can cause more threads than CPU cores. For
    this reason, the benchmarks set `VECLIB_MAXIMUM_THREADS=1`.
- **Models in the wheel:**
  - The 5-point predictor and the ResNet are public domain. They add approximately 28 MB to each
    wheel.
  - The wheel does not contain the 68-point model, because it is 100 MB and its training data is
    non-commercial.
  - maturin ignores the files that `.gitignore` lists. For this reason, `tool.maturin.include`
    lists `python/manyfaced/_models/`. Run `tools/download_models.py --bundle` before each
    build.
  - `_model_files.py` is the only source of the model names, URLs and SHA-256 hashes.
- **Golden data:**
  - The test images are the dlib examples. The repository does not contain them, because of their
    licenses (PASCAL VOC and photographer licenses).
  - The chip pixels are crops of those images. As a result, the repository contains only their
    digests. The pixels are in `tests/golden/local/`. `gen_golden.py chips` makes them again.
  - Pillow decodes the images. Each golden file keeps the SHA-256 of each decoded image. If a
    different libjpeg gives different pixels, `test_golden_inputs.py` fails. CI pins Pillow
    13.0.0.
- **No `panic = "abort"`:** PyO3 catches the unwind of a panic and changes it into a
  `PanicException`. An abort stops the interpreter.
- **Python 3.11 or later (abi3-py311):** the reason is the support lifecycle, not speed. The call
  overhead is the same within a few ns. The limited API has no free-threaded variant.
- **C++ probes without CMake:** a plain `c++` command builds each probe in `tools/cpp_probe/`.
  It uses the dlib headers and the dlib source files that the probe needs (usually
  `dlib/all/source.cpp`). The build command is at the top of each file. The probes write the
  intermediate values of dlib. Each probe has variants: `-ffp-contract=on/off`, with and without
  BLAS/LAPACK.

## Open items

- The CI workflows have not run on GitHub.
- It is not known if dlib-bin on the GitHub macOS runners gives bit-identical results to the
  local goldens. These runners have a different Apple CPU generation and different Accelerate
  kernels.
- There are no measurements on native x86_64 hardware at this time. Rosetta measurements are not
  meaningful.
- `-C target-cpu=neoverse-n1` for aarch64 Linux makes the detector approximately 3% faster. But
  it stops support for armv8.0 CPUs, for example the Raspberry Pi 4 and Graviton 1. This flag is
  not used.
