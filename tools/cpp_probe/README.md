# C++ probes

These small programs use `third_party/dlib` and print the values that dlib calculates. They are
not part of the build, the tests or CI. Use them when manyfaced and dlib give different results,
or to examine a new dlib release.

- The header of each file gives its usage and its build command (plain `c++`, no CMake).
- The probes write to `build/` and `out/`. `.gitignore` lists the two directories.

| Probe | Output |
|---|---|
| `extract_frontal_detector` | The serialized frontal face detector, byte for byte. This probe made `crates/manyfaced-core/data/frontal_face_detector.dat`. |
| `inspect_models` | The model structures and checksums from the dlib deserializer. The constants in `crates/manyfaced-core/tests/models.rs` come from it. Rust equivalent: `cargo run --release --example inspect_models`. |
| `hog_probe` | The detector intermediates (pyramid, FHOG, filters, saliency, detections) as `.npy`. Rust equivalent: `cargo run --release --example hog_dump`. |
| `hog_full_probe` | The saliency of the non-separable filter path. Rust equivalent: `cargo run --release --example hog_full_dump`. |
| `sp_probe` | The shape_predictor landmarks for a list of images and boxes. |
| `chip_probe` | `get_face_chip_details` with full precision. |

A dlib build option can change the results. To see the effect, build a probe in different
variants:

- With `-ffp-contract=on` or `-ffp-contract=off`.
- With or without `-DDLIB_USE_BLAS -DDLIB_USE_LAPACK -framework Accelerate`.

For the reasons, refer to [docs/internals.md](../../docs/internals.md).
