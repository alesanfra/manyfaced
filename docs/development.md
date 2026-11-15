# Development

## Requirements

- A Rust toolchain, version 1.85 or later.
- [uv](https://docs.astral.sh/uv/).

## Build and test

Do these steps from the root of the repository:

```bash
git clone --depth 1 --branch v20.0.1 https://github.com/davisking/dlib.git third_party/dlib
uv run --no-project tools/download_models.py --bundle  # models/ and the bundled copies
uv sync --locked                                        # dev venv, builds manyfaced with maturin
uv run --no-sync pytest
uv run --no-sync maturin develop --release --uv         # rebuild after a change to the Rust code
```

## Rules and tools

- `third_party/dlib` is the specification. Each algorithm is a port from that tag. Each port keeps
  the dlib order of floating point operations.
- The `golden` dependency group installs `dlib-bin`. Only `tools/gen_golden.py` (reference
  outputs) and `tools/bench.py` use it. manyfaced does not depend on dlib.
- `pytest -m parity` needs `third_party/dlib` and `models/`.
- The repository does not contain the face chip pixels, because they are crops of images with
  unclear licenses. To make them again, run `python tools/gen_golden.py chips`.
- `tools/cpp_probe/` contains small C++ programs. They use `third_party/dlib` and write the
  intermediate values of dlib. Use them when a component gives results that are different from
  dlib.
- [internals.md](internals.md) gives the dlib details that control parity, the differences
  between dlib builds and the reasons for the design decisions.

## Continuous integration

The workflows use a small number of Actions minutes. On private repositories, a macOS minute
costs 10 Linux minutes and a Windows minute costs 2 Linux minutes.

### CI

File: `.github/workflows/ci.yml`. It runs on each push to `main` and on each pull request. It
does not run when a change has only documentation.

1. One Linux job builds the abi3 wheel one time. It does these checks:
   - Lint, and clippy for the macOS and Windows on Arm targets.
   - The Rust tests and the MSRV check.
   - The Python tests on Python 3.11, on Python 3.15 and on the lowest dependencies.
2. If the Linux job passes, two jobs start:
   - `parity-macos` runs the Rust tests with the Accelerate backend. It makes the chip pixels
     with dlib-bin. Then it runs all of the Python tests, with the live comparisons with dlib.
   - `windows` runs the Python tests on Python 3.11 and 3.15.
3. `parity-linux` (x86_64 and aarch64) compares manyfaced with the chip pixels from
   `parity-macos`.

### Wheels

File: `.github/workflows/wheels.yml`. It runs on each push to `main` (except documentation-only
changes), on pull requests that change the packaging files and on manual runs. Each job builds
the wheels and does a smoke test on them. The jobs make these packages:

- manylinux 2.28 and musllinux 1.2, x86_64 and aarch64.
- macOS arm64.
- Windows x64, and Windows arm64 on the native `windows-11-arm` runner.
- The sdist.

### Release

File: `.github/workflows/release.yml`. It runs on `v*` tags and does not build again. It takes the
artifacts of the Wheels run on `main` for the tagged commit (it waits if the run is still going),
creates the GitHub Release with them and publishes them to PyPI with trusted publishing
(environment `pypi`).

To make a release:

1. Make sure that `version` in `Cargo.toml` is correct. maturin uses it as the Python version.
2. Push the commit to `main`.
3. Tag that commit and push the tag: `git tag v0.1.1 && git push origin v0.1.1`.

To publish an existing tag again (for example after a failed PyPI upload), start the workflow
manually: **Actions → Release → Run workflow**, with the tag as input. A GitHub Release that
already exists is kept.

Nothing is published to crates.io.
