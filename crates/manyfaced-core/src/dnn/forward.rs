//! CPU inference for loaded dlib networks.
//!
//! Ported from the forward passes in `dlib/dnn/layers.h`, `input.h` and the CPU kernels in
//! `dlib/cuda/cpu_dlib.cpp` (dlib v20.0.1, Boost Software License 1.0, see
//! `THIRD_PARTY_NOTICES`). Samples are independent at inference time (affine layers replace
//! batch norm), so a batch is evaluated one sample at a time, like dlib's CPU convolution.
//!
//! The heavy layers have three implementations: [`Backend::Gemm`] (im2col + the pure-Rust `gemm`
//! crate, the default except on macOS), `Backend::Accelerate` (im2col + `cblas_sgemm` from the
//! macOS system framework, the default on macOS with the `accelerate` feature) and
//! [`Backend::Naive`] (direct loops, kept as an obviously-correct reference for tests). All
//! accumulate in `f32` like dlib, but in different orders, so results agree to within float
//! rounding, not bit for bit (dlib itself uses BLAS `sgemm`, whose order is implementation
//! defined).

use super::network::{Affine, AffineMode, Conv, Fc, Layer, Network, Pool, RgbInput};
use crate::image::RgbView;

/// A single sample: `k` channels of `nr × nc`, channel-major (`(k, r, c)`, row-major planes).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub k: usize,
    pub nr: usize,
    pub nc: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(k: usize, nr: usize, nc: usize) -> Self {
        Tensor {
            k,
            nr,
            nc,
            data: vec![0.0; k * nr * nc],
        }
    }

    fn plane(&self, k: usize) -> &[f32] {
        let n = self.nr * self.nc;
        &self.data[k * n..(k + 1) * n]
    }
}

/// How convolutions and fully connected layers are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// im2col + the pure-Rust `gemm` crate (runtime SIMD dispatch).
    Gemm,
    /// im2col + `cblas_sgemm` from Apple's Accelerate framework (uses the AMX units on Apple
    /// Silicon). Same algorithm as dlib-bin on macOS, which also links Accelerate.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    Accelerate,
    /// Direct loops: slow, obviously correct reference.
    Naive,
}

impl Default for Backend {
    /// `Accelerate` on macOS when the `accelerate` feature is on (the default), `Gemm` elsewhere.
    fn default() -> Self {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        return Backend::Accelerate;
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        return Backend::Gemm;
    }
}

/// Errors when running a network on an input it cannot process.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ForwardError {
    #[error("the network expects a {expected_rows}x{expected_cols} RGB image, got {rows}x{cols}")]
    WrongInputSize {
        expected_rows: usize,
        expected_cols: usize,
        rows: usize,
        cols: usize,
    },
    #[error("layer {layer}: {msg}")]
    Shape { layer: usize, msg: String },
    #[error("the network does not end in a 1x1 output")]
    NotAVector,
}

impl RgbInput {
    /// `input_rgb_image_sized::to_tensor`: planar R, G, B with `(pixel - avg) / 256`.
    /// dlib computes `(uint8 - float)` in `float`, divides by `256.0` in `double` and stores a
    /// `float`; dividing by a power of two is exact, so doing it in `f32` is identical.
    pub fn to_tensor(&self, img: RgbView<'_>) -> Result<Tensor, ForwardError> {
        if img.height() != self.nr || img.width() != self.nc {
            return Err(ForwardError::WrongInputSize {
                expected_rows: self.nr,
                expected_cols: self.nc,
                rows: img.height(),
                cols: img.width(),
            });
        }
        let mut t = Tensor::zeros(3, self.nr, self.nc);
        let plane = self.nr * self.nc;
        let avg = [self.avg_red, self.avg_green, self.avg_blue];
        for r in 0..self.nr {
            let row = img.row(r);
            for c in 0..self.nc {
                for ch in 0..3 {
                    t.data[ch * plane + r * self.nc + c] =
                        (f32::from(row[3 * c + ch]) - avg[ch]) / 256.0;
                }
            }
        }
        Ok(t)
    }
}

impl Network {
    /// Runs the network on one input tensor and returns the flattened output of the last layer
    /// (for `loss_metric`, the embedding itself: dlib does not normalize it).
    pub fn forward(&self, input: Tensor, backend: Backend) -> Result<Vec<f32>, ForwardError> {
        let mut x = input;
        // Outputs of tag layers by id. A later tag with the same id shadows an earlier one,
        // which matches dlib's "nearest tag below" lookup.
        let mut tags: Vec<(u32, Tensor)> = Vec::new();
        let shape_err = |layer: usize, msg: String| ForwardError::Shape { layer, msg };
        // im2col buffer reused by every convolution of this call.
        let mut cols = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            x = match layer {
                Layer::Con(c) => conv(&x, c, backend, &mut cols).map_err(|m| shape_err(i, m))?,
                Layer::Affine(a) => {
                    affine(&mut x, a).map_err(|m| shape_err(i, m))?;
                    x
                }
                Layer::Relu { disabled } => {
                    // A disabled in-place layer returns without touching its output.
                    if !disabled {
                        relu(&mut x.data);
                    }
                    x
                }
                Layer::MaxPool(p) => pool(&x, p, true).map_err(|m| shape_err(i, m))?,
                Layer::AvgPool(p) => pool(&x, p, false).map_err(|m| shape_err(i, m))?,
                Layer::Fc(f) => fc(&x, f, backend).map_err(|m| shape_err(i, m))?,
                Layer::Tag(id) => {
                    match tags.iter_mut().find(|(t, _)| t == id) {
                        Some(slot) => slot.1 = x.clone(),
                        None => tags.push((*id, x.clone())),
                    }
                    x
                }
                Layer::Skip(id) => find_tag(&tags, *id).map_err(|m| shape_err(i, m))?.clone(),
                Layer::AddPrev { tag } => {
                    let other = find_tag(&tags, *tag).map_err(|m| shape_err(i, m))?;
                    add_zero_padded(&x, other)
                }
            };
        }
        if x.nr != 1 || x.nc != 1 {
            return Err(ForwardError::NotAVector);
        }
        Ok(x.data)
    }
}

fn find_tag(tags: &[(u32, Tensor)], id: u32) -> Result<&Tensor, String> {
    tags.iter()
        .find(|(t, _)| *t == id)
        .map(|(_, x)| x)
        .ok_or_else(|| format!("no tag{id} below this layer"))
}

/// Output size of a convolution/pooling window, `1 + (n + 2p - w) / s`, or an error when the
/// window does not fit (dlib asserts).
fn out_size(n: usize, pad: usize, window: usize, stride: usize) -> Result<usize, String> {
    if stride == 0 || window == 0 || window > n + 2 * pad {
        return Err(format!(
            "window {window} (padding {pad}, stride {stride}) does not fit an input of {n}"
        ));
    }
    Ok(1 + (n + 2 * pad - window) / stride)
}

/// `tensor_conv` + bias (+ fused relu): `out = filters * trans(img2col(x)) + bias`.
fn conv(x: &Tensor, c: &Conv, backend: Backend, cols: &mut Vec<f32>) -> Result<Tensor, String> {
    if x.k != c.k {
        return Err(format!("conv expects {} input channels, got {}", c.k, x.k));
    }
    let out_nr = out_size(x.nr, c.padding_y, c.nr, c.stride_y)?;
    let out_nc = out_size(x.nc, c.padding_x, c.nc, c.stride_x)?;
    let mut out = Tensor::zeros(c.num_filters, out_nr, out_nc);
    match backend {
        Backend::Naive => conv_naive(x, c, &mut out),
        fast => conv_gemm(x, c, &mut out, cols, fast),
    }
    if let Some(bias) = &c.bias {
        let n = out_nr * out_nc;
        for (plane, &b) in out.data.chunks_exact_mut(n).zip(bias) {
            for v in plane {
                *v += b;
            }
        }
    }
    if c.use_relu {
        relu(&mut out.data);
    }
    Ok(out)
}

/// The input value seen by filter tap (`k`, `y`, `x`) of output pixel (`r`, `c`), zero outside
/// the image (dlib's img2col).
#[inline]
fn tap(x: &Tensor, k: usize, yy: isize, xx: isize) -> f32 {
    if yy < 0 || xx < 0 || yy as usize >= x.nr || xx as usize >= x.nc {
        0.0
    } else {
        x.data[(k * x.nr + yy as usize) * x.nc + xx as usize]
    }
}

fn conv_naive(x: &Tensor, c: &Conv, out: &mut Tensor) {
    let taps = c.k * c.nr * c.nc;
    for f in 0..c.num_filters {
        let w = &c.weights[f * taps..(f + 1) * taps];
        for r in 0..out.nr {
            for col in 0..out.nc {
                let r0 = (r * c.stride_y) as isize - c.padding_y as isize;
                let c0 = (col * c.stride_x) as isize - c.padding_x as isize;
                let mut acc = 0.0f32;
                let mut t = 0;
                for k in 0..c.k {
                    for y in 0..c.nr {
                        for xk in 0..c.nc {
                            acc += w[t] * tap(x, k, r0 + y as isize, c0 + xk as isize);
                            t += 1;
                        }
                    }
                }
                out.data[(f * out.nr + r) * out.nc + col] = acc;
            }
        }
    }
}

/// Builds the `(k * nr * nc) × (out_nr * out_nc)` im2col matrix, row-major, rows in dlib's tap
/// order `(k, y, x)`. Every element is written, so `cols` can be a reused buffer.
fn im2col(x: &Tensor, c: &Conv, out_nr: usize, out_nc: usize, cols: &mut Vec<f32>) {
    let pixels = out_nr * out_nc;
    cols.resize(c.k * c.nr * c.nc * pixels, 0.0);
    let mut row = 0;
    for k in 0..c.k {
        let plane = x.plane(k);
        for y in 0..c.nr {
            for xk in 0..c.nc {
                let dst = &mut cols[row * pixels..(row + 1) * pixels];
                // Output columns whose input column xx = oc * stride + xk - pad is inside.
                let first = c.padding_x.saturating_sub(xk).div_ceil(c.stride_x);
                let last = if x.nc + c.padding_x > xk {
                    ((x.nc + c.padding_x - xk - 1) / c.stride_x + 1).min(out_nc)
                } else {
                    0
                };
                for r in 0..out_nr {
                    let dst_row = &mut dst[r * out_nc..(r + 1) * out_nc];
                    let yy = (r * c.stride_y + y) as isize - c.padding_y as isize;
                    if yy < 0 || yy as usize >= x.nr || first >= last {
                        dst_row.fill(0.0);
                        continue;
                    }
                    let src = &plane[yy as usize * x.nc..(yy as usize + 1) * x.nc];
                    dst_row[..first].fill(0.0);
                    dst_row[last..].fill(0.0);
                    let x0 = first * c.stride_x + xk - c.padding_x;
                    if c.stride_x == 1 {
                        dst_row[first..last].copy_from_slice(&src[x0..x0 + (last - first)]);
                    } else {
                        for (d, s) in dst_row[first..last]
                            .iter_mut()
                            .zip(src[x0..].iter().step_by(c.stride_x))
                        {
                            *d = *s;
                        }
                    }
                }
                row += 1;
            }
        }
    }
}

/// `true` if `buf` holds at least `rows * cols` elements. The unsafe matrix products depend on
/// this check, so the product must not wrap.
fn holds(buf: &[f32], rows: usize, cols: usize) -> bool {
    rows.checked_mul(cols).is_some_and(|n| buf.len() >= n)
}

/// `dst (m×n) = lhs (m×k) * rhs (k×n)`, all row-major and contiguous, with the given fast
/// backend (`Naive` falls back to `Gemm`; callers handle it themselves).
fn matmul(
    backend: Backend,
    m: usize,
    n: usize,
    k: usize,
    lhs: &[f32],
    rhs: &[f32],
    dst: &mut [f32],
) {
    assert!(holds(lhs, m, k) && holds(rhs, k, n) && holds(dst, m, n));
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        dst[..m * n].fill(0.0);
        return;
    }
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if backend == Backend::Accelerate && accelerate::sgemm(m, n, k, lhs, rhs, dst) {
        return;
    }
    let _ = backend;
    #[cfg(feature = "rayon")]
    let parallelism = gemm::Parallelism::Rayon(0);
    #[cfg(not(feature = "rayon"))]
    let parallelism = gemm::Parallelism::None;
    // SAFETY: `gemm` only has a raw-pointer API. The assertion above guarantees that every
    // element addressed by the given dimensions and strides (row stride = number of columns,
    // column stride = 1) lies inside the three slices, which do not alias (`dst` is `&mut`).
    // `read_dst = false`, so `dst` is write-only and may hold anything.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            1,
            n as isize,
            false,
            lhs.as_ptr(),
            1,
            k as isize,
            rhs.as_ptr(),
            1,
            n as isize,
            0.0,
            1.0,
            false,
            false,
            false,
            parallelism,
        );
    }
}

fn conv_gemm(x: &Tensor, c: &Conv, out: &mut Tensor, cols: &mut Vec<f32>, backend: Backend) {
    im2col(x, c, out.nr, out.nc, cols);
    let taps = c.k * c.nr * c.nc;
    matmul(
        backend,
        c.num_filters,
        out.nr * out.nc,
        taps,
        &c.weights,
        cols,
        &mut out.data,
    );
}

/// `affine_transform_conv` / `affine_transform`: `out = gamma * x + beta`.
fn affine(x: &mut Tensor, a: &Affine) -> Result<(), String> {
    if a.disabled {
        return Ok(());
    }
    let plane = x.nr * x.nc;
    match a.mode {
        AffineMode::Conv => {
            if a.gamma.len() != x.k {
                return Err(format!(
                    "affine has {} channels, input has {}",
                    a.gamma.len(),
                    x.k
                ));
            }
            for ((p, &g), &b) in x
                .data
                .chunks_exact_mut(plane.max(1))
                .zip(&a.gamma)
                .zip(&a.beta)
            {
                for v in p {
                    *v = g * *v + b;
                }
            }
        }
        AffineMode::Fc => {
            if a.gamma.len() != x.data.len() {
                return Err(format!(
                    "affine has {} values, input has {}",
                    a.gamma.len(),
                    x.data.len()
                ));
            }
            for ((v, &g), &b) in x.data.iter_mut().zip(&a.gamma).zip(&a.beta) {
                *v = g * *v + b;
            }
        }
    }
    Ok(())
}

/// `relu`: `lowerbound(x, 0)`.
fn relu(data: &mut [f32]) {
    for v in data {
        // A select (not a conditional store) so the loop vectorizes; NaN stays NaN like dlib.
        *v = if *v < 0.0 { 0.0 } else { *v };
    }
}

/// dlib's `pooling`: each output takes the max/mean of the window
/// `centered_rect(c*stride + w/2 - pad, r*stride + h/2 - pad, w, h)` clipped to the input.
/// A size of 0 means the whole input (`avg_pool_everything`).
fn pool(x: &Tensor, p: &Pool, max: bool) -> Result<Tensor, String> {
    let wh = if p.nr == 0 { x.nr } else { p.nr };
    let ww = if p.nc == 0 { x.nc } else { p.nc };
    let out_nr = out_size(x.nr, p.padding_y, wh, p.stride_y)?;
    let out_nc = out_size(x.nc, p.padding_x, ww, p.stride_x)?;
    let mut out = Tensor::zeros(x.k, out_nr, out_nc);
    for k in 0..x.k {
        let plane = x.plane(k);
        for r in 0..out_nr {
            // centered_rect(x, y, w, h).left() = x - w/2 = c*stride - pad.
            let top = (r * p.stride_y) as isize - p.padding_y as isize;
            let y0 = top.max(0) as usize;
            let y1 = ((top + wh as isize).min(x.nr as isize)).max(0) as usize;
            for c in 0..out_nc {
                let left = (c * p.stride_x) as isize - p.padding_x as isize;
                let x0 = left.max(0) as usize;
                let x1 = ((left + ww as isize).min(x.nc as isize)).max(0) as usize;
                let value = if y0 >= y1 || x0 >= x1 {
                    // dlib would reduce an empty matrix; never happens for valid windows.
                    0.0
                } else if max {
                    let mut m = f32::NEG_INFINITY;
                    for yy in y0..y1 {
                        for &v in &plane[yy * x.nc + x0..yy * x.nc + x1] {
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    m
                } else {
                    // mean(): float sum in row-major order, divided by the element count.
                    let mut s = 0.0f32;
                    for yy in y0..y1 {
                        for &v in &plane[yy * x.nc + x0..yy * x.nc + x1] {
                            s += v;
                        }
                    }
                    s / ((y1 - y0) * (x1 - x0)) as f32
                };
                out.data[(k * out_nr + r) * out_nc + c] = value;
            }
        }
    }
    Ok(out)
}

/// `fc_`: `out = flatten(x) * weights` with `weights` of shape `(num_inputs, num_outputs)`.
fn fc(x: &Tensor, f: &Fc, backend: Backend) -> Result<Tensor, String> {
    if x.data.len() != f.num_inputs {
        return Err(format!(
            "fc expects {} inputs, got {}",
            f.num_inputs,
            x.data.len()
        ));
    }
    let mut out = Tensor::zeros(f.num_outputs, 1, 1);
    match backend {
        Backend::Naive => {
            for (o, dst) in out.data.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (i, &v) in x.data.iter().enumerate() {
                    acc += v * f.weights[i * f.num_outputs + o];
                }
                *dst = acc;
            }
        }
        fast => matmul(
            fast,
            1,
            f.num_outputs,
            f.num_inputs,
            &x.data,
            &f.weights,
            &mut out.data,
        ),
    }
    if let Some(bias) = &f.bias {
        for (v, &b) in out.data.iter_mut().zip(bias) {
            *v += b;
        }
    }
    Ok(out)
}

/// `tt::add(dest, src1, src2)` for tensors of possibly different shapes: `dest` takes the
/// elementwise maximum shape and missing values count as 0 (cpu_dlib.cpp `add`). In dlib's face
/// ResNet this happens in every `ares_down` block, where the strided conv path is smaller than
/// the average-pooled skip path and has more channels.
fn add_zero_padded(a: &Tensor, b: &Tensor) -> Tensor {
    if a.k == b.k && a.nr == b.nr && a.nc == b.nc {
        let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
        return Tensor {
            k: a.k,
            nr: a.nr,
            nc: a.nc,
            data,
        };
    }
    let (k, nr, nc) = (a.k.max(b.k), a.nr.max(b.nr), a.nc.max(b.nc));
    let get = |t: &Tensor, kk: usize, r: usize, c: usize| {
        if kk < t.k && r < t.nr && c < t.nc {
            t.data[(kk * t.nr + r) * t.nc + c]
        } else {
            0.0
        }
    };
    let mut out = Tensor::zeros(k, nr, nc);
    let mut i = 0;
    for kk in 0..k {
        for r in 0..nr {
            for c in 0..nc {
                out.data[i] = get(a, kk, r, c) + get(b, kk, r, c);
                i += 1;
            }
        }
    }
    out
}

/// FFI to the `cblas_sgemm` of Apple's Accelerate framework, which ships with every macOS
/// (nothing is bundled with manyfaced).
#[cfg(all(feature = "accelerate", target_os = "macos"))]
mod accelerate {
    use std::os::raw::c_int;

    use super::holds;

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }

    /// `dst (m×n) = lhs (m×k) * rhs (k×n)`, row-major, contiguous, all dimensions > 0.
    /// Returns `false` (and does nothing) if a dimension does not fit in a C `int`, so the
    /// caller can fall back to the pure-Rust path.
    pub(super) fn sgemm(
        m: usize,
        n: usize,
        k: usize,
        lhs: &[f32],
        rhs: &[f32],
        dst: &mut [f32],
    ) -> bool {
        assert!(holds(lhs, m, k) && holds(rhs, k, n) && holds(dst, m, n));
        let (Ok(mi), Ok(ni), Ok(ki)) = (c_int::try_from(m), c_int::try_from(n), c_int::try_from(k))
        else {
            return false;
        };
        // SAFETY: the assertion guarantees the row-major m×k, k×n and m×n matrices with leading
        // dimensions k, n and n lie inside the three slices; `dst` is `&mut`, so it does not
        // alias the inputs; with beta = 0 `dst` is write-only. cblas_sgemm has no other
        // requirements and does not retain the pointers.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                mi,
                ni,
                ki,
                1.0,
                lhs.as_ptr(),
                ki,
                rhs.as_ptr(),
                ni,
                0.0,
                dst.as_mut_ptr(),
                ni,
            );
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv_layer(k: usize, f: usize, size: usize, stride: usize, pad: usize) -> Conv {
        let taps = k * size * size;
        Conv {
            num_filters: f,
            k,
            nr: size,
            nc: size,
            stride_y: stride,
            stride_x: stride,
            padding_y: pad,
            padding_x: pad,
            weights: (0..f * taps)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
                .collect(),
            bias: Some((0..f).map(|i| i as f32 * 0.25).collect()),
            use_relu: false,
        }
    }

    fn input(k: usize, nr: usize, nc: usize) -> Tensor {
        Tensor {
            k,
            nr,
            nc,
            data: (0..k * nr * nc)
                .map(|i| ((i * 31 % 17) as f32 - 8.0) / 4.0)
                .collect(),
        }
    }

    #[test]
    fn conv_backends_agree_and_match_hand_computation() {
        // 1 channel, 1 filter of ones, 3x3 padding 1: each output is the 3x3 neighbourhood sum.
        let mut c = conv_layer(1, 1, 3, 1, 1);
        c.weights = vec![1.0; 9];
        c.bias = None;
        let x = Tensor {
            k: 1,
            nr: 2,
            nc: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        for backend in [Backend::Naive, Backend::Gemm, Backend::default()] {
            assert_eq!(
                conv(&x, &c, backend, &mut Vec::new()).unwrap().data,
                [10.0; 4]
            );
        }
        for (k, f, size, stride, pad, n) in
            [(3, 4, 7, 2, 0, 20), (5, 3, 3, 1, 1, 9), (2, 6, 3, 2, 0, 8)]
        {
            let c = conv_layer(k, f, size, stride, pad);
            let x = input(k, n, n + 1);
            let a = conv(&x, &c, Backend::Naive, &mut Vec::new()).unwrap();
            // A dirty reused buffer must not leak into the result.
            let mut dirty = vec![f32::NAN; 100_000];
            let b = conv(&x, &c, Backend::Gemm, &mut dirty).unwrap();
            assert_eq!((a.k, a.nr, a.nc), (b.k, b.nr, b.nc));
            for (u, v) in a.data.iter().zip(&b.data) {
                assert!((u - v).abs() <= 1e-4 * u.abs().max(1.0), "{u} vs {v}");
            }
        }
    }

    #[test]
    fn output_sizes_follow_dlib() {
        let c = conv_layer(1, 1, 3, 2, 0);
        assert_eq!(
            conv(&input(1, 35, 35), &c, Backend::Gemm, &mut Vec::new())
                .unwrap()
                .nr,
            17
        );
        let p = Pool {
            nr: 2,
            nc: 2,
            stride_y: 2,
            stride_x: 2,
            padding_y: 0,
            padding_x: 0,
        };
        assert_eq!(pool(&input(1, 8, 8), &p, false).unwrap().nr, 4);
        assert_eq!(
            conv(&input(1, 8, 8), &c, Backend::Gemm, &mut Vec::new())
                .unwrap()
                .nr,
            3
        );
        assert!(conv(&input(1, 2, 2), &c, Backend::Gemm, &mut Vec::new()).is_err());
    }

    #[test]
    fn pooling_values() {
        let x = Tensor {
            k: 1,
            nr: 3,
            nc: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let p = Pool {
            nr: 2,
            nc: 2,
            stride_y: 1,
            stride_x: 1,
            padding_y: 0,
            padding_x: 0,
        };
        assert_eq!(pool(&x, &p, true).unwrap().data, [5.0, 6.0, 8.0, 9.0]);
        assert_eq!(pool(&x, &p, false).unwrap().data, [3.0, 4.0, 6.0, 7.0]);
        let everything = Pool { nr: 0, nc: 0, ..p };
        assert_eq!(pool(&x, &everything, false).unwrap().data, [5.0]);
        // Padding: the clipped window averages only the pixels inside the image.
        let padded = Pool {
            nr: 3,
            nc: 3,
            stride_y: 1,
            stride_x: 1,
            padding_y: 1,
            padding_x: 1,
        };
        assert_eq!(pool(&x, &padded, false).unwrap().data[0], 3.0);
    }

    #[test]
    fn mismatched_add_zero_pads() {
        let a = Tensor {
            k: 1,
            nr: 1,
            nc: 2,
            data: vec![1.0, 2.0],
        };
        let b = Tensor {
            k: 2,
            nr: 2,
            nc: 1,
            data: vec![10.0, 20.0, 30.0, 40.0],
        };
        let c = add_zero_padded(&a, &b);
        assert_eq!((c.k, c.nr, c.nc), (2, 2, 2));
        assert_eq!(c.data, [11.0, 2.0, 20.0, 0.0, 30.0, 0.0, 40.0, 0.0]);
    }

    #[test]
    fn fc_backends_agree() {
        let f = Fc {
            num_inputs: 5,
            num_outputs: 3,
            weights: (0..15).map(|i| i as f32 - 7.0).collect(),
            bias: None,
        };
        let x = input(5, 1, 1);
        let a = fc(&x, &f, Backend::Naive).unwrap();
        let b = fc(&x, &f, Backend::Gemm).unwrap();
        assert_eq!(a.data, b.data);
        assert!(fc(&input(4, 1, 1), &f, Backend::Gemm).is_err());
    }
}
