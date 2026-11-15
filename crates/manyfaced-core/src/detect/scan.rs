//! `scan_fhog_pyramid`: the FHOG filter bank, dlib's float spatial filtering and the sliding
//! window search over an image pyramid.
//!
//! Ported from `dlib/image_processing/scan_fhog_pyramid.h` and
//! `dlib/image_transforms/spatial_filtering.h` (dlib v20.0.1, Boost Software License 1.0, see
//! `THIRD_PARTY_NOTICES`). dlib's float filters run 8 columns at a time with three interleaved
//! accumulators and finish each row in scalar code; both summation orders are reproduced.
//!
//! The scalar tails follow what the clang-built reference (`dlib-bin`, arm64,
//! `-ffp-contract=on`) actually computes, established with `tools/cpp_probe/hog_probe.cpp` and
//! `hog_full_probe.cpp`: the column pass of the separable filter fuses `temp += s * f` into an
//! FMA, while the row pass and the full filter (`p = ...; temp += p * f`) do not. A residual
//! 1-ulp difference remains on the first tail column of some planes (13 of 3780 saliency values
//! on the largest level of `faces/2008_001322.jpg`), from code generation we could not attribute;
//! it does not reach the detections (all golden cases are bit-exact).

use super::fhog::{Fhog, NUM_PLANES, fhog_to_image};
use crate::geometry::{Point, Rect};
use crate::linalg::{Mat, svd};

/// The filters of one detector: the full `height × width` filter of each FHOG plane and its
/// separable approximation (`build_fhog_filterbank`).
#[derive(Debug, Clone)]
pub struct FilterBank {
    pub height: usize,
    pub width: usize,
    /// Per plane, the `height × width` filter, row-major.
    pub filters: Vec<Vec<f32>>,
    /// Per plane, the separable components: row filters (length `width`) and column filters
    /// (length `height`), largest singular value first.
    pub row_filters: Vec<Vec<Vec<f32>>>,
    pub col_filters: Vec<Vec<Vec<f32>>>,
}

impl FilterBank {
    /// `build_fhog_filterbank(weights)`: each plane's filter is decomposed with an SVD, singular
    /// values below `max(1e-4, 0.001 * max(w))` are dropped, and the rest become
    /// `u_j * sqrt(w_j)` (column) and `v_j * sqrt(w_j)` (row) filters, cast to `float`.
    pub fn build(weights: &[f64], height: usize, width: usize) -> FilterBank {
        let size = height * width;
        let mut fb = FilterBank {
            height,
            width,
            filters: Vec::with_capacity(NUM_PLANES),
            row_filters: Vec::with_capacity(NUM_PLANES),
            col_filters: Vec::with_capacity(NUM_PLANES),
        };
        for i in 0..NUM_PLANES {
            let f = Mat::from_rows(height, width, weights[i * size..(i + 1) * size].to_vec());
            fb.filters.push(f.data.iter().map(|&v| v as f32).collect());
            let s = svd(&f);
            // rsort_columns(u, w) and rsort_columns(v, w): descending singular values.
            let mut order: Vec<usize> = (0..s.w.len()).collect();
            order.sort_by(|&a, &b| s.w[b].total_cmp(&s.w[a]));
            let max_w = s.w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let thresh = (max_w * 0.001).max(1e-4);
            let (mut rows, mut cols) = (Vec::new(), Vec::new());
            for &j in &order {
                let wj = s.w[j];
                // round_zeros(w, thresh)
                if !(wj >= thresh || wj <= -thresh) {
                    continue;
                }
                let sq = wj.sqrt();
                cols.push((0..s.u.rows).map(|r| (s.u.get(r, j) * sq) as f32).collect());
                rows.push((0..s.v.rows).map(|r| (s.v.get(r, j) * sq) as f32).collect());
            }
            fb.row_filters.push(rows);
            fb.col_filters.push(cols);
        }
        fb
    }

    pub fn num_separable_filters(&self) -> usize {
        self.row_filters.iter().map(Vec::len).sum()
    }

    /// dlib uses the full filters when the separable ones would not be faster.
    pub fn use_separable(&self) -> bool {
        let limit = (NUM_PLANES * self.height.min(self.width)) as f64 / 3.0;
        self.num_separable_filters() as f64 <= limit
    }
}

/// A `rows × cols` float image.
#[derive(Debug, Clone, Default)]
pub struct FloatImage {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl FloatImage {
    fn set_size(&mut self, rows: usize, cols: usize) {
        if self.rows != rows || self.cols != cols {
            self.rows = rows;
            self.cols = cols;
            self.data.clear();
            self.data.resize(rows * cols, 0.0);
        }
    }

    /// `zero_border_pixels(img, inside)`.
    fn zero_border(&mut self, inside: Rect) {
        let img = Rect::new(0, 0, self.cols as i64 - 1, self.rows as i64 - 1);
        let inside = Rect::new(
            inside.left.max(img.left),
            inside.top.max(img.top),
            inside.right.min(img.right),
            inside.bottom.min(img.bottom),
        );
        if inside.is_empty() {
            self.data.fill(0.0);
            return;
        }
        let cols = self.cols;
        for (r, row) in self.data.chunks_exact_mut(cols).enumerate() {
            let r = r as i64;
            if r < inside.top || r > inside.bottom {
                row.fill(0.0);
            } else {
                row[..inside.left as usize].fill(0.0);
                row[inside.right as usize + 1..].fill(0.0);
            }
        }
    }
}

/// A borrowed `rows × cols` float image.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneRef<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

/// Reusable buffers for the filters.
#[derive(Debug, Clone, Default)]
pub struct FilterScratch {
    image: FloatImage,
}

/// Number of columns, from `first_col`, that dlib's 8-wide loop
/// `for (c = first_col; c < last_col - 7; c += 8)` covers.
fn simd_cols(first_col: i64, last_col: i64) -> usize {
    let span = last_col - 7 - first_col;
    if span > 0 {
        (span as usize).div_ceil(8) * 8
    } else {
        0
    }
}

/// dlib's scalar tail `temp = 0; for n: temp += src[n] * taps[n]` for `dst.len()` consecutive
/// columns at once (each column keeps its own sequential order), optionally fused.
#[inline]
fn scalar_tail<'s>(dst: &mut [f32], taps: &[f32], src: impl Fn(usize) -> &'s [f32], fused: bool) {
    dst.fill(0.0);
    for (n, &f) in taps.iter().enumerate() {
        let s = &src(n)[..dst.len()];
        if fused {
            for (t, &x) in dst.iter_mut().zip(s) {
                *t = x.mul_add(f, *t);
            }
        } else {
            for (t, &x) in dst.iter_mut().zip(s) {
                *t += x * f;
            }
        }
    }
}

/// One 8-column block of dlib's SIMD filter loop: lane `k` accumulates `taps[n] * src[n *
/// stride + k]` into three accumulators (taps `n ≡ 0, 1, 2 (mod 3)`, the taps after the last
/// full triple going to the first), combined as `t1 + (t2 + t3)`. The accumulators stay in
/// registers, like dlib's `simd8f`.
// `k` indexes the 8 lanes of dlib's simd8f.
#[inline(always)]
fn dot8(taps: &[f32], src: &[f32], stride: usize, acc: &mut [[f32; 8]; 3]) {
    // Every filter of the frontal face detector has 10 taps (80 px window / 8 px cells). With
    // the length known at compile time LLVM unrolls the tap loop (≈ 1.2× faster per filter
    // pass); the operations and their order are unchanged.
    if taps.len() == 10 {
        dot8_body(&taps[..10], src, stride, acc);
    } else {
        dot8_body(taps, src, stride, acc);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot8_body(taps: &[f32], src: &[f32], stride: usize, acc: &mut [[f32; 8]; 3]) {
    use std::arch::aarch64::{float32x4_t, vaddq_f32, vld1q_f32, vmulq_n_f32, vst1q_f32};
    let len = taps.len();
    if len == 0 {
        return;
    }
    // Every load below reads 8 floats at `n * stride` for `n < len`, all inside this window.
    let src = &src[..(len - 1) * stride + 8];
    let p = src.as_ptr();
    // SAFETY: NEON is part of the aarch64 baseline. Each `vld1q_f32(p.add(i))` reads 4 floats at
    // `i = n * stride` or `i + 4` with `n < len`, i.e. at most `(len - 1) * stride + 8` floats from
    // the start of `src`, which the slice above guarantees; `acc` rows are 8 floats. The
    // operations are dlib's `simd8f` ones: a separate multiply and add per lane, in tap order,
    // so the result is the same as the portable version bit for bit.
    unsafe {
        let ld = |i: usize| -> (float32x4_t, float32x4_t) {
            (vld1q_f32(p.add(i)), vld1q_f32(p.add(i + 4)))
        };
        let mut a = [
            (
                vld1q_f32(acc[0].as_ptr()),
                vld1q_f32(acc[0].as_ptr().add(4)),
            ),
            (
                vld1q_f32(acc[1].as_ptr()),
                vld1q_f32(acc[1].as_ptr().add(4)),
            ),
            (
                vld1q_f32(acc[2].as_ptr()),
                vld1q_f32(acc[2].as_ptr().add(4)),
            ),
        ];
        let mac = |a: &mut (float32x4_t, float32x4_t), s: (float32x4_t, float32x4_t), f: f32| {
            a.0 = vaddq_f32(a.0, vmulq_n_f32(s.0, f));
            a.1 = vaddq_f32(a.1, vmulq_n_f32(s.1, f));
        };
        let mut n = 0usize;
        while n + 2 < len {
            let (s0, s1, s2) = (ld(n * stride), ld((n + 1) * stride), ld((n + 2) * stride));
            mac(&mut a[0], s0, taps[n]);
            mac(&mut a[1], s1, taps[n + 1]);
            mac(&mut a[2], s2, taps[n + 2]);
            n += 3;
        }
        while n < len {
            mac(&mut a[0], ld(n * stride), taps[n]);
            n += 1;
        }
        for (dst, v) in acc.iter_mut().zip(a) {
            vst1q_f32(dst.as_mut_ptr(), v.0);
            vst1q_f32(dst.as_mut_ptr().add(4), v.1);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
#[inline(always)]
fn dot8_body(taps: &[f32], src: &[f32], stride: usize, acc: &mut [[f32; 8]; 3]) {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };
    let len = taps.len();
    if len == 0 {
        return;
    }
    // Every load below reads 8 floats at `n * stride` for `n < len`, all inside this window.
    let src = &src[..(len - 1) * stride + 8];
    let p = src.as_ptr();
    // SAFETY: AVX is enabled at compile time (x86-64-v3 baseline, see .cargo/config.toml).
    // Each `_mm256_loadu_ps(p.add(i))` reads 8 floats at `i = n * stride` with `n < len`, inside
    // the slice above; `acc` rows are 8 floats. A separate multiply and add per lane in tap
    // order: the same result as the portable version bit for bit.
    unsafe {
        let mut a: [__m256; 3] = [
            _mm256_loadu_ps(acc[0].as_ptr()),
            _mm256_loadu_ps(acc[1].as_ptr()),
            _mm256_loadu_ps(acc[2].as_ptr()),
        ];
        let mac = |a: &mut __m256, i: usize, f: f32| {
            *a = _mm256_add_ps(
                *a,
                _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), _mm256_set1_ps(f)),
            );
        };
        let mut n = 0usize;
        while n + 2 < len {
            mac(&mut a[0], n * stride, taps[n]);
            mac(&mut a[1], (n + 1) * stride, taps[n + 1]);
            mac(&mut a[2], (n + 2) * stride, taps[n + 2]);
            n += 3;
        }
        while n < len {
            mac(&mut a[0], n * stride, taps[n]);
            n += 1;
        }
        for (dst, v) in acc.iter_mut().zip(a) {
            _mm256_storeu_ps(dst.as_mut_ptr(), v);
        }
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx")
)))]
#[inline(always)]
fn dot8_body(taps: &[f32], src: &[f32], stride: usize, acc: &mut [[f32; 8]; 3]) {
    dot8_body_portable(taps, src, stride, acc);
}

#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx")
    ),
    allow(dead_code)
)]
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn dot8_body_portable(taps: &[f32], src: &[f32], stride: usize, acc: &mut [[f32; 8]; 3]) {
    let len = taps.len();
    if len == 0 {
        return;
    }
    // One bounds check for the whole window instead of one per tap.
    let src = &src[..(len - 1) * stride + 8];
    let lanes = |n: usize| -> &[f32; 8] {
        let s: &[f32; 8] = src[n * stride..n * stride + 8].try_into().unwrap();
        s
    };
    let mut n = 0usize;
    while n + 2 < len {
        let (s0, s1, s2) = (lanes(n), lanes(n + 1), lanes(n + 2));
        let (f0, f1, f2) = (taps[n], taps[n + 1], taps[n + 2]);
        for k in 0..8 {
            acc[0][k] += s0[k] * f0;
            acc[1][k] += s1[k] * f1;
            acc[2][k] += s2[k] * f2;
        }
        n += 3;
    }
    while n < len {
        let s0 = lanes(n);
        let f0 = taps[n];
        for k in 0..8 {
            acc[0][k] += s0[k] * f0;
        }
        n += 1;
    }
}

/// Stores (or adds to `dst`) `t1 + (t2 + t3)` for the 8 lanes.
#[inline(always)]
fn finish8(acc: &[[f32; 8]; 3], dst: &mut [f32], add_to: bool) {
    for (k, d) in dst[..8].iter_mut().enumerate() {
        let v = acc[0][k] + (acc[1][k] + acc[2][k]);
        *d = if add_to { v + *d } else { v };
    }
}

/// dlib's `float_spatially_filter_image_separable(in, out, row_filter, col_filter, scratch,
/// add_to)`. Returns the valid area.
pub(crate) fn filter_separable(
    plane: PlaneRef<'_>,
    out: &mut FloatImage,
    row_filter: &[f32],
    col_filter: &[f32],
    scratch: &mut FilterScratch,
    add_to: bool,
) -> Rect {
    let PlaneRef {
        data: input,
        rows,
        cols,
    } = plane;
    if rows == 0 || cols == 0 {
        *out = FloatImage::default();
        return Rect::new(0, 0, -1, -1);
    }
    out.set_size(rows, cols);
    let first_row = (col_filter.len() / 2) as i64;
    let first_col = (row_filter.len() / 2) as i64;
    let last_row = rows as i64 - (col_filter.len() as i64 - 1) / 2;
    let last_col = cols as i64 - (row_filter.len() as i64 - 1) / 2;
    let non_border = Rect::new(first_col, first_row, last_col - 1, last_row - 1);
    if !add_to {
        out.zero_border(non_border);
    }
    let tmp = &mut scratch.image;
    tmp.set_size(rows, cols);
    let w = simd_cols(first_col, last_col);
    let fc = first_col as usize;

    // Row filter: SIMD columns [fc, fc + w), then the scalar tail.
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        for j in (0..w).step_by(8) {
            let mut acc = [[0f32; 8]; 3];
            dot8(row_filter, &row[j..], 1, &mut acc);
            finish8(&acc, &mut tmp.data[r * cols + fc + j..], false);
        }
        let tail = fc + w..(last_col.max(0) as usize).max(fc + w);
        if !tail.is_empty() {
            scalar_tail(
                &mut tmp.data[r * cols + tail.start..r * cols + tail.end],
                row_filter,
                |n| &row[w + n..],
                false,
            );
        }
    }

    // Column filter.
    let tmp = &tmp.data;
    for r in first_row.max(0) as usize..last_row.max(0) as usize {
        let top = r - first_row as usize;
        for j in (0..w).step_by(8) {
            let mut acc = [[0f32; 8]; 3];
            dot8(col_filter, &tmp[top * cols + fc + j..], cols, &mut acc);
            finish8(&acc, &mut out.data[r * cols + fc + j..], add_to);
        }
        let tail = fc + w..(last_col.max(0) as usize).max(fc + w);
        if !tail.is_empty() {
            let mut t = [0f32; 8];
            let t = &mut t[..tail.len()];
            // Fused in the reference build.
            scalar_tail(
                t,
                col_filter,
                |m| &tmp[(top + m) * cols + tail.start..],
                true,
            );
            for (d, &v) in out.data[r * cols + tail.start..r * cols + tail.end]
                .iter_mut()
                .zip(t.iter())
            {
                *d = if add_to { *d + v } else { v };
            }
        }
    }
    non_border
}

/// dlib's `float_spatially_filter_image(in, out, filter, add_to)` (non-separable). Returns the
/// valid area.
// `k` indexes the 8 lanes of dlib's simd8f.
#[allow(clippy::needless_range_loop)]
pub(crate) fn filter_full(
    plane: PlaneRef<'_>,
    out: &mut FloatImage,
    filter: &[f32],
    filter_rows: usize,
    filter_cols: usize,
    add_to: bool,
) -> Rect {
    let PlaneRef {
        data: input,
        rows,
        cols,
    } = plane;
    if rows == 0 || cols == 0 {
        *out = FloatImage::default();
        return Rect::new(0, 0, -1, -1);
    }
    out.set_size(rows, cols);
    let first_row = (filter_rows / 2) as i64;
    let first_col = (filter_cols / 2) as i64;
    let last_row = rows as i64 - (filter_rows as i64 - 1) / 2;
    let last_col = cols as i64 - (filter_cols as i64 - 1) / 2;
    let non_border = Rect::new(first_col, first_row, last_col - 1, last_row - 1);
    if !add_to {
        out.zero_border(non_border);
    }
    let w = simd_cols(first_col, last_col);
    let fc = first_col as usize;
    for r in first_row.max(0) as usize..last_row.max(0) as usize {
        let top = r - first_row as usize;
        for j in (0..w).step_by(8) {
            // The three accumulators run across all filter rows before being combined.
            let mut acc = [[0f32; 8]; 3];
            for m in 0..filter_rows {
                let taps = &filter[m * filter_cols..(m + 1) * filter_cols];
                dot8(taps, &input[(top + m) * cols + j..], 1, &mut acc);
            }
            finish8(&acc, &mut out.data[r * cols + fc + j..], add_to);
        }
        for c in fc + w..last_col.max(0) as usize {
            let mut t = 0f32;
            for m in 0..filter_rows {
                let row = &input[(top + m) * cols..];
                for n in 0..filter_cols {
                    t += row[c - fc + n] * filter[m * filter_cols + n];
                }
            }
            let dst = &mut out.data[r * cols + c];
            *dst = if add_to { *dst + t } else { t };
        }
    }
    non_border
}

/// `impl::apply_filters_to_fhog(w, feats, saliency_image)`: the sum over the 31 planes of each
/// plane filtered by its filter. Returns the valid area.
pub fn apply_filters(
    fb: &FilterBank,
    feats: &Fhog,
    saliency: &mut FloatImage,
    scratch: &mut FilterScratch,
) -> Rect {
    let (rows, cols) = (feats.rows, feats.cols);
    let mut area = Rect::new(0, 0, -1, -1);
    if !fb.use_separable() {
        for i in 0..NUM_PLANES {
            area = filter_full(
                PlaneRef {
                    data: feats.plane(i),
                    rows,
                    cols,
                },
                saliency,
                &fb.filters[i],
                fb.height,
                fb.width,
                i != 0,
            );
        }
        return area;
    }
    *saliency = FloatImage::default();
    for i in 0..NUM_PLANES {
        for (row, col) in fb.row_filters[i].iter().zip(&fb.col_filters[i]) {
            let add_to = saliency.rows * saliency.cols != 0;
            area = filter_separable(
                PlaneRef {
                    data: feats.plane(i),
                    rows,
                    cols,
                },
                saliency,
                row,
                col,
                scratch,
                add_to,
            );
        }
    }
    if saliency.rows * saliency.cols == 0 {
        saliency.set_size(rows, cols);
        saliency.data.fill(0.0);
    }
    area
}

/// dlib's `centered_rect(point, width, height)`.
pub fn centered_rect(p: Point, width: usize, height: usize) -> Rect {
    let left = p.x - (width as i64) / 2;
    let top = p.y - (height as i64) / 2;
    Rect::new(left, top, left + width as i64 - 1, top + height as i64 - 1)
}

/// Candidate detections `(score, rect)` of one filter bank on one pyramid level, appended to
/// `dets` (the loop of `impl::detect_from_fhog_pyramid`). `rect_up` maps a level rectangle back
/// to the original image.
#[allow(clippy::too_many_arguments)]
pub fn scan_level(
    saliency: &FloatImage,
    area: Rect,
    thresh: f64,
    det_box: (usize, usize),
    cell_size: i64,
    padding: (i64, i64),
    rect_up: impl Fn(Rect) -> Rect,
    dets: &mut Vec<(f64, Rect)>,
) {
    if area.is_empty() {
        return;
    }
    for r in area.top..=area.bottom {
        for c in area.left..=area.right {
            let v = saliency.data[r as usize * saliency.cols + c as usize];
            if f64::from(v) >= thresh {
                let fr = centered_rect(Point::new(c, r), det_box.0, det_box.1);
                let rect = fhog_to_image(fr, cell_size, padding.0, padding.1);
                dets.push((f64::from(v), rect_up(rect)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separable_matches_full_filter_on_rank_one_input() {
        // A rank-1 filter applied separably equals the full filter up to rounding.
        let (rows, cols) = (13, 21);
        let input: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37) % 11) as f32 * 0.25)
            .collect();
        let row = [0.5f32, -1.0, 0.25];
        let col = [1.0f32, 2.0, 1.0];
        let full: Vec<f32> = col
            .iter()
            .flat_map(|&c| row.iter().map(move |&r| r * c))
            .collect();
        let (mut a, mut b, mut s): (FloatImage, FloatImage, FilterScratch) = Default::default();
        let ra = filter_separable(
            PlaneRef {
                data: &input,
                rows,
                cols,
            },
            &mut a,
            &row,
            &col,
            &mut s,
            false,
        );
        let rb = filter_full(
            PlaneRef {
                data: &input,
                rows,
                cols,
            },
            &mut b,
            &full,
            3,
            3,
            false,
        );
        assert_eq!(ra, rb);
        assert_eq!(ra, Rect::new(1, 1, cols as i64 - 2, rows as i64 - 2));
        for (x, y) in a.data.iter().zip(&b.data) {
            assert!((x - y).abs() < 1e-5, "{x} {y}");
        }
        // The border stays zero.
        assert_eq!(a.data[0], 0.0);
    }

    #[test]
    fn filters_larger_than_the_image() {
        let input = vec![1.0f32; 4 * 5];
        let taps = vec![1.0f32; 10];
        let (mut out, mut s) = (FloatImage::default(), FilterScratch::default());
        let area = filter_separable(
            PlaneRef {
                data: &input,
                rows: 4,
                cols: 5,
            },
            &mut out,
            &taps,
            &taps,
            &mut s,
            false,
        );
        assert!(area.is_empty());
        assert!(out.data.iter().all(|&v| v == 0.0));
        let empty = filter_separable(
            PlaneRef {
                data: &[],
                rows: 0,
                cols: 0,
            },
            &mut out,
            &taps,
            &taps,
            &mut s,
            false,
        );
        assert!(empty.is_empty());
    }

    #[test]
    fn centered_rect_matches_dlib() {
        assert_eq!(
            centered_rect(Point::new(0, 0), 80, 80),
            Rect::new(-40, -40, 39, 39)
        );
        assert_eq!(
            centered_rect(Point::new(5, 7), 10, 9),
            Rect::new(0, 3, 9, 11)
        );
    }
}
