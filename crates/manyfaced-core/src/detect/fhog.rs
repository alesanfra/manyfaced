//! Felzenszwalb HOG features (`extract_fhog_features`) and the coordinate mappings between image
//! and FHOG space.
//!
//! Ported from `dlib/image_transforms/fhog.h` (dlib v20.0.1, Boost Software License 1.0, see
//! `THIRD_PARTY_NOTICES`), itself derived from the voc-release `features.cc` by Felzenszwalb,
//! Girshick, McAllester and Ramanan (MIT-style license, reproduced in dlib's source).
//!
//! dlib computes the gradient histograms 8 columns at a time with SIMD types and finishes each row
//! in scalar code; the two paths differ in how RGB gradient ties are broken and in the order of
//! the bilinear weight products, so both are reproduced column by column.
//!
//! The reference build (`dlib-bin` on arm64) uses NEON, where `simd4f` division and square root
//! are Newton–Raphson approximations ([`super::neon`]), and was compiled by clang, which fuses
//! `a * b + c` inside single scalar expressions (`-ffp-contract=on`). Both are reproduced: the
//! SIMD paths use the NEON emulation, and the scalar statements that clang contracts use
//! `mul_add` (marked "clang fma" below).

use super::neon;
use crate::geometry::{Point, Rect};
use crate::image::interpolation::Pixels;

/// Number of FHOG planes: 18 contrast-sensitive orientations, 9 contrast-insensitive ones and
/// 4 texture (gradient energy) features.
pub const NUM_PLANES: usize = 31;

/// 31 planes of `rows × cols` features, plane-major, including the zero border dlib adds for
/// the filter padding.
#[derive(Debug, Clone, Default)]
pub struct Fhog {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Fhog {
    pub(crate) fn plane(&self, i: usize) -> &[f32] {
        let n = self.rows * self.cols;
        &self.data[i * n..(i + 1) * n]
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }
}

/// Unit vectors of the 9 orientation bins (`directions` in fhog.h).
const DIRECTIONS: [[f32; 2]; 9] = [
    [1.0000, 0.0000],
    [0.9397, 0.3420],
    [0.7660, 0.6428],
    [0.500, 0.8660],
    [0.1736, 0.9848],
    [-0.1736, 0.9848],
    [-0.5000, 0.8660],
    [-0.7660, 0.6428],
    [-0.9397, 0.3420],
];

/// Gradient `(dx, dy, |g|²)` at column `x` of `row`, from the rows above and below. For RGB
/// the channel with the largest magnitude wins; `later_wins_ties` selects dlib's SIMD
/// tie-breaking (later channel wins) instead of the scalar one (earlier channel wins).
#[inline(always)]
fn gradient_at<const C: usize>(
    up: &[u8],
    row: &[u8],
    down: &[u8],
    x: usize,
    later_wins_ties: bool,
) -> (i32, i32, i32) {
    if C == 1 {
        let gx = i32::from(row[x + 1]) - i32::from(row[x - 1]);
        let gy = i32::from(down[x]) - i32::from(up[x]);
        return (gx, gy, gx * gx + gy * gy);
    }
    let mut best = (0, 0, -1);
    for ch in 0..3 {
        let gx = i32::from(row[(x + 1) * C + ch]) - i32::from(row[(x - 1) * C + ch]);
        let gy = i32::from(down[x * C + ch]) - i32::from(up[x * C + ch]);
        let len = gx * gx + gy * gy;
        let better = if later_wins_ties {
            len >= best.2
        } else {
            len > best.2
        };
        if ch == 0 || better {
            best = (gx, gy, len);
        }
    }
    best
}

#[cfg(test)]
fn gradient<const C: usize>(
    img: &impl Pixels<C>,
    x: usize,
    y: usize,
    later_wins_ties: bool,
) -> (i32, i32, i32) {
    gradient_at::<C>(
        img.row(y - 1),
        img.row(y),
        img.row(y + 1),
        x,
        later_wins_ties,
    )
}

/// `neon::sqrt(len)` for every possible gradient energy `len <= 2 * 255²`.
fn neon_sqrt_table() -> &'static [f32] {
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=2 * 255 * 255)
            .map(|len| neon::sqrt(len as f32))
            .collect()
    })
}

/// dlib's SIMD orientation search for 8 lanes: the second comparison sees the best value
/// updated by the first, as with `select`.
#[inline(always)]
fn orientation8(gx: &[f32; 8], gy: &[f32; 8]) -> [usize; 8] {
    // Branch-free selects on `u32` lanes, like dlib's `select`, so LLVM can keep all 8 lanes in
    // vector registers (with data-dependent branches and `usize` indices it did not, and this
    // search took about half of the FHOG time). Same comparisons in the same order.
    let mut best_dot = [0f32; 8];
    let mut best_o = [0u32; 8];
    for (o, d) in DIRECTIONS.iter().enumerate() {
        let (o, o9) = (o as u32, o as u32 + 9);
        for k in 0..8 {
            let dot = gx[k] * d[0] + gy[k] * d[1];
            let m1 = dot > best_dot[k];
            best_dot[k] = if m1 { dot } else { best_dot[k] };
            best_o[k] = if m1 { o } else { best_o[k] };
            let m2 = -dot > best_dot[k];
            best_dot[k] = if m2 { -dot } else { best_dot[k] };
            best_o[k] = if m2 { o9 } else { best_o[k] };
        }
    }
    best_o.map(|o| o as usize)
}

/// Snap a gradient to one of the 18 contrast-sensitive orientations. The SIMD path computes
/// `gx * d0 + gy * d1`; the scalar path uses `dlib::dot`, a matrix product whose
/// `temp += a * b` clang fuses.
#[inline]
fn orientation(gx: f32, gy: f32, scalar: bool) -> usize {
    let mut best_dot = 0.0f32;
    let mut best_o = 0usize;
    for (o, d) in DIRECTIONS.iter().enumerate() {
        let dot = if scalar {
            d[1].mul_add(gy, d[0] * gx) // clang fma
        } else {
            gx * d[0] + gy * d[1]
        };
        if dot > best_dot {
            best_dot = dot;
            best_o = o;
        } else if -dot > best_dot {
            best_dot = -dot;
            best_o = o + 9;
        }
    }
    best_o
}

/// NEON `sum(simd4f)`: `(v2 + v0) + (v3 + v1)`.
#[inline]
fn sum4(v: [f32; 4]) -> f32 {
    (v[2] + v[0]) + (v[3] + v[1])
}

/// dlib's `extract_fhog_features(img, hog, cell_size, filter_rows_padding,
/// filter_cols_padding)` into planar `float` images.
pub fn extract_fhog<const C: usize>(
    img: &impl Pixels<C>,
    cell_size: usize,
    filter_rows_padding: usize,
    filter_cols_padding: usize,
) -> Fhog {
    assert!(cell_size > 1, "cell_size 1 is not supported");
    assert!(filter_rows_padding > 0 && filter_cols_padding > 0);
    let (nr, nc) = (img.height(), img.width());
    let cs = cell_size as f32;
    let cells_nr = ((nr as f32) / cs + 0.5) as usize;
    let cells_nc = ((nc as f32) / cs + 0.5) as usize;
    let hog_nr = cells_nr.saturating_sub(2);
    let hog_nc = cells_nc.saturating_sub(2);
    if hog_nr == 0 || hog_nc == 0 {
        return Fhog::default();
    }

    // Histograms with a 1 cell border: hist[(r * hist_nc + c) * 18 + o].
    let hist_nc = cells_nc + 2;
    let hist_row = hist_nc * 18;
    let mut hist = vec![0.0f32; (cells_nr + 2) * hist_row];
    let hidx = |r: usize, c: usize, o: usize| (r * hist_nc + c) * 18 + o;

    let visible_nr = (cells_nr * cell_size).min(nr) as i64 - 1;
    let visible_nc = (cells_nc * cell_size).min(nc) as i64 - 1;
    let simd_end = visible_nc - 7;
    // Columns handled by the 8-wide loop: x in 1..1 + simd_cols.
    let simd_cols = if simd_end > 1 {
        ((simd_end - 1) as usize).div_ceil(8) * 8
    } else {
        0
    };
    let tail_end = visible_nc.max(1) as usize;

    // The horizontal bilinear weights only depend on the column. SIMD path:
    // xp = (x + 0.5) / cs + 0.5 (NEON division), cell = trunc(xp). Scalar path:
    // xp = (x + 0.5) / (double)cs - 0.5, cell = floor(xp) + 1.
    let recip_cs = neon::reciprocal(cs);
    let mut col_cell = vec![0usize; tail_end.max(1)];
    let mut col_vx0 = vec![0f32; tail_end.max(1)];
    let mut col_vx1 = vec![0f32; tail_end.max(1)];
    for x in 1..tail_end {
        if x < 1 + simd_cols {
            let xp = (x as f32 + 0.5) * recip_cs + 0.5;
            let ixp = xp as i32;
            col_cell[x] = ixp as usize;
            col_vx0[x] = xp - ixp as f32;
            col_vx1[x] = 1.0f32 - col_vx0[x];
        } else {
            let xp = ((x as f64 + 0.5) / cell_size as f64 - 0.5) as f32;
            let ixp = xp.floor() as i64;
            col_cell[x] = (ixp + 1) as usize;
            col_vx0[x] = xp - ixp as f32;
            col_vx1[x] = (1.0 - f64::from(col_vx0[x])) as f32;
        }
    }
    let sqrt_table = neon_sqrt_table();

    for y in 1..visible_nr.max(1) as usize {
        let yp = ((y as f64 + 0.5) / f64::from(cs) - 0.5) as f32;
        let iyp = yp.floor() as i64;
        let vy0 = yp - iyp as f32;
        let vy1 = (1.0 - f64::from(vy0)) as f32;
        let hr = (iyp + 1) as usize;
        let (up, row, down) = (img.row(y - 1), img.row(y), img.row(y + 1));
        let (head, rest) = hist.split_at_mut((hr + 1) * hist_row);
        let top = &mut head[hr * hist_row..];
        let bottom = &mut rest[..hist_row];

        // The 8-wide SIMD loop.
        let mut x = 1usize;
        while x < 1 + simd_cols {
            let (mut gx, mut gy, mut v) = ([0f32; 8], [0f32; 8], [0f32; 8]);
            for k in 0..8 {
                let (dx, dy, len) = gradient_at::<C>(up, row, down, x + k, true);
                gx[k] = dx as f32;
                gy[k] = dy as f32;
                v[k] = sqrt_table[len as usize];
            }
            let o = orientation8(&gx, &gy);
            for k in 0..8 {
                let xl = x + k;
                let (vx1, vx0) = (col_vx1[xl] * v[k], col_vx0[xl] * v[k]);
                let c0 = col_cell[xl] * 18 + o[k];
                top[c0] += vy1 * vx1;
                bottom[c0] += vy0 * vx1;
                top[c0 + 18] += vy1 * vx0;
                bottom[c0 + 18] += vy0 * vx0;
            }
            x += 8;
        }
        // The scalar tail.
        while x < tail_end {
            let (gx, gy, len) = gradient_at::<C>(up, row, down, x, false);
            let o = orientation(gx as f32, gy as f32, true);
            let v = (len as f32).sqrt();
            let (vx0, vx1) = (col_vx0[x], col_vx1[x]);
            let c0 = col_cell[x] * 18 + o;
            // clang fma: h += (vy * vx) * v.
            top[c0] = (vy1 * vx1).mul_add(v, top[c0]);
            bottom[c0] = (vy0 * vx1).mul_add(v, bottom[c0]);
            top[c0 + 18] = (vy1 * vx0).mul_add(v, top[c0 + 18]);
            bottom[c0 + 18] = (vy0 * vx0).mul_add(v, bottom[c0 + 18]);
            x += 1;
        }
    }

    // Energy of each cell, summed over orientations.
    let mut norm = vec![0.0f32; cells_nr * cells_nc];
    for r in 0..cells_nr {
        for c in 0..cells_nc {
            let h = &hist[hidx(r + 1, c + 1, 0)..hidx(r + 1, c + 1, 18)];
            let n = &mut norm[r * cells_nc + c];
            for o in 0..9 {
                let s = h[o] + h[o + 9];
                *n = s.mul_add(s, *n); // clang fma
            }
        }
    }

    let rows = hog_nr + filter_rows_padding - 1;
    let cols = hog_nc + filter_cols_padding - 1;
    let plane_len = rows * cols;
    let mut data = vec![0.0f32; NUM_PLANES * plane_len];
    let row_off = (filter_rows_padding - 1) / 2;
    let col_off = (filter_cols_padding - 1) / 2;
    let nm = |r: usize, c: usize| norm[r * cells_nc + c];
    let eps = 0.0001f32;
    let (two_x, point_two, point_one) = ((2.0f64 * 0.2357) as f32, 0.2f64 as f32, 0.1f64 as f32);
    for y in 0..hog_nr {
        let yy = y + row_off;
        for x in 0..hog_nc {
            let xx = x + col_off;
            let z1 = [nm(y + 1, x + 1), nm(y, x + 1), nm(y + 1, x), nm(y, x)];
            let z2 = [
                nm(y + 1, x + 2),
                nm(y, x + 2),
                nm(y + 1, x + 1),
                nm(y, x + 1),
            ];
            let z3 = [
                nm(y + 2, x + 1),
                nm(y + 1, x + 1),
                nm(y + 2, x),
                nm(y + 1, x),
            ];
            let z4 = [
                nm(y + 2, x + 2),
                nm(y + 1, x + 2),
                nm(y + 2, x + 1),
                nm(y + 1, x + 1),
            ];
            let mut nn = [0f32; 4];
            let mut n = [0f32; 4];
            for k in 0..4 {
                nn[k] = point_two * neon::sqrt(z1[k] + z2[k] + z3[k] + z4[k] + eps);
                n[k] = neon::div(point_one, nn[k]);
            }
            let h = &hist[hidx(y + 2, x + 2, 0)..hidx(y + 2, x + 2, 18)];
            let clip = |v: f32| -> [f32; 4] {
                let mut out = [0f32; 4];
                for k in 0..4 {
                    out[k] = v.min(nn[k]) * n[k];
                }
                out
            };
            let mut set = |o: usize, v: f32| data[o * plane_len + yy * cols + xx] = v;
            let mut t = [0f32; 4];
            // Contrast-sensitive features.
            for o in (0..18).step_by(3) {
                let (h0, h1, h2) = (clip(h[o]), clip(h[o + 1]), clip(h[o + 2]));
                set(o, sum4(h0));
                set(o + 1, sum4(h1));
                set(o + 2, sum4(h2));
                for k in 0..4 {
                    t[k] += h0[k] + h1[k] + h2[k];
                }
            }
            for v in &mut t {
                *v *= two_x;
            }
            // Contrast-insensitive features.
            for o in (0..9).step_by(3) {
                let h0 = clip(h[o] + h[o + 9]);
                let h1 = clip(h[o + 1] + h[o + 10]);
                let h2 = clip(h[o + 2] + h[o + 11]);
                set(o + 18, sum4(h0));
                set(o + 19, sum4(h1));
                set(o + 20, sum4(h2));
            }
            // Texture features.
            for (k, &v) in t.iter().enumerate() {
                set(27 + k, v);
            }
        }
    }
    Fhog { rows, cols, data }
}

/// dlib's `image_to_fhog(point)`.
pub fn image_to_fhog_point(p: Point, cell_size: i64, rows_pad: i64, cols_pad: i64) -> Point {
    // Integer division truncates towards zero, like dlib's vector<long> / int.
    Point::new(
        (p.x - 1) / cell_size - 1 + (cols_pad - 1) / 2,
        (p.y - 1) / cell_size - 1 + (rows_pad - 1) / 2,
    )
}

/// dlib's `image_to_fhog(rectangle)`.
pub fn image_to_fhog(r: Rect, cell_size: i64, rows_pad: i64, cols_pad: i64) -> Rect {
    let tl = image_to_fhog_point(r.tl_corner(), cell_size, rows_pad, cols_pad);
    let br = image_to_fhog_point(r.br_corner(), cell_size, rows_pad, cols_pad);
    Rect::new(tl.x, tl.y, br.x, br.y)
}

/// dlib's `fhog_to_image(point)`: the centre of the cell in image coordinates.
pub fn fhog_to_image_point(p: Point, cell_size: i64, rows_pad: i64, cols_pad: i64) -> Point {
    let x = (p.x + 1 - (cols_pad - 1) / 2) * cell_size + 1;
    let y = (p.y + 1 - (rows_pad - 1) / 2) * cell_size + 1;
    let half = cell_size / 2;
    let ox = if x >= 0 { half } else { -half };
    let oy = if y >= 0 { half } else { -half };
    Point::new(x + ox, y + oy)
}

/// dlib's `fhog_to_image(rectangle)`.
pub fn fhog_to_image(r: Rect, cell_size: i64, rows_pad: i64, cols_pad: i64) -> Rect {
    let tl = fhog_to_image_point(r.tl_corner(), cell_size, rows_pad, cols_pad);
    let br = fhog_to_image_point(r.br_corner(), cell_size, rows_pad, cols_pad);
    Rect::new(tl.x, tl.y, br.x, br.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageView;

    #[test]
    fn coordinate_mappings_match_dlib() {
        // The frontal detector's 80x80 window maps to a 10x10 FHOG window.
        let r = image_to_fhog(Rect::new(-40, -40, 39, 39), 8, 1, 1);
        assert_eq!(r, Rect::new(-6, -6, 3, 3));
        assert_eq!((r.width(), r.height()), (10, 10));
        // Negative coordinates use the negative half-cell offset.
        assert_eq!(
            fhog_to_image_point(Point::new(-1, 0), 8, 1, 1),
            Point::new(5, 13)
        );
        assert_eq!(
            fhog_to_image_point(Point::new(-2, 0), 8, 1, 1),
            Point::new(-11, 13)
        );
        assert_eq!(
            fhog_to_image_point(Point::new(0, 0), 8, 10, 10),
            Point::new(-27, -27)
        );
    }

    #[test]
    fn small_images_give_empty_features() {
        // cells = round(size / 8); the features need at least 3 cells per side.
        for (w, h) in [(0, 0), (1, 1), (15, 40), (40, 19)] {
            let data = vec![9u8; w * h];
            let v = ImageView::<1>::packed(&data, w, h).unwrap();
            assert!(extract_fhog(&v, 8, 10, 10).is_empty(), "{w}x{h}");
        }
        let data = vec![9u8; 24 * 24];
        let v = ImageView::<1>::packed(&data, 24, 24).unwrap();
        let f = extract_fhog(&v, 8, 10, 10);
        assert_eq!((f.rows, f.cols), (10, 10));
        // Constant image: no gradient, all features zero.
        assert!(f.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gradient_tie_breaking() {
        // Red and blue have the same |g|² with different directions.
        let mut data = vec![0u8; 3 * 3 * 3];
        let px = |x: usize, y: usize| (y * 3 + x) * 3;
        data[px(2, 1)] = 10; // red dx = 10
        data[px(1, 2) + 2] = 10; // blue dy = 10
        let v = ImageView::<3>::packed(&data, 3, 3).unwrap();
        assert_eq!(gradient(&v, 1, 1, false), (10, 0, 100));
        assert_eq!(gradient(&v, 1, 1, true), (0, 10, 100));
    }
}
