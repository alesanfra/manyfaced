//! `pyramid_down<6>`, `pyramid_up` and dlib's bilinear `resize_image`.
//!
//! Ported from `dlib/image_transforms/image_pyramid.h` and `interpolation.h` (dlib v20.0.1,
//! Boost Software License 1.0, see `THIRD_PARTY_NOTICES`).
//!
//! dlib's `resize_image` with `interpolate_bilinear` has dedicated grayscale and RGB versions that
//! process 4 output columns at a time in `float` (SIMD) and the remaining columns of each row in
//! scalar code with different types and rounding. Both paths are reproduced column by column,
//! including the multiply-adds that clang fuses in the scalar statements of the reference build
//! (`-ffp-contract=on`, marked "clang fma").

use super::Image;
use super::interpolation::Pixels;
use crate::geometry::{DPoint, DRect, Rect};

/// `pyramid_down<6>::point_down`: `(p - 0.3) * (5/6)`.
pub fn pyramid_down6_point_down(p: DPoint) -> DPoint {
    let ratio = (6.0 - 1.0) / 6.0;
    DPoint::new((p.x - 0.3) * ratio, (p.y - 0.3) * ratio)
}

/// `pyramid_down<6>::point_up`: `p * (6/5) + 0.3`.
pub fn pyramid_down6_point_up(p: DPoint) -> DPoint {
    let ratio = 6.0 / (6.0 - 1.0);
    DPoint::new(p.x * ratio + 0.3, p.y * ratio + 0.3)
}

/// `pyramid_down<6>::rect_down(rectangle)`, converted back to an integer `rectangle`.
pub fn pyramid_down6_rect_down(r: Rect) -> Rect {
    let d = DRect::from(r);
    DRect::from_corners(
        pyramid_down6_point_down(d.tl_corner()),
        pyramid_down6_point_down(d.br_corner()),
    )
    .to_rect()
}

/// `pyramid_down<6>::rect_up(rectangle, levels)`, converted back to an integer `rectangle`.
pub fn pyramid_down6_rect_up(r: Rect, levels: usize) -> Rect {
    let d = DRect::from(r);
    let (mut tl, mut br) = (d.tl_corner(), d.br_corner());
    for _ in 0..levels {
        tl = pyramid_down6_point_up(tl);
        br = pyramid_down6_point_up(br);
    }
    DRect::from_corners(tl, br).to_rect()
}

/// `pyramid_down<2>::point_up`: `(p + (1.25, 0.75)) * 2`.
pub fn pyramid_down2_point_up(p: DPoint) -> DPoint {
    DPoint::new((p.x + 1.25) * 2.0, (p.y + 0.75) * 2.0)
}

/// `pyramid_down<2>::rect_down(rectangle, levels)`, converted back to an integer `rectangle`.
pub fn pyramid_down2_rect_down_levels(r: Rect, levels: usize) -> Rect {
    let d = DRect::from(r);
    let (mut tl, mut br) = (d.tl_corner(), d.br_corner());
    for _ in 0..levels {
        tl = super::interpolation::pyramid_down2_point_down(tl);
        br = super::interpolation::pyramid_down2_point_down(br);
    }
    DRect::from_corners(tl, br).to_rect()
}

/// `pyramid_down<6>` applied to an image: `resize_image` to `(5 * rows / 6) × (5 * cols / 6)`
/// (integer division, as dlib's `lround` of an integer quotient).
pub fn pyramid_down6<const C: usize>(img: &impl Pixels<C>) -> Image<C> {
    let rows = 5 * img.height() / 6;
    let cols = 5 * img.width() / 6;
    resize_bilinear(img, cols, rows)
}

/// `pyramid_up(img, pyramid_down<2>)`: resize to the size of `rect_up(get_rect(img))`.
pub fn pyramid_up<const C: usize>(img: &impl Pixels<C>) -> Image<C> {
    if img.width() == 0 || img.height() == 0 {
        return Image::zeros(0, 0);
    }
    let rect = DRect::from(Rect::new(
        0,
        0,
        img.width() as i64 - 1,
        img.height() as i64 - 1,
    ));
    let up = DRect::from_corners(
        pyramid_down2_point_up(rect.tl_corner()),
        pyramid_down2_point_up(rect.br_corner()),
    )
    .to_rect();
    if up.is_empty() {
        return Image::zeros(0, 0);
    }
    resize_bilinear(img, (up.right + 1) as usize, (up.bottom + 1) as usize)
}

/// dlib's `resize_image(in, out, interpolate_bilinear())` for 8-bit grayscale (`C = 1`) or RGB
/// (`C = 3`) images of the same pixel type.
pub fn resize_bilinear<const C: usize>(
    src: &impl Pixels<C>,
    out_w: usize,
    out_h: usize,
) -> Image<C> {
    let mut out = Image::<C>::zeros(out_w, out_h);
    let (in_w, in_h) = (src.width(), src.height());
    if out_w == 0 || out_h == 0 || in_w == 0 || in_h == 0 {
        return out;
    }
    let x_scale = (in_w - 1) as f64 / (out_w.max(2) - 1) as f64;
    let y_scale = (in_h - 1) as f64 / (out_h.max(2) - 1) as f64;
    let in_nc = in_w as i64;
    let mut y = -y_scale;
    for r in 0..out_h {
        y += y_scale;
        let top = y.floor() as i64;
        let bottom = (top + 1).min(in_h as i64 - 1);
        let tb_frac = y - top as f64;
        let top_row = src.row(top as usize);
        let bottom_row = src.row(bottom as usize);
        let row_out = out.row_mut(r);

        // The 4-wide float loop.
        let x = -4.0 * x_scale;
        let tbf = tb_frac as f32;
        let inv_tbf = (1.0 - tb_frac) as f32;
        let xs4 = (4.0 * x_scale) as f32;
        let mut xv = [
            x as f32,
            (x + x_scale) as f32,
            2.0f64.mul_add(x_scale, x) as f32, // clang fma
            3.0f64.mul_add(x_scale, x) as f32, // clang fma
        ];
        let mut c = 0usize;
        loop {
            let mut left = [0i64; 4];
            let mut lr = [0f32; 4];
            for k in 0..4 {
                xv[k] += xs4;
                // vcvtq_s32_f32 / _mm_cvttps_epi32: truncation towards zero.
                left[k] = xv[k] as i32 as i64;
                lr[k] = xv[k] - left[k] as f32;
            }
            // dlib stops when the right neighbour of the last lane leaves the image. (The output
            // bound is implied by it in practice; it is checked too so no index can overflow.)
            if left[3] + 1 >= in_nc || c + 3 >= out_w {
                break;
            }
            // Weights per lane, then each channel across the 4 lanes (lane-innermost so LLVM can
            // vectorize it); per lane the expression and its order are unchanged.
            let (mut tlf, mut trf, mut blf, mut brf) = ([0f32; 4], [0f32; 4], [0f32; 4], [0f32; 4]);
            let (mut l, mut rr) = ([0usize; 4], [0usize; 4]);
            for k in 0..4 {
                let inv_lr = 1.0 - lr[k];
                tlf[k] = inv_tbf * inv_lr;
                trf[k] = inv_tbf * lr[k];
                blf[k] = tbf * inv_lr;
                brf[k] = tbf * lr[k];
                l[k] = left[k] as usize * C;
                rr[k] = (left[k] as usize + 1) * C;
            }
            for ch in 0..C {
                let mut v = [0f32; 4];
                for k in 0..4 {
                    let tl = f32::from(top_row[l[k] + ch]);
                    let tr = f32::from(top_row[rr[k] + ch]);
                    let bl = f32::from(bottom_row[l[k] + ch]);
                    let br = f32::from(bottom_row[rr[k] + ch]);
                    v[k] = tlf[k] * tl + trf[k] * tr + blf[k] * bl + brf[k] * br;
                }
                for k in 0..4 {
                    row_out[(c + k) * C + ch] = if C == 1 {
                        // static_cast<T>(value + 0.5), with 0.5 a double.
                        (f64::from(v[k]) + 0.5) as u8
                    } else {
                        // simd4i(float) truncates, then static_cast<unsigned char>.
                        v[k] as i32 as u8
                    };
                }
            }
            c += 4;
        }

        // The scalar tail.
        let mut x = (c as f64).mul_add(x_scale, -x_scale); // clang fma
        while c < out_w {
            x += x_scale;
            let left = x.floor() as i64;
            let right = (left + 1).min(in_nc - 1);
            let (l, rr) = (left as usize, right as usize);
            for ch in 0..C {
                let (tl, tr) = (top_row[l * C + ch], top_row[rr * C + ch]);
                let (bl, br) = (bottom_row[l * C + ch], bottom_row[rr * C + ch]);
                row_out[c * C + ch] = if C == 1 {
                    let lr = (x - left as f64) as f32;
                    let (tl, tr, bl, br) =
                        (f32::from(tl), f32::from(tr), f32::from(bl), f32::from(br));
                    // clang fma: each `a * b + c * d` becomes fma(a, b, c * d).
                    let top_mix = (1.0 - lr).mul_add(tl, lr * tr);
                    let bottom_mix = (1.0 - lr).mul_add(bl, lr * br);
                    let temp = (1.0 - tb_frac)
                        .mul_add(f64::from(top_mix), tb_frac * f64::from(bottom_mix))
                        as f32;
                    // assign_pixel(unsigned char, float): truncation, clamped to [0, 255].
                    temp.clamp(0.0, 255.0) as u8
                } else {
                    let lr = x - left as f64;
                    let mix = |a: u8, b: u8| f64::from(a) * (1.0 - lr) + f64::from(b) * lr;
                    let v = mix(tl, tr) * (1.0 - tb_frac) + mix(bl, br) * tb_frac;
                    // vector_to_pixel: static_cast<unsigned char>(double).
                    v as u8
                };
            }
            c += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageView;

    #[test]
    fn pyramid_sizes() {
        let data = vec![0u8; 100 * 60];
        let v = ImageView::<1>::packed(&data, 100, 60).unwrap();
        let d = pyramid_down6(&v);
        assert_eq!((d.width(), d.height()), (83, 50));
        // rect_up of (0,0,99,59) is (2.5,1.5,200.5,119.5) -> lround -> 3,2,201,120.
        let u = pyramid_up(&v);
        assert_eq!((u.width(), u.height()), (202, 121));
        let empty = ImageView::<1>::packed(&[], 0, 0).unwrap();
        assert_eq!(pyramid_up(&empty).width(), 0);
        assert_eq!(pyramid_down6(&empty).width(), 0);
    }

    #[test]
    fn constant_images() {
        // The RGB SIMD path truncates (weights summing to slightly less than 1 can turn 77 into
        // 76), the grayscale one rounds.
        let data = vec![77u8; 37 * 23 * 3];
        let v = ImageView::<3>::packed(&data, 37, 23).unwrap();
        for img in [pyramid_down6(&v), pyramid_up(&v)] {
            assert!(img.as_bytes().iter().all(|&b| b == 77 || b == 76));
        }
        let data = vec![200u8; 37 * 23];
        let v = ImageView::<1>::packed(&data, 37, 23).unwrap();
        for img in [pyramid_down6(&v), pyramid_up(&v)] {
            assert!(img.as_bytes().iter().all(|&b| b == 200));
        }
    }

    #[test]
    fn rect_mappings() {
        // point_up(p) = 1.2 p + 0.3, then lround.
        assert_eq!(
            pyramid_down6_rect_up(Rect::new(10, 20, 30, 40), 1),
            Rect::new(12, 24, 36, 48)
        );
        assert_eq!(
            pyramid_down6_rect_up(Rect::new(10, 20, 30, 40), 0),
            Rect::new(10, 20, 30, 40)
        );
        assert_eq!(
            pyramid_down6_rect_down(Rect::new(0, 0, 99, 59)),
            Rect::new(0, 0, 82, 49)
        );
        assert_eq!(
            pyramid_down2_rect_down_levels(Rect::new(10, 10, 50, 50), 1),
            Rect::new(4, 4, 24, 24)
        );
    }
}
