//! Bilinear interpolation, `pyramid_down<2>` and `transform_image`.
//!
//! Ported from `dlib/image_transforms/interpolation.h` and `image_pyramid.h` (dlib v20.0.1,
//! Boost Software License 1.0, see `THIRD_PARTY_NOTICES`).

use super::{Image, ImageView};
use crate::geometry::{DPoint, DRect, PointTransformAffine, Rect};

/// Read access to an 8-bit image with `C` channels.
pub trait Pixels<const C: usize> {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn pixel(&self, x: usize, y: usize) -> [u8; C];
    /// The `width * C` bytes of row `y`.
    fn row(&self, y: usize) -> &[u8];
}

impl<const C: usize> Pixels<C> for ImageView<'_, C> {
    fn width(&self) -> usize {
        ImageView::width(self)
    }
    fn height(&self) -> usize {
        ImageView::height(self)
    }
    #[inline]
    fn pixel(&self, x: usize, y: usize) -> [u8; C] {
        ImageView::pixel(self, x, y)
    }
    #[inline]
    fn row(&self, y: usize) -> &[u8] {
        ImageView::row(self, y)
    }
}

impl<const C: usize> Pixels<C> for Image<C> {
    fn width(&self) -> usize {
        Image::width(self)
    }
    fn height(&self) -> usize {
        Image::height(self)
    }
    #[inline]
    fn pixel(&self, x: usize, y: usize) -> [u8; C] {
        let i = (y * Image::width(self) + x) * C;
        let mut p = [0u8; C];
        p.copy_from_slice(&self.as_bytes()[i..i + C]);
        p
    }
    #[inline]
    fn row(&self, y: usize) -> &[u8] {
        let w = Image::width(self) * C;
        &self.as_bytes()[y * w..(y + 1) * w]
    }
}

/// dlib's `sub_image(img, rect)` for a rectangle inside the image (an empty rectangle gives an
/// empty image).
#[derive(Clone, Copy)]
pub struct SubImage<'a, const C: usize> {
    view: ImageView<'a, C>,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
}

impl<'a, const C: usize> SubImage<'a, C> {
    /// Clips `rect` to the image, like the rectangles dlib passes to `sub_image` here, which are
    /// always intersected with the image first.
    pub fn new(view: ImageView<'a, C>, rect: Rect) -> Self {
        let full = Rect::new(0, 0, view.width() as i64 - 1, view.height() as i64 - 1);
        let r = Rect::new(
            rect.left.max(full.left),
            rect.top.max(full.top),
            rect.right.min(full.right),
            rect.bottom.min(full.bottom),
        );
        if r.is_empty() {
            return SubImage {
                view,
                x0: 0,
                y0: 0,
                width: 0,
                height: 0,
            };
        }
        SubImage {
            view,
            x0: r.left as usize,
            y0: r.top as usize,
            width: r.width() as usize,
            height: r.height() as usize,
        }
    }
}

impl<const C: usize> Pixels<C> for SubImage<'_, C> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn pixel(&self, x: usize, y: usize) -> [u8; C] {
        self.view.pixel(self.x0 + x, self.y0 + y)
    }
    #[inline]
    fn row(&self, y: usize) -> &[u8] {
        &self.view.row(self.y0 + y)[self.x0 * C..(self.x0 + self.width) * C]
    }
}

/// `interpolate_bilinear`: `None` when the 2×2 neighbourhood is not fully inside the image
/// (so the last row and column never interpolate). The result is truncated to `u8` per channel
/// (`vector_to_pixel` uses `static_cast<unsigned char>`).
#[inline]
pub fn interpolate_bilinear<const C: usize>(img: &impl Pixels<C>, p: DPoint) -> Option<[u8; C]> {
    let left = p.x.floor() as i64;
    let top = p.y.floor() as i64;
    let right = left.checked_add(1)?;
    let bottom = top.checked_add(1)?;
    if !(left >= 0 && top >= 0 && right < img.width() as i64 && bottom < img.height() as i64) {
        return None;
    }
    let lr_frac = p.x - left as f64;
    let tb_frac = p.y - top as f64;
    let (l, t, b) = (left as usize, top as usize, bottom as usize);
    // The 2x2 neighbourhood as two `2 * C` byte runs: left pixel then right pixel.
    let top_px = &img.row(t)[l * C..(l + 2) * C];
    let bottom_px = &img.row(b)[l * C..(l + 2) * C];
    let mut out = [0u8; C];
    for i in 0..C {
        let (tl, tr) = (top_px[i], top_px[C + i]);
        let (bl, br) = (bottom_px[i], bottom_px[C + i]);
        let top_row = (1.0 - lr_frac) * f64::from(tl) + lr_frac * f64::from(tr);
        let bottom_row = (1.0 - lr_frac) * f64::from(bl) + lr_frac * f64::from(br);
        let v = (1.0 - tb_frac) * top_row + tb_frac * bottom_row;
        out[i] = v as u8;
    }
    Some(out)
}

/// `transform_image(in, out, interpolate_bilinear(), map_point)` with a black background:
/// `out[r][c] = interp(in, map_point(c, r))`.
pub fn transform_image<const C: usize>(
    src: &impl Pixels<C>,
    out: &mut Image<C>,
    map_point: &PointTransformAffine,
) {
    for r in 0..out.height() {
        let row = out.row_mut(r);
        for (c, px) in row.chunks_exact_mut(C).enumerate() {
            let p = map_point.apply(DPoint::new(c as f64, r as f64));
            px.copy_from_slice(&interpolate_bilinear(src, p).unwrap_or([0; C]));
        }
    }
}

/// `pyramid_down<2>::point_down`: `p / 2 - (1.25, 0.75)`.
pub fn pyramid_down2_point_down(p: DPoint) -> DPoint {
    DPoint::new(p.x / 2.0 - 1.25, p.y / 2.0 - 0.75)
}

/// `pyramid_down<2>::rect_down`.
pub fn pyramid_down2_rect_down(r: &DRect) -> DRect {
    DRect::from_corners(
        pyramid_down2_point_down(r.tl_corner()),
        pyramid_down2_point_down(r.br_corner()),
    )
}

/// `pyramid_down<2>` applied to an image: a separable 5×5 `[1 4 6 4 1]` Gaussian with integer
/// arithmetic and `/256` truncation, keeping every other row and column. The output is
/// `((rows - 3) / 2) × ((cols - 3) / 2)`, or empty when the input has 8 or fewer rows or
/// columns.
pub fn pyramid_down2<const C: usize>(original: &impl Pixels<C>) -> Image<C> {
    let (nr, nc) = (original.height(), original.width());
    if nr <= 8 || nc <= 8 {
        return Image::zeros(0, 0);
    }
    let tcols = (nc - 3) / 2;
    let tw = tcols * C;
    // Row filter, every row, every other column. Values are at most 16 * 255.
    let mut temp = vec![0u32; nr * tw];
    for (r, dst) in temp.chunks_exact_mut(tw).enumerate() {
        let src = original.row(r);
        for (c, d) in dst.chunks_exact_mut(C).enumerate() {
            // The 5-pixel window of output column c: columns 2c..2c+4.
            let w = &src[2 * c * C..2 * c * C + 5 * C];
            for (i, d) in d.iter_mut().enumerate() {
                *d = u32::from(w[i])
                    + u32::from(w[C + i]) * 4
                    + u32::from(w[2 * C + i]) * 6
                    + u32::from(w[3 * C + i]) * 4
                    + u32::from(w[4 * C + i]);
            }
        }
    }
    // Column filter on rows 2, 4, ... < nr - 2, then /256 (truncating).
    let drows = (nr - 3) / 2;
    let mut out = vec![0u8; drows * tw];
    for (dr, dst) in out.chunks_exact_mut(tw).enumerate() {
        let r = 2 + 2 * dr;
        let rows = [
            &temp[(r - 2) * tw..(r - 1) * tw],
            &temp[(r - 1) * tw..r * tw],
            &temp[r * tw..(r + 1) * tw],
            &temp[(r + 1) * tw..(r + 2) * tw],
            &temp[(r + 2) * tw..(r + 3) * tw],
        ];
        let [a, b, m, d, e] = rows;
        for (o, ((((a, b), m), d), e)) in dst.iter_mut().zip(a.iter().zip(b).zip(m).zip(d).zip(e)) {
            let v = a + b * 4 + m * 6 + d * 4 + e;
            // At most 65280 / 256 = 255; dlib's assign_pixel would clamp anyway.
            *o = (v / 256).min(255) as u8;
        }
    }
    Image::from_vec(out, tcols, drows).unwrap_or_else(|_| Image::zeros(0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_matches_hand_computation() {
        let img = Image::<1>::from_vec(vec![0, 100, 200, 50], 2, 2).unwrap();
        assert_eq!(interpolate_bilinear(&img, DPoint::new(0.0, 0.0)), Some([0]));
        // 0.75 * (0.5*0 + 0.5*100) + 0.25 * (0.5*200 + 0.5*50) = 37.5 + 31.25 = 68.75 -> 68
        assert_eq!(
            interpolate_bilinear(&img, DPoint::new(0.5, 0.25)),
            Some([68])
        );
        // The last row/column cannot be the top-left of a 2x2 neighbourhood.
        assert_eq!(interpolate_bilinear(&img, DPoint::new(1.0, 0.0)), None);
        assert_eq!(interpolate_bilinear(&img, DPoint::new(-0.1, 0.0)), None);
        assert_eq!(
            interpolate_bilinear(&img, DPoint::new(f64::NAN, 0.0)),
            Some([0])
        );
        assert_eq!(interpolate_bilinear(&img, DPoint::new(1e300, 0.0)), None);
    }

    #[test]
    fn pyramid_down_sizes_and_constant_images() {
        let img = Image::<3>::from_vec(vec![77; 20 * 13 * 3], 20, 13).unwrap();
        let d = pyramid_down2(&img);
        assert_eq!((d.width(), d.height()), (8, 5));
        // A constant image stays constant (16 * 16 * 77 / 256 = 77).
        assert!(d.as_bytes().iter().all(|&v| v == 77));
        let small = Image::<1>::zeros(9, 8);
        assert_eq!(pyramid_down2(&small).width(), 0);
    }

    #[test]
    fn pyramid_down_filter_weights() {
        // Output (c, dr) reads columns 2c..2c+4 and rows 2dr..2dr+4 with weights 1 4 6 4 1.
        // A bright pixel at (4, 4) is the centre of output (1, 1): weight 6 * 6.
        let mut v = vec![0u8; 81];
        v[4 * 9 + 4] = 255;
        let img = Image::<1>::from_vec(v, 9, 9).unwrap();
        let d = pyramid_down2(&img);
        assert_eq!((d.width(), d.height()), (3, 3));
        assert_eq!(d.as_bytes()[4], (255 * 36 / 256) as u8);
        // For output (0, 1) it is the last column of the window: weight 1 * 6.
        assert_eq!(d.as_bytes()[3], (255 * 6 / 256) as u8);
    }

    #[test]
    fn sub_image_clips() {
        let data: Vec<u8> = (0..30).collect();
        let v = ImageView::<1>::packed(&data, 6, 5).unwrap();
        let s = SubImage::new(v, Rect::new(4, 3, 10, 10));
        assert_eq!((Pixels::width(&s), Pixels::height(&s)), (2, 2));
        assert_eq!(Pixels::pixel(&s, 1, 1), [29]);
        let e = SubImage::new(v, Rect::new(10, 10, 12, 12));
        assert_eq!(Pixels::width(&e), 0);
    }
}
