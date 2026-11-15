//! Face chips: `chip_details`, `get_face_chip_details` and `extract_image_chip(s)`.
//!
//! Ported from `dlib/image_transforms/interpolation.h` (dlib v20.0.1, Boost Software License
//! 1.0, see `THIRD_PARTY_NOTICES`), keeping dlib's order of floating point operations.

use crate::geometry::drect::rotate_point;
use crate::geometry::{DPoint, DRect, Rect, find_affine_transform, find_similarity_transform};
use crate::image::interpolation::{
    SubImage, pyramid_down2, pyramid_down2_rect_down, transform_image,
};
use crate::image::{Image, ImageView};

/// Where to extract a chip from an image: a rectangle rotated by `angle` radians around its
/// centre, resampled to `rows × cols` pixels (dlib's `chip_details`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChipDetails {
    pub rect: DRect,
    pub angle: f64,
    pub rows: u64,
    pub cols: u64,
}

/// Invalid face chip inputs. dlib checks these with `DLIB_CASSERT`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ChipError {
    #[error("face chips need 5 or 68 landmarks, got {0}")]
    WrongLandmarkCount(usize),
    #[error("chip size must be > 0")]
    ZeroSize,
    #[error("padding must be >= 0, got {0}")]
    InvalidPadding(f64),
    #[error("a {rows}x{cols} chip is too large")]
    TooLarge { rows: u64, cols: u64 },
    #[error("chip rectangle and angle must be finite, got {0:?}")]
    NonFinite(ChipDetails),
    #[error("jitter needs a square, non-empty chip, got {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },
}

impl ChipDetails {
    /// Rejects non-finite rectangles and angles, for which dlib's pyramid level search never
    /// terminates (an infinite area never drops below the chip size).
    pub fn validate(&self) -> Result<(), ChipError> {
        let r = &self.rect;
        if [r.left, r.top, r.right, r.bottom, self.angle]
            .iter()
            .all(|v| v.is_finite())
        {
            Ok(())
        } else {
            Err(ChipError::NonFinite(*self))
        }
    }
}

impl ChipDetails {
    /// `chip_details(chip_points, img_points, chip_dims(rows, cols))`: the chip whose similarity
    /// transform maps `chip_points` (in chip pixels) onto `img_points` (in image pixels).
    /// Requires at least 2 point pairs of equal length.
    pub fn from_points(
        chip_points: &[DPoint],
        img_points: &[DPoint],
        rows: u64,
        cols: u64,
    ) -> Self {
        let tform = find_similarity_transform(chip_points, img_points);
        // p = tform.get_m() * (1, 0)
        let m = tform.m;
        let p = DPoint::new(m[0][0] * 1.0 + m[0][1] * 0.0, m[1][0] * 1.0 + m[1][1] * 0.0);
        let angle = p.y.atan2(p.x);
        let scale = p.length();
        // tform(point(cols, rows) / 2.0)
        let center = tform.apply(DPoint::new(cols as f64 / 2.0, rows as f64 / 2.0));
        let rect = DRect::centered(center, cols as f64 * scale, rows as f64 * scale);
        ChipDetails {
            rect,
            angle,
            rows,
            cols,
        }
    }

    /// `rows * cols`.
    pub fn size(&self) -> u64 {
        self.rows.saturating_mul(self.cols)
    }
}

/// Mean positions of the 5-point model's landmarks in a unit chip.
const MEAN_FACE_5: [(f64, f64); 5] = [
    (0.8595674595992, 0.2134981538014),
    (0.6460604764104, 0.2289674387677),
    (0.1205750620789, 0.2137274526848),
    (0.3340850613712, 0.2290642403242),
    (0.4901123135679, 0.6277975316475),
];

/// Average positions of the 68-point model's landmarks 17..=67 in a unit chip.
const MEAN_FACE_68_X: [f64; 51] = [
    0.000213256,
    0.0752622,
    0.18113,
    0.29077,
    0.393397,
    0.586856,
    0.689483,
    0.799124,
    0.904991,
    0.98004,
    0.490127,
    0.490127,
    0.490127,
    0.490127,
    0.36688,
    0.426036,
    0.490127,
    0.554217,
    0.613373,
    0.121737,
    0.187122,
    0.265825,
    0.334606,
    0.260918,
    0.182743,
    0.645647,
    0.714428,
    0.793132,
    0.858516,
    0.79751,
    0.719335,
    0.254149,
    0.340985,
    0.428858,
    0.490127,
    0.551395,
    0.639268,
    0.726104,
    0.642159,
    0.556721,
    0.490127,
    0.423532,
    0.338094,
    0.290379,
    0.428096,
    0.490127,
    0.552157,
    0.689874,
    0.553364,
    0.490127,
    0.42689,
];
const MEAN_FACE_68_Y: [f64; 51] = [
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891, 0.0187482, 0.038915,
    0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326, 0.609345, 0.628106, 0.609345,
    0.587326, 0.216423, 0.178758, 0.179852, 0.231733, 0.245099, 0.244077, 0.231733, 0.179852,
    0.178758, 0.216423, 0.244077, 0.245099, 0.780233, 0.745405, 0.727388, 0.742578, 0.727388,
    0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746,
    0.785343, 0.778746, 0.784792, 0.824182, 0.831803, 0.824182,
];

/// dlib's `get_face_chip_details(det, size, padding)`: the `size × size` chip that aligns the
/// face described by 5 or 68 landmarks (dlib's landmarking models) with dlib's mean face,
/// with `padding` (a fraction of the face width) added on each side.
///
/// dlib's own default arguments are `size = 200`, `padding = 0.2`; `dlib.get_face_chip` uses
/// 150 and 0.25 and passes `padding` through a C++ `float` first.
pub fn face_chip_details(
    landmarks: &[DPoint],
    size: u64,
    padding: f64,
) -> Result<ChipDetails, ChipError> {
    if landmarks.len() != 5 && landmarks.len() != 68 {
        return Err(ChipError::WrongLandmarkCount(landmarks.len()));
    }
    // Also rejects NaN, as dlib's `padding >= 0` assert does.
    if padding.is_nan() || padding < 0.0 {
        return Err(ChipError::InvalidPadding(padding));
    }
    if size == 0 {
        return Err(ChipError::ZeroSize);
    }
    let sizef = size as f64;
    let mut from = Vec::new();
    let mut to = Vec::new();
    if landmarks.len() == 5 {
        // p = (padding + p) / (2 * padding + 1) is a dlib matrix expression: the division by a
        // scalar is evaluated as a multiplication by its reciprocal.
        let inv = 1.0 / (2.0 * padding + 1.0);
        for (i, &(x, y)) in MEAN_FACE_5.iter().enumerate() {
            let p = DPoint::new((x + padding) * inv, (y + padding) * inv);
            from.push(DPoint::new(p.x * sizef, p.y * sizef));
            to.push(landmarks[i]);
        }
    } else {
        for (i, &landmark) in landmarks.iter().enumerate().skip(17) {
            // Ignore the lower lip and the eyebrows.
            if (55..=59).contains(&i) || (65..=67).contains(&i) || (17..=26).contains(&i) {
                continue;
            }
            // Scalar arithmetic here: a true division.
            let x = (padding + MEAN_FACE_68_X[i - 17]) / (2.0 * padding + 1.0);
            let y = (padding + MEAN_FACE_68_Y[i - 17]) / (2.0 * padding + 1.0);
            from.push(DPoint::new(x * sizef, y * sizef));
            to.push(landmark);
        }
    }
    Ok(ChipDetails::from_points(&from, &to, size, size))
}

fn image_rect<const C: usize>(img: &ImageView<'_, C>) -> Rect {
    Rect::new(0, 0, img.width() as i64 - 1, img.height() as i64 - 1)
}

/// The largest chip, in pixels (16384 × 16384). A larger chip is almost certainly an error in
/// the arguments, and its memory could be reserved lazily and then exhaust the RAM.
pub const MAX_CHIP_PIXELS: u64 = 1 << 28;

/// Allocates a black `rows × cols` image, refusing sizes that cannot be allocated.
fn black_chip<const C: usize>(rows: u64, cols: u64) -> Result<Image<C>, ChipError> {
    let too_large = ChipError::TooLarge { rows, cols };
    if rows.checked_mul(cols).is_none_or(|n| n > MAX_CHIP_PIXELS) {
        return Err(too_large);
    }
    // Both fit in usize: their product is at most MAX_CHIP_PIXELS.
    Image::try_zeros(cols as usize, rows as usize).ok_or(too_large)
}

/// `impl::basic_extract_image_chip`: a plain copy of `location` (black outside the image).
fn basic_extract<const C: usize>(
    img: &ImageView<'_, C>,
    location: Rect,
) -> Result<Image<C>, ChipError> {
    let mut chip = black_chip::<C>(location.height(), location.width())?;
    let area = {
        let full = image_rect(img);
        Rect::new(
            location.left.max(full.left),
            location.top.max(full.top),
            location.right.min(full.right),
            location.bottom.min(full.bottom),
        )
    };
    if area.is_empty() {
        return Ok(chip);
    }
    for rr in area.top..=area.bottom {
        for cc in area.left..=area.right {
            let p = img.pixel(cc as usize, rr as usize);
            chip.set_pixel(
                (cc - location.left) as usize,
                (rr - location.top) as usize,
                p,
            );
        }
    }
    Ok(chip)
}

fn is_plain_copy(d: &ChipDetails) -> bool {
    d.angle == 0.0 && d.rows as f64 == d.rect.height() && d.cols as f64 == d.rect.width()
}

/// dlib's `extract_image_chips(img, chip_locations, chips)` with bilinear interpolation.
///
/// When a chip is much smaller than its source rectangle, dlib first builds a `pyramid_down<2>`
/// image pyramid of the region covering *all* the chips and samples from the appropriate
/// level, so extracting several chips at once can give slightly different pixels than
/// extracting them one by one. This function reproduces that exactly.
///
/// Chips with 0 rows or columns come out empty (dlib asserts on them, and without asserts its
/// pyramid depth search never terminates for them); they are excluded from the pyramid
/// computation.
pub fn extract_image_chips<const C: usize>(
    img: ImageView<'_, C>,
    chips: &[ChipDetails],
) -> Result<Vec<Image<C>>, ChipError> {
    for d in chips {
        d.validate()?;
    }
    let img_drect = DRect::from(image_rect(&img));
    // Depth of the image pyramid and the region it must cover.
    let mut max_depth = 0usize;
    let mut bbox = Rect::new(0, 0, -1, -1);
    for d in chips {
        let size = d.size() as f64;
        if d.size() == 0 {
            continue;
        }
        let mut depth = 0usize;
        let mut grow = 2.0f64;
        let mut rect = pyramid_down2_rect_down(&d.rect);
        while rect.area() > size {
            rect = pyramid_down2_rect_down(&rect);
            depth += 1;
            // dlib drops the size by 2 per level and keeps a 2 pixel border.
            grow = grow * 2.0 + 2.0;
        }
        let cent = d.rect.center();
        let mut rot_rect = DRect::default();
        for corner in [
            d.rect.tl_corner(),
            d.rect.tr_corner(),
            d.rect.bl_corner(),
            d.rect.br_corner(),
        ] {
            rot_rect.include(rotate_point(cent, corner, d.angle));
        }
        let covered = rot_rect.grow(grow).intersect(&img_drect).to_rect();
        bbox = bbox.union(covered);
        max_depth = max_depth.max(depth);
    }

    let region = SubImage::new(img, bbox);
    let mut levels: Vec<Image<C>> = Vec::with_capacity(max_depth);
    if max_depth > 0 {
        levels.push(pyramid_down2(&region));
        for i in 1..max_depth {
            let next = pyramid_down2(&levels[i - 1]);
            levels.push(next);
        }
    }

    let mut out = Vec::with_capacity(chips.len());
    for d in chips {
        if d.size() == 0 {
            out.push(black_chip::<C>(d.rows, d.cols)?);
            continue;
        }
        if is_plain_copy(d) {
            out.push(basic_extract(&img, d.rect.to_rect())?);
            continue;
        }
        let mut chip = black_chip::<C>(d.rows, d.cols)?;
        let size = d.size() as f64;
        // Pick the pyramid level whose resolution matches the chip.
        let mut level: isize = -1;
        let mut rect = d.rect.translate(-(bbox.left as f64), -(bbox.top as f64));
        while pyramid_down2_rect_down(&rect).area() > size {
            level += 1;
            rect = pyramid_down2_rect_down(&rect);
        }
        // dlib indexes levels[level] directly; the depth search above guarantees
        // level < max_depth up to floating point coincidences, which we clamp.
        let level = if levels.is_empty() {
            -1
        } else {
            level.min(levels.len() as isize - 1)
        };
        let (rows, cols) = (d.rows as f64, d.cols as f64);
        let from = [
            DPoint::new(0.0, 0.0),
            DPoint::new(cols - 1.0, 0.0),
            DPoint::new(0.0, rows - 1.0),
        ];
        let c = rect.center();
        let to = [
            rotate_point(c, rect.tl_corner(), d.angle),
            rotate_point(c, rect.tr_corner(), d.angle),
            rotate_point(c, rect.bl_corner(), d.angle),
        ];
        let trns = find_affine_transform(&from, &to);
        if level < 0 {
            transform_image(&region, &mut chip, &trns);
        } else {
            transform_image(&levels[level as usize], &mut chip, &trns);
        }
        out.push(chip);
    }
    Ok(out)
}

/// dlib's `extract_image_chip(img, location, chip)` with bilinear interpolation.
pub fn extract_image_chip<const C: usize>(
    img: ImageView<'_, C>,
    location: &ChipDetails,
) -> Result<Image<C>, ChipError> {
    location.validate()?;
    if is_plain_copy(location) {
        return basic_extract(&img, location.rect.to_rect());
    }
    let mut chips = extract_image_chips(img, std::slice::from_ref(location))?;
    Ok(chips.pop().unwrap_or_else(|| Image::zeros(0, 0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn five_points() -> Vec<DPoint> {
        // A plausible upright face: eye corners, nose.
        vec![
            DPoint::new(160.0, 100.0),
            DPoint::new(135.0, 102.0),
            DPoint::new(70.0, 100.0),
            DPoint::new(95.0, 102.0),
            DPoint::new(115.0, 150.0),
        ]
    }

    #[test]
    fn non_finite_chips_are_rejected() {
        let data = vec![0u8; 30 * 30];
        let img = ImageView::<1>::packed(&data, 30, 30).unwrap();
        for rect in [
            DRect::new(0.0, 0.0, f64::INFINITY, f64::INFINITY),
            DRect::new(f64::NAN, 0.0, 10.0, 10.0),
        ] {
            let d = ChipDetails {
                rect,
                angle: 0.1,
                rows: 10,
                cols: 10,
            };
            assert!(matches!(
                extract_image_chip(img, &d),
                Err(ChipError::NonFinite(_))
            ));
            assert!(matches!(
                extract_image_chips(img, &[d]),
                Err(ChipError::NonFinite(_))
            ));
        }
    }

    #[test]
    fn invalid_inputs() {
        let p = five_points();
        assert_eq!(
            face_chip_details(&p[..4], 150, 0.25),
            Err(ChipError::WrongLandmarkCount(4))
        );
        assert_eq!(face_chip_details(&p, 0, 0.25), Err(ChipError::ZeroSize));
        assert!(matches!(
            face_chip_details(&p, 150, -0.1),
            Err(ChipError::InvalidPadding(_))
        ));
        assert!(face_chip_details(&p, 150, f64::NAN).is_err());
    }

    #[test]
    fn upright_face_has_small_angle_and_requested_size() {
        let d = face_chip_details(&five_points(), 150, 0.25).unwrap();
        assert_eq!((d.rows, d.cols), (150, 150));
        assert!(d.angle.abs() < 0.1, "{d:?}");
        assert!(d.rect.width() > 100.0 && d.rect.width() < 300.0, "{d:?}");
    }

    #[test]
    fn chips_outside_or_on_empty_images_are_black() {
        let d = face_chip_details(&five_points(), 50, 0.25).unwrap();
        let empty = ImageView::<3>::packed(&[], 0, 0).unwrap();
        let chip = extract_image_chip(empty, &d).unwrap();
        assert_eq!((chip.width(), chip.height()), (50, 50));
        assert!(chip.as_bytes().iter().all(|&v| v == 0));
        let data = vec![200u8; 30 * 30];
        let img = ImageView::<1>::packed(&data, 30, 30).unwrap();
        let far = ChipDetails {
            rect: DRect::new(1000.0, 1000.0, 1100.0, 1100.0),
            angle: 0.3,
            rows: 20,
            cols: 20,
        };
        let chip = extract_image_chip(img, &far).unwrap();
        assert!(chip.as_bytes().iter().all(|&v| v == 0));
    }

    #[test]
    fn degenerate_details_do_not_panic() {
        let data = vec![10u8; 40 * 40 * 3];
        let img = ImageView::<3>::packed(&data, 40, 40).unwrap();
        for d in [
            ChipDetails {
                rect: DRect::default(),
                angle: 0.0,
                rows: 0,
                cols: 0,
            },
            ChipDetails {
                rect: DRect::new(-1e300, -1e300, 1e300, 1e300),
                angle: 1.0,
                rows: 2,
                cols: 2,
            },
        ] {
            let chips = extract_image_chips(img, &[d]).unwrap();
            assert_eq!(chips[0].height() as u64, d.rows);
        }
        let huge = ChipDetails {
            rect: DRect::new(0.0, 0.0, 10.0, 10.0),
            angle: 0.1,
            rows: u64::MAX,
            cols: 2,
        };
        assert!(matches!(
            extract_image_chip(img, &huge),
            Err(ChipError::TooLarge { .. })
        ));
    }

    #[test]
    fn plain_copy_path() {
        let data: Vec<u8> = (0..100).collect();
        let img = ImageView::<1>::packed(&data, 10, 10).unwrap();
        let d = ChipDetails {
            rect: DRect::new(8.0, -1.0, 10.0, 1.0),
            angle: 0.0,
            rows: 3,
            cols: 3,
        };
        let chip = extract_image_chip(img, &d).unwrap();
        assert_eq!(chip.as_bytes(), [0, 0, 0, 8, 9, 0, 18, 19, 0]);
    }
}
