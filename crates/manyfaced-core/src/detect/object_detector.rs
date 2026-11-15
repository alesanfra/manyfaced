//! dlib's `object_detector<scan_fhog_pyramid<pyramid_down<6>>>`, the type of the HOG frontal
//! face detector and of detectors trained with `dlib.train_simple_object_detector`.
//!
//! Ported from `dlib/image_processing/object_detector.h`, `scan_fhog_pyramid.h` and
//! `box_overlap_testing.h` (dlib v20.0.1, Boost Software License 1.0, see
//! `THIRD_PARTY_NOTICES`). The embedded frontal face detector weights come from
//! `dlib/image_processing/frontal_face_detector.h`, extracted byte for byte by
//! `tools/cpp_probe/extract_frontal_detector.cpp`.

use std::path::Path;
use std::sync::OnceLock;

use super::fhog::{NUM_PLANES, extract_fhog, image_to_fhog};
use super::scan::{
    FilterBank, FilterScratch, FloatImage, apply_filters, centered_rect, scan_level,
};
use crate::geometry::{Point, Rect};
use crate::image::interpolation::Pixels;
use crate::image::pyramid::{
    pyramid_down2_rect_down_levels, pyramid_down6, pyramid_down6_rect_down, pyramid_down6_rect_up,
    pyramid_up,
};
use crate::image::{Image, ImageView};
use crate::serialize::{self, Reader};

/// The serialized frontal face detector shipped with dlib (5 FHOG filters).
static FRONTAL_FACE_DAT: &[u8] = include_bytes!("../../data/frontal_face_detector.dat");

/// Configuration of a `scan_fhog_pyramid` as stored in the file. The pyramid type
/// (`pyramid_down<6>`) is part of the C++ type and is not serialized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanFhogPyramidConfig {
    cell_size: u32,
    padding: u64,
    window_width: u64,
    window_height: u64,
    max_pyramid_levels: u64,
    min_pyramid_layer_width: u64,
    min_pyramid_layer_height: u64,
    nuclear_norm_regularization_strength: f64,
    num_dimensions: usize,
}

impl ScanFhogPyramidConfig {
    pub fn cell_size(&self) -> u32 {
        self.cell_size
    }

    /// Padding of the detection window, in cells.
    pub fn padding(&self) -> u64 {
        self.padding
    }

    pub fn window_width(&self) -> u64 {
        self.window_width
    }

    pub fn window_height(&self) -> u64 {
        self.window_height
    }

    pub fn max_pyramid_levels(&self) -> u64 {
        self.max_pyramid_levels
    }

    pub fn min_pyramid_layer_width(&self) -> u64 {
        self.min_pyramid_layer_width
    }

    pub fn min_pyramid_layer_height(&self) -> u64 {
        self.min_pyramid_layer_height
    }

    /// Training-time parameter, kept for completeness.
    pub fn nuclear_norm_regularization_strength(&self) -> f64 {
        self.nuclear_norm_regularization_strength
    }

    /// Feature vector length, as stored in the file (`get_num_dimensions()`).
    pub fn num_dimensions(&self) -> usize {
        self.num_dimensions
    }
}

/// dlib's `test_box_overlap`: two boxes overlap if their intersection covers more than
/// `iou_thresh` of their bounding union, or more than `percent_covered_thresh` of either box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxOverlap {
    pub iou_thresh: f64,
    pub percent_covered_thresh: f64,
}

impl BoxOverlap {
    /// `test_box_overlap::operator()(a, b)`. Note that dlib divides by the area of the
    /// bounding box of the two rectangles (`(a + b).area()`), not by the area of their union.
    pub fn overlaps(&self, a: Rect, b: Rect) -> bool {
        let inter = Rect::new(
            a.left.max(b.left),
            a.top.max(b.top),
            a.right.min(b.right),
            a.bottom.min(b.bottom),
        );
        let inner = inter.area() as f64;
        if inner == 0.0 {
            return false;
        }
        let outer = a.union(b).area() as f64;
        inner / outer > self.iou_thresh
            || inner / a.area() as f64 > self.percent_covered_thresh
            || inner / b.area() as f64 > self.percent_covered_thresh
    }
}

/// `compute_fhog_window_size`: the detection window mapped to FHOG cells and grown by the
/// padding. `None` for degenerate configurations.
fn fhog_window_size(s: &ScanFhogPyramidConfig) -> Option<(usize, usize)> {
    let (ww, wh, pad) = (
        i64::try_from(s.window_width).ok()?,
        i64::try_from(s.window_height).ok()?,
        i64::try_from(s.padding).ok()?,
    );
    if ww > 1 << 30 || wh > 1 << 30 || pad > 1 << 30 {
        return None;
    }
    let rect = centered_rect(Point::new(0, 0), ww as usize, wh as usize);
    let f = image_to_fhog(rect, i64::from(s.cell_size), 1, 1);
    let grown = Rect::new(f.left - pad, f.top - pad, f.right + pad, f.bottom + pad);
    let (w, h) = (grown.width() as usize, grown.height() as usize);
    // Real detectors have windows of tens of cells. The limit keeps `w * h * NUM_PLANES` and the
    // filter allocations small for a hostile file.
    if w > MAX_FHOG_WINDOW_CELLS || h > MAX_FHOG_WINDOW_CELLS {
        return None;
    }
    // The detection box is the window without padding; it must not be empty.
    (w > 2 * pad as usize && h > 2 * pad as usize).then_some((w, h))
}

/// The largest FHOG window side, in cells, that a detector file can have.
const MAX_FHOG_WINDOW_CELLS: usize = 1 << 12;

/// A FHOG sliding-window detector with one weight vector per filter. Immutable after loading,
/// hence `Send + Sync` (unlike dlib's, which caches the image pyramid inside the scanner).
#[derive(Debug, Clone)]
pub struct FhogObjectDetector {
    scanner: ScanFhogPyramidConfig,
    overlap: BoxOverlap,
    weights: Vec<Vec<f64>>,
    /// One filter bank per weight vector, built at load (`processed_weight_vector::init`).
    banks: Vec<FilterBank>,
    /// FHOG detection window `(width, height)` in cells, padding included.
    fhog_window: (usize, usize),
}

/// One detection: the rectangle, `score - threshold` of the filter that found it, and that
/// filter's index (`rect_detection`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub rect: Rect,
    pub confidence: f64,
    pub weight_index: usize,
}

/// Why a detection could not run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DetectError {
    #[error("upsampling a {width}x{height} image {upsample} times would need too much memory")]
    UpsampleTooLarge {
        width: usize,
        height: usize,
        upsample: u32,
    },
}

/// Images above this many pixels after upsampling are refused instead of exhausting memory.
const MAX_UPSAMPLED_PIXELS: u128 = 1 << 32;

const WHAT: &str = "dlib object_detector";

impl FhogObjectDetector {
    /// Loads a serialized `object_detector<scan_fhog_pyramid<pyramid_down<6>>>` file.
    pub fn load(path: impl AsRef<Path>) -> crate::Result<Self> {
        crate::load_file(path.as_ref(), WHAT, Self::parse)
    }

    /// Parses a serialized `object_detector<scan_fhog_pyramid<pyramid_down<6>>>`. Like dlib,
    /// trailing bytes are ignored.
    pub fn from_bytes(data: &[u8]) -> crate::Result<Self> {
        crate::load_bytes(data, WHAT, Self::parse)
    }

    /// dlib's `get_frontal_face_detector()`. Parsed once, then cloned.
    pub fn frontal_face() -> Self {
        static DETECTOR: OnceLock<FhogObjectDetector> = OnceLock::new();
        DETECTOR
            .get_or_init(|| {
                // The embedded bytes are a constant of the crate; the unit test
                // `embedded_frontal_face_detector` proves they parse.
                Self::parse(FRONTAL_FACE_DAT).expect("embedded frontal face detector is valid")
            })
            .clone()
    }

    pub fn scanner(&self) -> &ScanFhogPyramidConfig {
        &self.scanner
    }

    pub fn overlap(&self) -> BoxOverlap {
        self.overlap
    }

    /// One weight vector per filter. Each has at least `num_dimensions()` elements; dlib uses
    /// the element after the filter weights as the detection threshold.
    pub fn weights(&self) -> &[Vec<f64>] {
        &self.weights
    }

    pub fn num_detectors(&self) -> usize {
        self.weights.len()
    }

    /// The filter bank of each weight vector.
    pub fn filter_banks(&self) -> &[FilterBank] {
        &self.banks
    }

    /// The FHOG detection window `(width, height)` in cells, padding included.
    pub fn fhog_window(&self) -> (usize, usize) {
        self.fhog_window
    }

    /// `object_detector::operator()(img, dets, adjust_threshold)`: runs every filter over the
    /// `pyramid_down<6>` FHOG pyramid of `img`, keeps windows scoring at least the filter's
    /// threshold plus `adjust_threshold`, sorts them by decreasing confidence and applies
    /// non-maximum suppression with the detector's `test_box_overlap`.
    pub fn detect<const C: usize>(
        &self,
        img: &impl Pixels<C>,
        adjust_threshold: f64,
    ) -> Vec<Detection> {
        let s = &self.scanner;
        let (width, height) = self.fhog_window;
        let cell = s.cell_size as usize;
        let det_box = (
            width - 2 * s.padding as usize,
            height - 2 * s.padding as usize,
        );
        let pad = (height as i64, width as i64);
        let num_levels = self.num_pyramid_levels(img.width(), img.height());
        let dims = s.num_dimensions;

        let mut per_filter: Vec<Vec<(f64, Rect)>> = vec![Vec::new(); self.banks.len()];
        let (mut saliency, mut scratch) = (FloatImage::default(), FilterScratch::default());
        let mut level_img: Option<Image<C>> = None;
        for l in 0..num_levels {
            if l > 0 {
                level_img = Some(match &level_img {
                    None => pyramid_down6(img),
                    Some(prev) => pyramid_down6(prev),
                });
            }
            let feats = match &level_img {
                None => extract_fhog(img, cell, height, width),
                Some(li) => extract_fhog(li, cell, height, width),
            };
            for (i, bank) in self.banks.iter().enumerate() {
                let area = apply_filters(bank, &feats, &mut saliency, &mut scratch);
                let thresh = self.weights[i][dims] + adjust_threshold;
                scan_level(
                    &saliency,
                    area,
                    thresh,
                    det_box,
                    s.cell_size as i64,
                    pad,
                    |r| pyramid_down6_rect_up(r, l),
                    &mut per_filter[i],
                );
            }
        }

        let mut accum = Vec::new();
        for (i, mut dets) in per_filter.into_iter().enumerate() {
            // std::sort(dets.rbegin(), dets.rend()): decreasing score.
            dets.sort_by(|a, b| b.0.total_cmp(&a.0));
            let thresh = self.weights[i][dims];
            accum.extend(dets.into_iter().map(|(score, rect)| Detection {
                rect,
                confidence: score - thresh,
                weight_index: i,
            }));
        }
        if self.banks.len() > 1 {
            accum.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        }
        let mut finals: Vec<Detection> = Vec::new();
        for d in accum {
            if !finals.iter().any(|f| self.overlap.overlaps(f.rect, d.rect)) {
                finals.push(d);
            }
        }
        finals
    }

    /// What dlib's Python `detector.run(img, upsample_num_times, adjust_threshold)` does:
    /// `pyramid_up` the image `upsample` times, detect, and map the rectangles back with
    /// `pyramid_down<2>::rect_down(rect, upsample)`.
    pub fn detect_upsampled<const C: usize>(
        &self,
        img: ImageView<'_, C>,
        upsample: u32,
        adjust_threshold: f64,
    ) -> Result<Vec<Detection>, DetectError> {
        if upsample == 0 {
            return Ok(self.detect(&img, adjust_threshold));
        }
        let (mut w, mut h) = (img.width() as u128, img.height() as u128);
        for _ in 0..upsample {
            if w * h == 0 {
                break;
            }
            // pyramid_up output: (2h + 1) x (2w + 2).
            (w, h) = (2 * w + 2, 2 * h + 1);
            if w * h > MAX_UPSAMPLED_PIXELS {
                return Err(DetectError::UpsampleTooLarge {
                    width: img.width(),
                    height: img.height(),
                    upsample,
                });
            }
        }
        let mut up = pyramid_up(&img);
        for _ in 1..upsample {
            up = pyramid_up(&up);
        }
        let mut dets = self.detect(&up, adjust_threshold);
        for d in &mut dets {
            d.rect = pyramid_down2_rect_down_levels(d.rect, upsample as usize);
        }
        Ok(dets)
    }

    /// Number of levels of `impl::create_fhog_pyramid` for a `width × height` image.
    fn num_pyramid_levels(&self, width: usize, height: usize) -> usize {
        let s = &self.scanner;
        let mut rect = Rect::new(0, 0, width as i64 - 1, height as i64 - 1);
        let mut levels = 0u64;
        loop {
            let next = pyramid_down6_rect_down(rect);
            levels += 1;
            // A small level can map to itself (1x1 stays 1x1). After that, dlib makes copies of
            // it up to max_pyramid_levels, which a file can set to 2^64 - 1. Stop at the first
            // copy. Real detectors stop much earlier, at min_pyramid_layer_width/height.
            if next == rect {
                break;
            }
            rect = next;
            if !(rect.width() >= s.min_pyramid_layer_width
                && rect.height() >= s.min_pyramid_layer_height
                && levels < s.max_pyramid_levels)
            {
                break;
            }
        }
        levels as usize
    }

    fn parse(data: &[u8]) -> serialize::Result<Self> {
        let mut r = Reader::new(data);
        r.context(
            || "object_detector",
            |r| {
                let version = r.int_version("dlib::object_detector", 1..=2)?;
                let scanner = r.context(|| "scanner", read_scanner)?;
                let (overlap, weights) = if version == 1 {
                    let w = r.context(|| "w", Reader::col_vector_f64)?;
                    let overlap = r.context(|| "boxes_overlap", read_overlap)?;
                    (overlap, vec![w])
                } else {
                    let overlap = r.context(|| "boxes_overlap", read_overlap)?;
                    let weights = r.context(|| "w", |r| r.vec(4, |r, _| r.col_vector_f64()))?;
                    (overlap, weights)
                };
                let fhog_window = fhog_window_size(&scanner).ok_or_else(|| {
                    r.invalid(format!(
                        "invalid detection window {}x{} with padding {}",
                        scanner.window_width, scanner.window_height, scanner.padding
                    ))
                })?;
                // dlib's deserialize checks the stored dimensions against the FHOG geometry.
                let dims = fhog_window.0 * fhog_window.1 * NUM_PLANES;
                if scanner.num_dimensions != dims {
                    return Err(r.invalid(format!(
                        "scanner stores {} dimensions but its {}x{} FHOG window needs {dims}",
                        scanner.num_dimensions, fhog_window.0, fhog_window.1
                    )));
                }
                if weights.is_empty() {
                    return Err(r.invalid("object detector without weight vectors"));
                }
                for (i, w) in weights.iter().enumerate() {
                    // The detection threshold is w[dims], so one more element than the filters.
                    if w.len() <= dims {
                        return Err(r.invalid(format!(
                            "weight vector {i} has {} elements, the scanner needs {}",
                            w.len(),
                            dims + 1
                        )));
                    }
                }
                let banks = weights
                    .iter()
                    .map(|w| FilterBank::build(w, fhog_window.1, fhog_window.0))
                    .collect();
                Ok(FhogObjectDetector {
                    scanner,
                    overlap,
                    weights,
                    banks,
                    fhog_window,
                })
            },
        )
    }
}

fn read_scanner(r: &mut Reader) -> serialize::Result<ScanFhogPyramidConfig> {
    r.int_version("dlib::scan_fhog_pyramid", 1..=1)?;
    // default_fhog_feature_extractor serializes nothing.
    r.context(
        || "feats",
        |r| {
            let _max_size = r.u64()?;
            let size = r.u64()?;
            if size != 0 {
                return Err(r.invalid(format!(
                    "scanner holds {size} cached FHOG images, which manyfaced does not read"
                )));
            }
            Ok(())
        },
    )?;
    let cell_size = r.i32()?;
    // dlib has a separate FHOG implementation for cell_size 1, which manyfaced does not port.
    let cell_size = u32::try_from(cell_size)
        .ok()
        .filter(|&c| c > 1)
        .ok_or_else(|| r.invalid(format!("unsupported cell_size {cell_size} (must be >= 2)")))?;
    let padding = r.u64()?;
    let window_width = r.u64()?;
    let window_height = r.u64()?;
    if window_width == 0 || window_height == 0 {
        return Err(r.invalid(format!(
            "invalid detection window {window_width}x{window_height}"
        )));
    }
    let max_pyramid_levels = r.u64()?;
    let min_pyramid_layer_width = r.u64()?;
    let min_pyramid_layer_height = r.u64()?;
    // dlib's set_min_pyramid_layer_size asserts this. With a size of 0, the pyramid of small
    // levels never ends (a 1x1 level maps to itself) until max_pyramid_levels, which a file can
    // set to 2^64 - 1.
    if min_pyramid_layer_width == 0 || min_pyramid_layer_height == 0 {
        return Err(r.invalid(format!(
            "invalid minimum pyramid layer size {min_pyramid_layer_width}x{min_pyramid_layer_height}"
        )));
    }
    let nuclear_norm_regularization_strength = r.f64()?;
    // Checked against the FHOG window geometry by the caller.
    let dims = r.i64()?;
    let num_dimensions = usize::try_from(dims)
        .map_err(|_| r.invalid(format!("invalid number of dimensions {dims}")))?;
    Ok(ScanFhogPyramidConfig {
        cell_size,
        padding,
        window_width,
        window_height,
        max_pyramid_levels,
        min_pyramid_layer_width,
        min_pyramid_layer_height,
        nuclear_norm_regularization_strength,
        num_dimensions,
    })
}

fn read_overlap(r: &mut Reader) -> serialize::Result<BoxOverlap> {
    let iou_thresh = r.f64()?;
    let percent_covered_thresh = r.f64()?;
    Ok(BoxOverlap {
        iou_thresh,
        percent_covered_thresh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FhogObjectDetector>();
    };

    #[test]
    fn embedded_frontal_face_detector() {
        let d = FhogObjectDetector::from_bytes(FRONTAL_FACE_DAT).unwrap();
        let s = d.scanner();
        assert_eq!(d.num_detectors(), 5);
        assert_eq!(s.cell_size(), 8);
        assert_eq!(s.padding(), 0);
        assert_eq!((s.window_width(), s.window_height()), (80, 80));
        assert_eq!(s.max_pyramid_levels(), 1000);
        assert_eq!(
            (s.min_pyramid_layer_width(), s.min_pyramid_layer_height()),
            (64, 64)
        );
        assert_eq!(s.nuclear_norm_regularization_strength(), 9.0);
        assert_eq!(s.num_dimensions(), 3100);
        for w in d.weights() {
            assert_eq!(w.len(), 3101);
            assert!(w.iter().all(|v| v.is_finite()));
        }
        let o = d.overlap();
        assert!(o.iou_thresh > 0.0 && o.iou_thresh < 1.0, "{o:?}");
        assert!(
            o.percent_covered_thresh > 0.0 && o.percent_covered_thresh <= 1.0,
            "{o:?}"
        );
        // The cached copy is the same detector.
        let f = FhogObjectDetector::frontal_face();
        assert_eq!(f.weights(), d.weights());
        assert_eq!(f.overlap(), d.overlap());
    }

    #[test]
    fn truncations_are_errors() {
        let n = FRONTAL_FACE_DAT.len();
        for cut in (0..64).chain((64..n).step_by(997)).chain([n - 1]) {
            let e = FhogObjectDetector::from_bytes(&FRONTAL_FACE_DAT[..cut]).unwrap_err();
            assert!(e.to_string().contains("object_detector"), "{e}");
        }
    }

    #[test]
    fn wrong_version_is_rejected() {
        let mut bytes = FRONTAL_FACE_DAT.to_vec();
        assert_eq!(bytes[..2], [0x01, 0x02]);
        bytes[1] = 3;
        let e = FhogObjectDetector::from_bytes(&bytes)
            .unwrap_err()
            .to_string();
        assert!(e.contains("unexpected version \"3\""), "{e}");
        assert!(e.contains("dlib::object_detector"), "{e}");
    }

    #[test]
    fn cached_fhog_images_are_rejected() {
        let mut bytes = FRONTAL_FACE_DAT.to_vec();
        // feats size is the second u64 after the two version ints: 01 02 01 01 01 00 [01 00].
        assert_eq!(bytes[6..8], [0x01, 0x00]);
        bytes[7] = 1;
        let e = FhogObjectDetector::from_bytes(&bytes)
            .unwrap_err()
            .to_string();
        assert!(e.contains("cached FHOG images"), "{e}");
    }
}
