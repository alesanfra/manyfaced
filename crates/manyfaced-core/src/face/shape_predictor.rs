//! Ensemble of Regression Trees landmark predictor (Kazemi & Sullivan, 2014).
//!
//! Ported from `dlib/image_processing/shape_predictor.h` (dlib v20.0.1, Boost Software
//! License 1.0, see `THIRD_PARTY_NOTICES`). Loads dlib's `shape_predictor_*.dat` files.

use std::path::Path;

use crate::geometry::{
    DPoint, Point, PointTransformAffine, Rect, find_affine_transform, find_similarity_transform,
};
use crate::image::ImageView;
use crate::serialize::{self, Reader};

/// A node test: go to the left child iff `px[idx1] - px[idx2] > thresh`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitFeature {
    pub idx1: u32,
    pub idx2: u32,
    pub thresh: f32,
}

/// A complete binary regression tree stored as an implicit heap: the children of node `i` are
/// `2i + 1` and `2i + 2`, and leaf `j` follows the last split.
#[derive(Debug, Clone)]
pub struct RegressionTree {
    splits: Vec<SplitFeature>,
    /// `num_leaves` shape deltas of `shape_len` values each, concatenated.
    leaf_values: Vec<f32>,
    shape_len: usize,
}

impl RegressionTree {
    pub fn splits(&self) -> &[SplitFeature] {
        &self.splits
    }

    pub fn num_leaves(&self) -> usize {
        self.splits.len() + 1
    }

    /// Number of split levels (`num_leaves == 2^depth`).
    pub fn depth(&self) -> u32 {
        self.num_leaves().trailing_zeros()
    }

    /// The shape delta stored in leaf `i`.
    pub fn leaf(&self, i: usize) -> &[f32] {
        &self.leaf_values[i * self.shape_len..(i + 1) * self.shape_len]
    }

    /// Walks the tree on the feature pixel values and returns the index of the leaf reached
    /// (`regression_tree::operator()`). Indices were validated on load.
    #[inline]
    pub fn leaf_index(&self, pixels: &[f32]) -> usize {
        let mut i = 0;
        while i < self.splits.len() {
            let s = self.splits[i];
            i = if pixels[s.idx1 as usize] - pixels[s.idx2 as usize] > s.thresh {
                2 * i + 1
            } else {
                2 * i + 2
            };
        }
        i - self.splits.len()
    }
}

/// One level of the cascade: its trees and the feature pixels they read, encoded relative to
/// the initial shape (`anchor` landmark + `delta` offset in normalized shape space).
#[derive(Debug, Clone)]
pub struct Cascade {
    trees: Vec<RegressionTree>,
    anchor_idx: Vec<u32>,
    deltas: Vec<[f32; 2]>,
}

impl Cascade {
    pub fn trees(&self) -> &[RegressionTree] {
        &self.trees
    }

    pub fn num_feature_pixels(&self) -> usize {
        self.anchor_idx.len()
    }

    pub fn anchor_indices(&self) -> &[u32] {
        &self.anchor_idx
    }

    pub fn deltas(&self) -> &[[f32; 2]] {
        &self.deltas
    }
}

/// dlib's `shape_predictor`. Immutable after loading, hence `Send + Sync`.
#[derive(Debug, Clone)]
pub struct ShapePredictor {
    /// Mean shape in normalized coordinates, `[x0, y0, x1, y1, ...]`.
    initial_shape: Vec<f32>,
    cascades: Vec<Cascade>,
}

const WHAT: &str = "dlib shape_predictor";

impl ShapePredictor {
    /// Loads a dlib `shape_predictor` `.dat` file.
    pub fn load(path: impl AsRef<Path>) -> crate::Result<Self> {
        crate::load_file(path.as_ref(), WHAT, Self::parse)
    }

    /// Parses the contents of a dlib `shape_predictor` `.dat` file. Like dlib, trailing bytes
    /// after the predictor are ignored.
    pub fn from_bytes(data: &[u8]) -> crate::Result<Self> {
        crate::load_bytes(data, WHAT, Self::parse)
    }

    pub fn num_parts(&self) -> usize {
        self.initial_shape.len() / 2
    }

    pub fn initial_shape(&self) -> &[f32] {
        &self.initial_shape
    }

    pub fn cascades(&self) -> &[Cascade] {
        &self.cascades
    }

    /// Total number of leaves, as dlib's `shape_predictor::num_features()`.
    pub fn num_features(&self) -> usize {
        self.cascades
            .iter()
            .flat_map(|c| &c.trees)
            .map(RegressionTree::num_leaves)
            .sum()
    }

    /// Predicts the landmarks of the object in `rect` (`shape_predictor::operator()`), in image
    /// coordinates. `C` is 1 for grayscale and 3 for RGB images (RGB pixels are read through
    /// dlib's `get_pixel_intensity`, `(r + g + b) / 3`). Any rectangle is accepted; feature
    /// pixels that fall outside the image read as 0, exactly like dlib.
    pub fn predict<const C: usize>(&self, image: ImageView<'_, C>, rect: Rect) -> Vec<Point> {
        let mut shape = self.initial_shape.clone();
        let mut pixels = Vec::new();
        let tform_to_img = unnormalizing_tform(rect);
        let mut from = Vec::with_capacity(self.num_parts());
        let mut to = Vec::with_capacity(self.num_parts());
        for cascade in &self.cascades {
            let tform = tform_between_shapes(&self.initial_shape, &shape, &mut from, &mut to);
            extract_feature_pixel_values(image, &tform_to_img, &shape, tform, cascade, &mut pixels);
            for tree in &cascade.trees {
                let leaf = tree.leaf(tree.leaf_index(&pixels));
                for (v, &d) in shape.iter_mut().zip(leaf) {
                    *v += d;
                }
            }
        }
        shape
            .chunks_exact(2)
            .map(|p| {
                tform_to_img
                    .apply(DPoint::new(f64::from(p[0]), f64::from(p[1])))
                    .round()
            })
            .collect()
    }

    fn parse(data: &[u8]) -> serialize::Result<Self> {
        let mut r = Reader::new(data);
        r.context(
            || "shape_predictor",
            |r| {
                r.int_version("dlib::shape_predictor", 1..=1)?;
                let initial_shape = r.context(|| "initial_shape", Reader::col_vector_f32)?;
                if initial_shape.is_empty() || initial_shape.len() % 2 != 0 {
                    return Err(r.invalid(format!(
                        "initial_shape has {} values, expected a non-empty even count",
                        initial_shape.len()
                    )));
                }
                let shape_len = initial_shape.len();
                let forests = r.context(
                    || "forests",
                    |r| r.vec(2, |r, _| r.vec(4, |r, _| read_tree(r, shape_len))),
                )?;
                let anchor_idx = r.context(
                    || "anchor_idx",
                    |r| r.vec(2, |r, _| r.vec(2, |r, _| r.u64())),
                )?;
                let deltas = r.context(
                    || "deltas",
                    |r| r.vec(2, |r, _| r.vec(8, |r, _| Ok([r.f32()?, r.f32()?]))),
                )?;
                let cascades = build_cascades(r, shape_len / 2, forests, anchor_idx, deltas)?;
                Ok(ShapePredictor {
                    initial_shape,
                    cascades,
                })
            },
        )
    }
}

/// `impl::unnormalizing_tform(rect)`: maps (0,0) to the top-left and (1,1) to the bottom-right
/// corner of `rect`, via `find_affine_transform` on three corners. dlib stores the corners as
/// `vector<float,2>`, hence the `f32` round trip.
fn unnormalizing_tform(rect: Rect) -> PointTransformAffine {
    let corner = |p: Point| DPoint::new(f64::from(p.x as f32), f64::from(p.y as f32));
    let from = [
        DPoint::new(0.0, 0.0),
        DPoint::new(1.0, 0.0),
        DPoint::new(1.0, 1.0),
    ];
    let to = [
        corner(rect.tl_corner()),
        corner(rect.tr_corner()),
        corner(rect.br_corner()),
    ];
    find_affine_transform(&from, &to)
}

/// `impl::find_tform_between_shapes(from_shape, to_shape)`, returned as dlib's
/// `matrix_cast<float>(... .get_m())`. `from`/`to` are reused scratch buffers.
fn tform_between_shapes(
    from_shape: &[f32],
    to_shape: &[f32],
    from: &mut Vec<DPoint>,
    to: &mut Vec<DPoint>,
) -> [[f32; 2]; 2] {
    if from_shape.len() == 2 {
        // A single landmark: dlib uses the identity transform.
        return [[1.0, 0.0], [0.0, 1.0]];
    }
    let points = |shape: &[f32], out: &mut Vec<DPoint>| {
        out.clear();
        out.extend(
            shape
                .chunks_exact(2)
                .map(|p| DPoint::new(f64::from(p[0]), f64::from(p[1]))),
        );
    };
    points(from_shape, from);
    points(to_shape, to);
    let m = find_similarity_transform(from, to).m;
    [
        [m[0][0] as f32, m[0][1] as f32],
        [m[1][0] as f32, m[1][1] as f32],
    ]
}

/// `impl::extract_feature_pixel_values`: for each feature pixel, maps its delta through the
/// current shape's transform (in `f32`), adds its anchor landmark (in `f32`), maps the result
/// to image space (in `f64`) and rounds it to the nearest pixel.
fn extract_feature_pixel_values<const C: usize>(
    image: ImageView<'_, C>,
    tform_to_img: &PointTransformAffine,
    shape: &[f32],
    tform: [[f32; 2]; 2],
    cascade: &Cascade,
    out: &mut Vec<f32>,
) {
    out.clear();
    let (w, h) = (image.width() as i64, image.height() as i64);
    for (&anchor, &[dx, dy]) in cascade.anchor_idx.iter().zip(&cascade.deltas) {
        let a = anchor as usize;
        // tform * delta: dlib's matrix product accumulates t(r,0)*d0 then += t(r,1)*d1.
        let x = (tform[0][0] * dx + tform[0][1] * dy) + shape[2 * a];
        let y = (tform[1][0] * dx + tform[1][1] * dy) + shape[2 * a + 1];
        let p = tform_to_img
            .apply(DPoint::new(f64::from(x), f64::from(y)))
            .round();
        let value = if p.x >= 0 && p.x < w && p.y >= 0 && p.y < h {
            image.intensity(p.x as usize, p.y as usize)
        } else {
            0
        };
        out.push(f32::from(value));
    }
}

fn read_tree(r: &mut Reader, shape_len: usize) -> serialize::Result<RegressionTree> {
    let splits = r.context(
        || "splits",
        |r| {
            r.vec(8, |r, _| {
                let idx1 = r.u64()?;
                let idx2 = r.u64()?;
                let thresh = r.f32()?;
                let to_u32 = |v: u64| {
                    u32::try_from(v).map_err(|_| r.invalid(format!("split index {v} too large")))
                };
                Ok(SplitFeature {
                    idx1: to_u32(idx1)?,
                    idx2: to_u32(idx2)?,
                    thresh,
                })
            })
        },
    )?;
    let leaves = r.context(|| "leaf_values", |r| r.vec(4, |r, _| r.col_vector_f32()))?;
    // Inference walks the implicit heap until it leaves the split array, which is only
    // well-defined for a complete tree.
    if !leaves.len().is_power_of_two() || leaves.len() != splits.len() + 1 {
        return Err(r.invalid(format!(
            "tree with {} splits and {} leaves is not a complete binary tree",
            splits.len(),
            leaves.len()
        )));
    }
    let mut leaf_values = Vec::with_capacity(leaves.len() * shape_len);
    for (i, leaf) in leaves.iter().enumerate() {
        if leaf.len() != shape_len {
            return Err(r.invalid(format!(
                "leaf {i} has {} values, expected {shape_len}",
                leaf.len()
            )));
        }
        leaf_values.extend_from_slice(leaf);
    }
    Ok(RegressionTree {
        splits,
        leaf_values,
        shape_len,
    })
}

fn build_cascades(
    r: &Reader,
    num_parts: usize,
    forests: Vec<Vec<RegressionTree>>,
    anchor_idx: Vec<Vec<u64>>,
    deltas: Vec<Vec<[f32; 2]>>,
) -> serialize::Result<Vec<Cascade>> {
    if forests.len() != anchor_idx.len() || forests.len() != deltas.len() {
        return Err(r.invalid(format!(
            "{} forests but {} anchor lists and {} delta lists",
            forests.len(),
            anchor_idx.len(),
            deltas.len()
        )));
    }
    let mut cascades = Vec::with_capacity(forests.len());
    for (c, ((trees, anchors), deltas)) in
        forests.into_iter().zip(anchor_idx).zip(deltas).enumerate()
    {
        if anchors.len() != deltas.len() {
            return Err(r.invalid(format!(
                "cascade {c}: {} anchors but {} deltas",
                anchors.len(),
                deltas.len()
            )));
        }
        let anchor_idx = anchors
            .iter()
            .map(|&a| {
                usize::try_from(a)
                    .ok()
                    .filter(|&a| a < num_parts)
                    .map(|a| a as u32)
                    .ok_or_else(|| {
                        r.invalid(format!("cascade {c}: anchor {a} out of {num_parts} parts"))
                    })
            })
            .collect::<serialize::Result<Vec<_>>>()?;
        let num_pixels = anchor_idx.len();
        for (t, tree) in trees.iter().enumerate() {
            if let Some(s) = tree
                .splits
                .iter()
                .find(|s| s.idx1 as usize >= num_pixels || s.idx2 as usize >= num_pixels)
            {
                return Err(r.invalid(format!(
                    "cascade {c} tree {t}: split reads pixels ({}, {}) but the cascade has {num_pixels}",
                    s.idx1, s.idx2
                )));
            }
        }
        cascades.push(Cascade {
            trees,
            anchor_idx,
            deltas,
        });
    }
    Ok(cascades)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::serialize::writer::Writer;

    /// A tiny valid predictor: 2 parts, 1 cascade, 1 depth-1 tree, 2 feature pixels.
    pub(crate) fn tiny_predictor_bytes() -> Vec<u8> {
        let mut w = Writer::default();
        w.i64(1);
        w.matrix_f32(4, 1, &[0.25, 0.5, 0.75, 0.5]);
        // forests: 1 cascade, 1 tree
        w.u64(1);
        w.u64(1);
        w.u64(1); // splits
        w.u64(0);
        w.u64(1);
        w.f32(-3.5);
        w.u64(2); // leaf_values
        w.matrix_f32(4, 1, &[0.1, 0.0, -0.1, 0.0]);
        w.matrix_f32(4, 1, &[0.0, 0.2, 0.0, -0.2]);
        // anchor_idx
        w.u64(1);
        w.u64(2);
        w.u64(0);
        w.u64(1);
        // deltas
        w.u64(1);
        w.u64(2);
        for v in [0.01f32, -0.02, 0.03, 0.04] {
            w.f32(v);
        }
        w.buf
    }

    #[test]
    fn parses_tiny_predictor() {
        let sp = ShapePredictor::from_bytes(&tiny_predictor_bytes()).unwrap();
        assert_eq!(sp.num_parts(), 2);
        assert_eq!(sp.cascades().len(), 1);
        let c = &sp.cascades()[0];
        assert_eq!(c.num_feature_pixels(), 2);
        assert_eq!(c.anchor_indices(), [0, 1]);
        assert_eq!(c.deltas(), [[0.01, -0.02], [0.03, 0.04]]);
        let t = &c.trees()[0];
        assert_eq!(t.depth(), 1);
        assert_eq!(
            t.splits(),
            [SplitFeature {
                idx1: 0,
                idx2: 1,
                thresh: -3.5
            }]
        );
        assert_eq!(t.leaf(1), [0.0, 0.2, 0.0, -0.2]);
        assert_eq!(sp.num_features(), 2);
    }

    #[test]
    fn predict_follows_the_tree() {
        use crate::image::{GrayView, RgbView};
        let sp = ShapePredictor::from_bytes(&tiny_predictor_bytes()).unwrap();
        let rect = Rect::new(0, 0, 100, 100);
        // Feature pixels land on (26, 48) and (78, 54). Uniform image: 0 - 0 > -3.5 holds, so
        // the left leaf (+0.1 / -0.1 in x) is taken.
        let flat = vec![7u8; 101 * 101];
        let gray = GrayView::packed(&flat, 101, 101).unwrap();
        assert_eq!(
            sp.predict(gray, rect),
            [Point::new(35, 50), Point::new(65, 50)]
        );
        // px0 - px1 = 0 - 10 fails the test: right leaf (+0.2 / -0.2 in y).
        let mut img = vec![0u8; 101 * 101];
        img[54 * 101 + 78] = 10;
        let gray = GrayView::packed(&img, 101, 101).unwrap();
        assert_eq!(
            sp.predict(gray, rect),
            [Point::new(25, 70), Point::new(75, 30)]
        );
        // The same through RGB intensity: (9 + 10 + 11) / 3 = 10.
        let mut rgb = vec![0u8; 101 * 101 * 3];
        rgb[(54 * 101 + 78) * 3..(54 * 101 + 78) * 3 + 3].copy_from_slice(&[9, 10, 11]);
        let rgb = RgbView::packed(&rgb, 101, 101).unwrap();
        assert_eq!(
            sp.predict(rgb, rect),
            [Point::new(25, 70), Point::new(75, 30)]
        );
        // Feature pixels outside the image read as 0: an empty image takes the left leaf.
        let empty = GrayView::packed(&[], 0, 0).unwrap();
        assert_eq!(
            sp.predict(empty, rect),
            [Point::new(35, 50), Point::new(65, 50)]
        );
    }

    #[test]
    fn every_truncation_is_an_error() {
        let bytes = tiny_predictor_bytes();
        for n in 0..bytes.len() {
            let e = ShapePredictor::from_bytes(&bytes[..n]).unwrap_err();
            assert!(e.to_string().contains("shape_predictor"), "{e}");
        }
    }

    #[test]
    fn out_of_range_split_index_is_rejected() {
        let mut bytes = tiny_predictor_bytes();
        // The split's idx2 (1) is encoded as [0x01, 0x01] right after idx1 [0x01, 0x00].
        let pos = bytes.windows(4).position(|w| w == [1, 0, 1, 1]).unwrap();
        bytes[pos + 3] = 7;
        let e = ShapePredictor::from_bytes(&bytes).unwrap_err().to_string();
        assert!(e.contains("split reads pixels (0, 7)"), "{e}");
    }

    #[test]
    fn wrong_version_is_rejected() {
        let mut bytes = tiny_predictor_bytes();
        bytes[1] = 2;
        let e = ShapePredictor::from_bytes(&bytes).unwrap_err().to_string();
        assert!(e.contains("unexpected version \"2\""), "{e}");
    }
}
