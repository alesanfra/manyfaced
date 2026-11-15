//! dlib's face recognition ResNet (`dlib_face_recognition_resnet_model_v1.dat`).
//!
//! The network type is ported from `examples/dnn_face_recognition_ex.cpp` (dlib v20.0.1, Boost
//! Software License 1.0, see `THIRD_PARTY_NOTICES`).

use std::path::Path;

use crate::dnn::spec::{InputSpec, LayerSpec, LossSpec, NetSpec};
use crate::dnn::{Backend, ForwardError, Network, network};
use crate::face::ChipError;
use crate::face::chip::{ChipDetails, extract_image_chip, extract_image_chips, face_chip_details};
use crate::geometry::{DPoint, DRect, Point, Rect};
use crate::image::{Image, ImageView, RgbImage, RgbView};
use crate::serialize::{self, Reader};

/// Builds a [`NetSpec`] in execution order, mirroring the template aliases of
/// `dnn_face_recognition_ex.cpp`.
struct Builder(Vec<LayerSpec>);

impl Builder {
    fn con(&mut self, num_filters: usize, size: usize, stride: usize) {
        self.0.push(LayerSpec::Con {
            num_filters,
            nr: size,
            nc: size,
            stride_y: stride,
            stride_x: stride,
        });
    }

    /// `block<N,affine,stride> = affine<con<N,3,3,1,1,relu<affine<con<N,3,3,stride,stride>>>>>>`
    fn block(&mut self, n: usize, stride: usize) {
        self.con(n, 3, stride);
        self.0.push(LayerSpec::Affine);
        self.0.push(LayerSpec::Relu);
        self.con(n, 3, 1);
        self.0.push(LayerSpec::Affine);
    }

    /// `ares<N> = relu<add_prev1<block<N,affine,1,tag1<SUBNET>>>>`
    fn ares(&mut self, n: usize) {
        self.0.push(LayerSpec::Tag(1));
        self.block(n, 1);
        self.0.push(LayerSpec::AddPrev { tag: 1 });
        self.0.push(LayerSpec::Relu);
    }

    /// `ares_down<N> = relu<add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,affine,2,tag1<SUBNET>>>>>>>>`
    fn ares_down(&mut self, n: usize) {
        self.0.push(LayerSpec::Tag(1));
        self.block(n, 2);
        self.0.push(LayerSpec::Tag(2));
        self.0.push(LayerSpec::Skip(1));
        self.0.push(LayerSpec::AvgPool {
            nr: 2,
            nc: 2,
            stride_y: 2,
            stride_x: 2,
        });
        self.0.push(LayerSpec::AddPrev { tag: 2 });
        self.0.push(LayerSpec::Relu);
    }
}

/// The type of `anet_type` in `dnn_face_recognition_ex.cpp`:
///
/// ```text
/// loss_metric<fc_no_bias<128,avg_pool_everything<
///     alevel0<alevel1<alevel2<alevel3<alevel4<
///     max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,input_rgb_image_sized<150>>>>>>>>>>>>>
/// ```
pub fn resnet_v1_spec() -> NetSpec {
    let mut b = Builder(Vec::new());
    b.con(32, 7, 2);
    b.0.push(LayerSpec::Affine);
    b.0.push(LayerSpec::Relu);
    b.0.push(LayerSpec::MaxPool {
        nr: 3,
        nc: 3,
        stride_y: 2,
        stride_x: 2,
    });
    // alevel4 = ares<32,ares<32,ares<32,SUBNET>>>
    for _ in 0..3 {
        b.ares(32);
    }
    // alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>
    b.ares_down(64);
    for _ in 0..3 {
        b.ares(64);
    }
    // alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>
    b.ares_down(128);
    for _ in 0..2 {
        b.ares(128);
    }
    // alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>
    b.ares_down(256);
    for _ in 0..2 {
        b.ares(256);
    }
    // alevel0 = ares_down<256,SUBNET>
    b.ares_down(256);
    // avg_pool_everything = avg_pool<0,0,1,1>
    b.0.push(LayerSpec::AvgPool {
        nr: 0,
        nc: 0,
        stride_y: 1,
        stride_x: 1,
    });
    b.0.push(LayerSpec::FcNoBias { num_outputs: 128 });
    NetSpec {
        input: InputSpec::RgbImageSized { nr: 150, nc: 150 },
        layers: b.0,
        loss: LossSpec::Metric,
    }
}

/// dlib's `face_recognition_model_v1`: maps a 150×150 face chip to a 128-D descriptor.
/// Immutable after loading, hence `Send + Sync`.
#[derive(Debug, Clone)]
pub struct FaceRecognitionModel {
    net: Network,
}

const WHAT: &str = "dlib face recognition ResNet";

impl FaceRecognitionModel {
    /// Loads `dlib_face_recognition_resnet_model_v1.dat`.
    pub fn load(path: impl AsRef<Path>) -> crate::Result<Self> {
        crate::load_file(path.as_ref(), WHAT, Self::parse)
    }

    pub fn from_bytes(data: &[u8]) -> crate::Result<Self> {
        crate::load_bytes(data, WHAT, Self::parse)
    }

    pub fn network(&self) -> &Network {
        &self.net
    }

    fn parse(data: &[u8]) -> serialize::Result<Self> {
        let mut r = Reader::new(data);
        let net = network::parse(&mut r, &resnet_v1_spec())?;
        Ok(FaceRecognitionModel { net })
    }

    /// Side of the square face chips the network takes (150).
    pub fn chip_size(&self) -> usize {
        self.net.input.nr
    }

    /// The descriptor of an aligned `chip_size()`×`chip_size()` RGB face chip, as extracted by
    /// [`face_chip_details`] + [`extract_image_chip`] (dlib's `get_face_chip`).
    pub fn descriptor(&self, chip: RgbView<'_>) -> Result<Vec<f32>, RecognitionError> {
        self.descriptor_with(chip, Backend::default())
    }

    /// [`descriptor`](Self::descriptor) with an explicit backend (the naive one is a test
    /// reference).
    pub fn descriptor_with(
        &self,
        chip: RgbView<'_>,
        backend: Backend,
    ) -> Result<Vec<f32>, RecognitionError> {
        let input = self.net.input.to_tensor(chip)?;
        Ok(self.net.forward(input, backend)?)
    }

    /// The descriptor of a chip averaged over `num_jitters` randomly jittered copies, like dlib
    /// with `num_jitters > 1`; with `num_jitters <= 1` this is [`descriptor`](Self::descriptor)
    /// (dlib does not jitter for 1 either). The jitter follows dlib's `jitter_image` but uses
    /// manyfaced's own random generator, so jittered descriptors are not bit-identical to dlib's.
    pub fn jittered_descriptor(
        &self,
        chip: RgbView<'_>,
        num_jitters: u32,
        rng: &mut JitterRng,
    ) -> Result<Vec<f32>, RecognitionError> {
        if num_jitters <= 1 {
            return self.descriptor(chip);
        }
        let mut sum: Option<Vec<f32>> = None;
        for _ in 0..num_jitters {
            let crop = jitter_image(chip, rng)?;
            let d = self.descriptor(crop.view())?;
            match &mut sum {
                None => sum = Some(d),
                Some(s) => s.iter_mut().zip(&d).for_each(|(a, b)| *a += b),
            }
        }
        // mean(): sum / n, with dlib's matrix division by a scalar = multiplication by 1/n.
        let inv = 1.0 / num_jitters as f32;
        Ok(sum
            .unwrap_or_default()
            .into_iter()
            .map(|v| v * inv)
            .collect())
    }

    /// dlib's `compute_face_descriptor(img, faces, num_jitters, padding)`: extracts all the
    /// faces' chips together (`get_face_chip_details(face, 150, padding)` +
    /// `extract_image_chips`), then computes one descriptor per face. `padding` is a C++ `float`
    /// in dlib's Python API, hence `f32`. `seed` drives the jitter (unused when
    /// `num_jitters <= 1`).
    pub fn compute_face_descriptors<L: AsRef<[DPoint]>>(
        &self,
        image: RgbView<'_>,
        faces: &[L],
        num_jitters: u32,
        padding: f32,
        seed: u64,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let size = self.chip_size() as u64;
        let details = faces
            .iter()
            .map(|f| face_chip_details(f.as_ref(), size, f64::from(padding)))
            .collect::<Result<Vec<_>, _>>()?;
        let chips = extract_image_chips(image, &details)?;
        let mut rng = JitterRng::new(seed);
        chips
            .iter()
            .map(|c| self.jittered_descriptor(c.view(), num_jitters, &mut rng))
            .collect()
    }
}

/// Errors computing face descriptors.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RecognitionError {
    #[error(transparent)]
    Chip(#[from] ChipError),
    #[error(transparent)]
    Forward(#[from] ForwardError),
}

/// The random generator behind descriptor jittering (SplitMix64). Deterministic for a given
/// seed; it is not dlib's `dlib::rand`, so jittered descriptors differ from dlib's.
#[derive(Debug, Clone)]
pub struct JitterRng(u64);

impl JitterRng {
    pub fn new(seed: u64) -> Self {
        JitterRng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn random_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn double_in_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random_double()
    }
}

/// dlib's `jitter_image` (`image_transforms/interpolation.h`): a random crop shifted by up to
/// 2% of the face size, scaled by 0.97–0.99999, rotated by up to ±3 degrees, and mirrored with
/// probability 1/2. Requires a non-empty square image.
pub fn jitter_image(img: RgbView<'_>, rng: &mut JitterRng) -> Result<RgbImage, ChipError> {
    let (rows, cols) = (img.height(), img.width());
    if rows == 0 || rows != cols {
        return Err(ChipError::NotSquare { rows, cols });
    }
    const MAX_ROTATION_DEGREES: f64 = 3.0;
    const MIN_OBJECT_HEIGHT: f64 = 0.97;
    const MAX_OBJECT_HEIGHT: f64 = 0.99999;
    const TRANSLATE_AMOUNT: f64 = 0.02;
    // shrink_rect(get_rect(img), 3)
    let rect = Rect::new(3, 3, cols as i64 - 4, rows as i64 - 4);
    let tx = rng.double_in_range(-TRANSLATE_AMOUNT, TRANSLATE_AMOUNT) * rect.width() as f64;
    let ty = rng.double_in_range(-TRANSLATE_AMOUNT, TRANSLATE_AMOUNT) * rect.height() as f64;
    let translate = DPoint::new(tx, ty).round();
    let scale = rng.double_in_range(MIN_OBJECT_HEIGHT, MAX_OBJECT_HEIGHT);
    let box_size = (rect.height() as f64 / scale) as i64;
    let c = rect.center();
    let center = Point::new(c.x + translate.x, c.y + translate.y);
    // centered_rect(p, w, h)
    let left = center.x - box_size / 2;
    let top = center.y - box_size / 2;
    let crop_rect = Rect::new(left, top, left + box_size - 1, top + box_size - 1);
    let angle = rng.double_in_range(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
        * std::f64::consts::PI
        / 180.0;
    let details = ChipDetails {
        rect: DRect::from(crop_rect),
        angle,
        rows: rows as u64,
        cols: cols as u64,
    };
    let mut crop = extract_image_chip(img, &details)?;
    if rng.random_double() > 0.5 {
        flip_left_right(&mut crop);
    }
    Ok(crop)
}

fn flip_left_right<const C: usize>(img: &mut Image<C>) {
    let (w, h) = (img.width(), img.height());
    let mut flipped = Image::<C>::zeros(w, h);
    let view = img.view();
    for y in 0..h {
        for x in 0..w {
            flipped.set_pixel(w - 1 - x, y, view.pixel(x, y));
        }
    }
    *img = flipped;
}

/// Expands a grayscale image to RGB (`assign_image` from `unsigned char` to `rgb_pixel`).
pub fn gray_to_rgb(img: ImageView<'_, 1>) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbImage::zeros(w, h);
    for y in 0..h {
        for (x, &v) in img.row(y).iter().enumerate() {
            out.set_pixel(x, y, [v, v, v]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FaceRecognitionModel>();
    };

    #[test]
    fn jitter_is_deterministic_and_keeps_the_size() {
        let data: Vec<u8> = (0..20 * 20 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let img = RgbView::packed(&data, 20, 20).unwrap();
        let a = jitter_image(img, &mut JitterRng::new(7)).unwrap();
        let b = jitter_image(img, &mut JitterRng::new(7)).unwrap();
        assert_eq!(a, b);
        assert_eq!((a.width(), a.height()), (20, 20));
        assert!(
            jitter_image(
                RgbView::packed(&data[..60], 20, 1).unwrap(),
                &mut JitterRng::new(0)
            )
            .is_err()
        );
    }

    #[test]
    fn rng_range() {
        let mut r = JitterRng::new(1);
        for _ in 0..1000 {
            let v = r.double_in_range(-3.0, 3.0);
            assert!((-3.0..3.0).contains(&v));
        }
    }

    #[test]
    fn spec_matches_the_cpp_type() {
        let spec = resnet_v1_spec();
        let count = |f: fn(&LayerSpec) -> bool| spec.layers.iter().filter(|l| f(l)).count();
        // 1 stem conv + 2 per residual block (3 + 4 + 3 + 3 + 1 = 14 blocks).
        assert_eq!(count(|l| matches!(l, LayerSpec::Con { .. })), 29);
        assert_eq!(count(|l| matches!(l, LayerSpec::Affine)), 29);
        assert_eq!(count(|l| matches!(l, LayerSpec::AddPrev { .. })), 14);
        assert_eq!(count(|l| matches!(l, LayerSpec::Skip(_))), 4);
        // Stem (con, affine, relu, max_pool), 10 `ares` of 8 layers, 4 `ares_down` of 11
        // layers, avg_pool_everything and fc_no_bias.
        assert_eq!(spec.layers.len(), 4 + 10 * 8 + 4 * 11 + 2);
    }
}
