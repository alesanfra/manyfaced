//! Structure tests on the original dlib model files.
//!
//! The models are not committed: run `python3 tools/download_models.py` first (CI does), or set
//! `MANYFACED_MODELS_DIR`. A missing model is a test failure, not a skip.

use std::path::PathBuf;

use manyfaced_core::dnn::network::{self, AffineMode, Layer, Loss};
use manyfaced_core::face::recognition::resnet_v1_spec;
use manyfaced_core::face::{FaceRecognitionModel, ShapePredictor};
use manyfaced_core::serialize::Reader;

fn model_path(name: &str) -> PathBuf {
    let dir = std::env::var_os("MANYFACED_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
    let path = dir.join(name);
    assert!(
        path.exists(),
        "{} not found: run `python3 tools/download_models.py`",
        path.display()
    );
    path
}

fn read_model(name: &str) -> Vec<u8> {
    std::fs::read(model_path(name)).unwrap()
}

/// Cut points spread over the whole file, plus the first bytes.
fn cut_points(len: usize) -> impl Iterator<Item = usize> {
    (0..32)
        .chain((1..40).map(move |i| i * len / 40))
        .chain([len - 1])
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ShapePredictor>();
    assert_send_sync::<FaceRecognitionModel>();
};

fn check_shape_predictor(sp: &ShapePredictor, parts: usize, pixels: usize) {
    assert_eq!(sp.num_parts(), parts);
    assert_eq!(sp.cascades().len(), 15);
    for c in sp.cascades() {
        assert_eq!(c.trees().len(), 500);
        assert_eq!(c.num_feature_pixels(), pixels);
        assert!(c.trees().iter().all(|t| t.depth() == 4));
        assert!(c.deltas().iter().flatten().all(|v| v.is_finite()));
    }
    assert_eq!(sp.num_features(), 15 * 500 * 16);
    // The mean shape lives in the normalized [0, 1] box (with some slack for the jaw).
    assert!(
        sp.initial_shape()
            .iter()
            .all(|&v| (-0.5..=1.5).contains(&v))
    );
}

#[test]
fn shape_predictor_5_points() {
    let sp = ShapePredictor::load(model_path("shape_predictor_5_face_landmarks.dat")).unwrap();
    check_shape_predictor(&sp, 5, 800);
}

#[test]
fn shape_predictor_68_points() {
    let sp = ShapePredictor::load(model_path("shape_predictor_68_face_landmarks.dat")).unwrap();
    check_shape_predictor(&sp, 68, 500);
}

#[test]
fn shape_predictor_truncated_files_are_errors() {
    let bytes = read_model("shape_predictor_5_face_landmarks.dat");
    for n in cut_points(bytes.len()) {
        let e = ShapePredictor::from_bytes(&bytes[..n])
            .unwrap_err()
            .to_string();
        assert!(e.contains("invalid dlib shape_predictor"), "{e}");
    }
    let e = ShapePredictor::from_bytes(&bytes[..bytes.len() / 2])
        .unwrap_err()
        .to_string();
    assert!(e.contains("truncated"), "{e}");
}

#[test]
fn face_recognition_resnet() {
    let path = model_path("dlib_face_recognition_resnet_model_v1.dat");
    let model = FaceRecognitionModel::load(&path).unwrap();
    let net = model.network();

    assert_eq!((net.input.nr, net.input.nc), (150, 150));
    assert_eq!(
        (net.input.avg_red, net.input.avg_green, net.input.avg_blue),
        (122.782, 117.001, 104.298)
    );
    assert_eq!(
        net.loss,
        Loss::Metric {
            margin: 0.04,
            dist_thresh: 0.6
        }
    );
    assert_eq!(net.layers.len(), 130);
    assert_eq!(net.num_params(), 5_614_592);
    // dlib's count_parameters() skips affine layers (their get_layer_params() is empty):
    // tools/cpp_probe/inspect_models prints 5607488.
    let affine_params: usize = net
        .layers
        .iter()
        .map(|l| match l {
            Layer::Affine(a) => a.gamma.len() + a.beta.len(),
            _ => 0,
        })
        .sum();
    assert_eq!(net.num_params() - affine_params, 5_607_488);

    let convs: Vec<_> = net
        .layers
        .iter()
        .filter_map(|l| match l {
            Layer::Con(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(convs.len(), 29);
    let stem = convs[0];
    assert_eq!((stem.num_filters, stem.k, stem.nr, stem.nc), (32, 3, 7, 7));
    assert_eq!((stem.stride_y, stem.padding_y), (2, 0));
    // Channels grow 32 -> 64 -> 128 -> 256 -> 256 and each conv reads the previous width.
    let filters: Vec<_> = convs.iter().map(|c| c.num_filters).collect();
    assert_eq!(
        filters,
        [
            [32; 7].as_slice(),
            &[64; 8],
            &[128; 6],
            &[256; 6],
            &[256; 2]
        ]
        .concat()
    );
    for pair in convs.windows(2) {
        assert_eq!(pair[1].k, pair[0].num_filters);
    }
    for c in &convs {
        assert!(c.bias.is_some() && !c.use_relu);
        assert_eq!(c.padding_y, if c.stride_y == 1 { 1 } else { 0 });
    }
    for l in &net.layers {
        if let Layer::Affine(a) = l {
            assert_eq!(a.mode, AffineMode::Conv);
            assert!(!a.disabled && !a.from_batch_norm);
        }
    }
    let fc = match net.layers.last().unwrap() {
        Layer::Fc(fc) => fc,
        l => panic!("last layer is {l:?}"),
    };
    assert_eq!((fc.num_inputs, fc.num_outputs), (256, 128));

    // Checksums printed by tools/cpp_probe/inspect_models (dlib's own deserializer), summed
    // in f64 in storage order (weights, then biases), printed with %.9g.
    let sum = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-8 * b.abs().max(1.0);
    let stem_sum = sum(&stem.weights) + sum(stem.bias.as_ref().unwrap());
    assert!(close(stem_sum, -1.03509322), "{stem_sum}");
    assert!(close(f64::from(stem.weights[0]), -0.0293299202));
    let Layer::Affine(stem_affine) = &net.layers[1] else {
        panic!("layer 1 is {:?}", net.layers[1]);
    };
    assert!(close(f64::from(stem_affine.gamma[0]), 12.1663227));
    assert!(close(f64::from(stem_affine.beta[0]), 0.464286059));
    assert!(
        close(sum(&fc.weights), -0.109791151),
        "{}",
        sum(&fc.weights)
    );

    // The network accounts for every byte of the file.
    let bytes = std::fs::read(&path).unwrap();
    let mut r = Reader::new(&bytes);
    network::parse(&mut r, &resnet_v1_spec()).unwrap();
    assert_eq!(r.remaining(), 0);
}

#[test]
fn face_recognition_truncated_files_are_errors() {
    let bytes = read_model("dlib_face_recognition_resnet_model_v1.dat");
    for n in cut_points(bytes.len()) {
        let e = FaceRecognitionModel::from_bytes(&bytes[..n])
            .unwrap_err()
            .to_string();
        assert!(e.contains("invalid dlib face recognition ResNet"), "{e}");
    }
}

#[test]
fn corrupted_bytes_never_panic() {
    // Flipping bytes may or may not produce an error (a flipped weight is still a valid
    // float), but must never panic or allocate unboundedly.
    let sp = read_model("shape_predictor_5_face_landmarks.dat");
    let net = read_model("dlib_face_recognition_resnet_model_v1.dat");
    for (bytes, parse) in [
        (
            &sp,
            (|b: &[u8]| ShapePredictor::from_bytes(b).is_ok()) as fn(&[u8]) -> bool,
        ),
        (&net, |b: &[u8]| FaceRecognitionModel::from_bytes(b).is_ok()),
    ] {
        let mut copy = bytes.clone();
        for pos in (0..64).chain((1..32).map(|i| i * bytes.len() / 32)) {
            for flip in [0x80, 0xff] {
                copy[pos] ^= flip;
                parse(&copy);
                copy[pos] ^= flip;
            }
        }
    }
}

#[test]
fn wrong_model_type_is_a_descriptive_error() {
    let e = ShapePredictor::load(model_path("dlib_face_recognition_resnet_model_v1.dat"))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("dlib_face_recognition_resnet_model_v1.dat"),
        "{e}"
    );
    assert!(e.contains("invalid dlib shape_predictor"), "{e}");
    let e = FaceRecognitionModel::load(model_path("shape_predictor_5_face_landmarks.dat"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("invalid dlib face recognition ResNet"), "{e}");
    assert!(e.contains("(in add_loss_layer)"), "{e}");
}

#[test]
fn missing_file_is_an_io_error() {
    let e = ShapePredictor::load("/nonexistent/sp.dat").unwrap_err();
    assert!(matches!(e, manyfaced_core::Error::Io { .. }), "{e}");
}
