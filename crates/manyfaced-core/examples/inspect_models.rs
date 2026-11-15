//! Prints the structure of dlib model files: `cargo run --release --example inspect_models -- FILE...`
//! Files whose name contains `shape_predictor` are read as shape predictors, others as the
//! face recognition ResNet.
use manyfaced_core::dnn::Layer;
use manyfaced_core::face::{FaceRecognitionModel, ShapePredictor};

fn main() {
    for path in std::env::args().skip(1) {
        let t = std::time::Instant::now();
        let is_sp = path.contains("shape_predictor");
        match ShapePredictor::load(&path) {
            Ok(sp) => {
                let c = sp.cascades();
                let depths: std::collections::BTreeSet<u32> = c
                    .iter()
                    .flat_map(|c| c.trees())
                    .map(|t| t.depth())
                    .collect();
                println!(
                    "{path}: {} parts, {} cascades, trees per cascade {:?}, depths {:?}, pixels {:?}, features {} ({:?})",
                    sp.num_parts(),
                    c.len(),
                    c.iter()
                        .map(|c| c.trees().len())
                        .collect::<std::collections::BTreeSet<_>>(),
                    depths,
                    c.iter()
                        .map(|c| c.num_feature_pixels())
                        .collect::<std::collections::BTreeSet<_>>(),
                    sp.num_features(),
                    t.elapsed()
                );
            }
            Err(e) if is_sp => println!("{e}"),
            Err(_) => match FaceRecognitionModel::load(&path) {
                Ok(m) => {
                    let net = m.network();
                    let kinds: std::collections::BTreeMap<&str, usize> =
                        net.layers.iter().fold(Default::default(), |mut acc, l| {
                            let k = match l {
                                Layer::Con(_) => "con",
                                Layer::Affine(a) if a.from_batch_norm => "affine(bn)",
                                Layer::Affine(_) => "affine",
                                Layer::Relu { .. } => "relu",
                                Layer::AddPrev { .. } => "add_prev",
                                Layer::MaxPool(_) => "max_pool",
                                Layer::AvgPool(_) => "avg_pool",
                                Layer::Fc(_) => "fc",
                                Layer::Tag(_) => "tag",
                                Layer::Skip(_) => "skip",
                            };
                            *acc.entry(k).or_default() += 1;
                            acc
                        });
                    println!(
                        "{path}: {} layers {kinds:?}, {} params, input {:?}, loss {:?} ({:?})",
                        net.layers.len(),
                        net.num_params(),
                        net.input,
                        net.loss,
                        t.elapsed()
                    );
                }
                Err(e) => println!("{e}"),
            },
        }
    }
}
