//! Loaded dlib networks and the spec-driven parser for dlib's DNN serialization.
//!
//! Ported from `dlib/dnn/core.h`, `layers.h`, `input.h`, `loss.h` and `cuda/tensor.h`
//! (dlib v20.0.1, Boost Software License 1.0, see `THIRD_PARTY_NOTICES`).
//!
//! The stream nests from the loss down to the input, and every `add_layer` writes its subnet
//! *before* its own details. [`parse`] therefore reads the layer headers top-down, then the input
//! layer, then the layer details bottom-up, without recursion.

use super::spec::{InputSpec, LayerSpec, LossSpec, NetSpec, default_padding};
use crate::serialize::{Reader, Result, TensorShape, tensor_len};

/// `affine_` mode: per channel (`CONV_MODE`) or per element (`FC_MODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineMode {
    Conv,
    Fc,
}

/// A `con_` layer. `weights` has shape `(num_filters, k, nr, nc)`, row-major.
#[derive(Debug, Clone)]
pub struct Conv {
    pub num_filters: usize,
    pub k: usize,
    pub nr: usize,
    pub nc: usize,
    pub stride_y: usize,
    pub stride_x: usize,
    pub padding_y: usize,
    pub padding_x: usize,
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    /// `con_6`'s fused relu.
    pub use_relu: bool,
}

/// An `affine_` layer: `out = gamma * in + beta`. In [`AffineMode::Conv`] `gamma`/`beta` have
/// one value per channel.
#[derive(Debug, Clone)]
pub struct Affine {
    pub mode: AffineMode,
    /// Shape of `gamma` and `beta`.
    pub shape: TensorShape,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    /// Whether the values come from a serialized `bn_` converted at load time.
    pub from_batch_norm: bool,
    pub disabled: bool,
}

/// A `max_pool_` or `avg_pool_` layer. A size of 0 means "the whole input" (`avg_pool_everything`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool {
    pub nr: usize,
    pub nc: usize,
    pub stride_y: usize,
    pub stride_x: usize,
    pub padding_y: usize,
    pub padding_x: usize,
}

/// An `fc_` layer. `weights` has shape `(num_inputs, num_outputs)`, row-major.
#[derive(Debug, Clone)]
pub struct Fc {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub enum Layer {
    Con(Conv),
    Affine(Affine),
    Relu { disabled: bool },
    AddPrev { tag: u32 },
    MaxPool(Pool),
    AvgPool(Pool),
    Fc(Fc),
    Tag(u32),
    Skip(u32),
}

/// `input_rgb_image_sized`: the expected image size and the per-channel means subtracted from
/// the input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbInput {
    pub nr: usize,
    pub nc: usize,
    pub avg_red: f32,
    pub avg_green: f32,
    pub avg_blue: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Loss {
    Metric { margin: f32, dist_thresh: f32 },
}

/// A loaded network. `layers[0]` is applied to the input.
#[derive(Debug, Clone)]
pub struct Network {
    pub input: RgbInput,
    pub layers: Vec<Layer>,
    pub loss: Loss,
}

impl Network {
    /// Number of learned scalars (weights, biases, affine gammas and betas). Unlike dlib's
    /// `count_parameters()`, this includes the affine layers.
    pub fn num_params(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                Layer::Con(c) => c.weights.len() + c.bias.as_ref().map_or(0, Vec::len),
                Layer::Affine(a) => a.gamma.len() + a.beta.len(),
                Layer::Fc(f) => f.weights.len() + f.bias.as_ref().map_or(0, Vec::len),
                _ => 0,
            })
            .sum()
    }
}

/// Parses a serialized network of type `spec`.
pub fn parse(r: &mut Reader, spec: &NetSpec) -> Result<Network> {
    if spec.layers.is_empty() {
        return Err(r.invalid("network spec has no layers"));
    }
    let loss = r.context(
        || "add_loss_layer",
        |r| {
            r.int_version("dlib::add_loss_layer", 1..=1)?;
            read_loss(r, spec.loss)
        },
    )?;

    // Headers, outermost (last executed) layer first.
    let n = spec.layers.len();
    let mut versions = vec![0; n];
    for i in (0..n).rev() {
        let ls = &spec.layers[i];
        versions[i] = r.context(
            || layer_name(i, ls),
            |r| {
                let over_input = i == 0;
                match (ls, over_input) {
                    (LayerSpec::Tag(_), false) => r.int_version("dlib::add_tag_layer", 1..=1),
                    (LayerSpec::Tag(_), true) => r.int_version("dlib::add_tag_layer", 1..=2),
                    (LayerSpec::Skip(_), false) => r.int_version("dlib::add_skip_layer", 1..=1),
                    (LayerSpec::Skip(_), true) => {
                        Err(r.invalid("a skip layer cannot sit on the input"))
                    }
                    (_, false) => r.int_version("dlib::add_layer", 1..=2),
                    (_, true) => r.int_version("dlib::add_layer", 2..=3),
                }
            },
        )?;
    }

    let input = r.context(|| "input", |r| read_input(r, spec.input))?;

    // Details and trailers, innermost layer first.
    let mut layers = Vec::with_capacity(n);
    for (i, ls) in spec.layers.iter().enumerate() {
        let layer = r.context(
            || layer_name(i, ls),
            |r| {
                let over_input = i == 0;
                let version = versions[i];
                match *ls {
                    LayerSpec::Tag(id) => {
                        if over_input {
                            // add_tag_layer<ID, input>: cached_output, grad_final,
                            // gradient_input_is_stale, [v2: sample_expansion_factor].
                            r.skip_resizable_tensor()?;
                            r.skip_resizable_tensor()?;
                            r.bool()?;
                            if version >= 2 {
                                r.u32()?;
                            }
                        }
                        Ok(Layer::Tag(id))
                    }
                    LayerSpec::Skip(id) => Ok(Layer::Skip(id)),
                    _ => {
                        let layer = read_details(r, ls)?;
                        // this_layer_setup_called, gradient_input_is_stale,
                        // get_output_and_gradient_input_disabled, x_grad, cached_output.
                        for _ in 0..3 {
                            r.bool()?;
                        }
                        r.skip_resizable_tensor()?;
                        r.skip_resizable_tensor()?;
                        if over_input {
                            r.skip_resizable_tensor()?; // grad_final
                            if version >= 3 {
                                r.u32()?; // _sample_expansion_factor
                            }
                        } else if version >= 2 {
                            r.skip_resizable_tensor()?; // params_grad
                        }
                        Ok(layer)
                    }
                }
            },
        )?;
        layers.push(layer);
    }
    check_tags(r, &layers)?;
    Ok(Network {
        input,
        layers,
        loss,
    })
}

fn layer_name(i: usize, spec: &LayerSpec) -> String {
    format!("layer {i} ({spec:?})")
}

/// Every `add_prev`/`skip` must refer to a tag below it, so inference can always resolve it.
fn check_tags(r: &Reader, layers: &[Layer]) -> Result<()> {
    let mut seen = Vec::new();
    for (i, l) in layers.iter().enumerate() {
        match *l {
            Layer::Tag(id) => seen.push(id),
            Layer::AddPrev { tag: id } | Layer::Skip(id) if !seen.contains(&id) => {
                return Err(r.invalid(format!(
                    "layer {i} refers to tag {id}, which is not below it"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

fn to_usize(r: &Reader, v: i64, what: &str) -> Result<usize> {
    usize::try_from(v).map_err(|_| r.invalid(format!("negative {what} {v}")))
}

fn expect_eq(r: &Reader, what: &str, found: usize, expected: usize) -> Result<()> {
    if found != expected {
        return Err(r.invalid(format!(
            "{what} is {found}, the network type expects {expected}"
        )));
    }
    Ok(())
}

fn expect_shape(r: &Reader, what: &str, found: TensorShape, expected: TensorShape) -> Result<()> {
    if found != expected {
        return Err(r.invalid(format!("{what} has shape {found:?}, expected {expected:?}")));
    }
    Ok(())
}

/// `params[offset..offset + len]`, checked.
fn slice_params(
    r: &Reader,
    params: &[f32],
    offset: usize,
    len: usize,
    what: &str,
) -> Result<Vec<f32>> {
    let end = offset.saturating_add(len);
    params.get(offset..end).map(<[f32]>::to_vec).ok_or_else(|| {
        r.invalid(format!(
            "{what} needs params[{offset}..{end}] but the layer has {} params",
            params.len()
        ))
    })
}

fn read_details(r: &mut Reader, spec: &LayerSpec) -> Result<Layer> {
    match *spec {
        LayerSpec::Con {
            num_filters,
            nr,
            nc,
            stride_y,
            stride_x,
        } => read_con(r, num_filters, nr, nc, stride_y, stride_x).map(Layer::Con),
        LayerSpec::Affine => read_affine(r).map(Layer::Affine),
        LayerSpec::Relu => {
            let v = r.string_version("dlib::relu_", &["relu_", "relu_2"])?;
            let disabled = if v == "relu_2" { r.bool()? } else { false };
            Ok(Layer::Relu { disabled })
        }
        LayerSpec::AddPrev { tag } => {
            r.string_version("dlib::add_prev_", &["add_prev_"])?;
            Ok(Layer::AddPrev { tag })
        }
        LayerSpec::MaxPool {
            nr,
            nc,
            stride_y,
            stride_x,
        } => {
            r.string_version("dlib::max_pool_", &["max_pool_2"])?;
            read_pool(r, nr, nc, stride_y, stride_x).map(Layer::MaxPool)
        }
        LayerSpec::AvgPool {
            nr,
            nc,
            stride_y,
            stride_x,
        } => {
            r.string_version("dlib::avg_pool_", &["avg_pool_2"])?;
            read_pool(r, nr, nc, stride_y, stride_x).map(Layer::AvgPool)
        }
        LayerSpec::FcNoBias { num_outputs } => read_fc_no_bias(r, num_outputs).map(Layer::Fc),
        LayerSpec::Tag(_) | LayerSpec::Skip(_) => {
            Err(r.invalid("internal error: tags and skips have no details"))
        }
    }
}

fn read_con(
    r: &mut Reader,
    num_filters: usize,
    nr: usize,
    nc: usize,
    stride_y: usize,
    stride_x: usize,
) -> Result<Conv> {
    let v = r.string_version("dlib::con_", &["con_4", "con_5", "con_6"])?;
    let (_, params) = r.context(|| "params", Reader::resizable_tensor)?;
    let f_num_filters = r.i64()?;
    let f_nr = r.i64()?;
    let f_nc = r.i64()?;
    let f_stride_y = i64::from(r.i32()?);
    let f_stride_x = i64::from(r.i32()?);
    let f_padding_y = i64::from(r.i32()?);
    let f_padding_x = i64::from(r.i32()?);
    let filters = r.context(|| "filters", Reader::alias_tensor)?;
    let biases = r.context(|| "biases", Reader::alias_tensor)?;
    for _ in 0..4 {
        r.f64()?; // learning rate and weight decay multipliers
    }
    let use_bias = if v != "con_4" { r.bool()? } else { true };
    let use_relu = if v == "con_6" { r.bool()? } else { false };

    let (padding_y, padding_x) = (default_padding(nr, stride_y), default_padding(nc, stride_x));
    for (what, found, expected) in [
        ("num_filters", f_num_filters, num_filters),
        ("nr", f_nr, nr),
        ("nc", f_nc, nc),
        ("stride_y", f_stride_y, stride_y),
        ("stride_x", f_stride_x, stride_x),
        ("padding_y", f_padding_y, padding_y),
        ("padding_x", f_padding_x, padding_x),
    ] {
        let found = to_usize(r, found, what)?;
        expect_eq(r, what, found, expected)?;
    }
    let k = filters[1];
    expect_shape(r, "filters", filters, [num_filters, k, nr, nc])?;
    let weights = slice_params(r, &params, 0, tensor_len(filters), "filters")?;
    let bias = if use_bias {
        expect_shape(r, "biases", biases, [1, num_filters, 1, 1])?;
        Some(slice_params(
            r,
            &params,
            weights.len(),
            num_filters,
            "biases",
        )?)
    } else {
        None
    };
    Ok(Conv {
        num_filters,
        k,
        nr,
        nc,
        stride_y,
        stride_x,
        padding_y,
        padding_x,
        weights,
        bias,
        use_relu,
    })
}

fn affine_mode(r: &Reader, mode: i32) -> Result<AffineMode> {
    match mode {
        0 => Ok(AffineMode::Conv),
        1 => Ok(AffineMode::Fc),
        m => Err(r.invalid(format!("invalid affine layer mode {m}"))),
    }
}

fn read_affine(r: &mut Reader) -> Result<Affine> {
    let v = r.string_version(
        "dlib::affine_",
        &["affine_", "affine_2", "bn_con2", "bn_fc2"],
    )?;
    if v.starts_with("bn_") {
        return read_bn_as_affine(
            r,
            if v == "bn_con2" {
                AffineMode::Conv
            } else {
                AffineMode::Fc
            },
        );
    }
    let (_, params) = r.context(|| "params", Reader::resizable_tensor)?;
    let gamma = r.context(|| "gamma", Reader::alias_tensor)?;
    let beta = r.context(|| "beta", Reader::alias_tensor)?;
    let mode = r.i32()?;
    let mode = affine_mode(r, mode)?;
    let disabled = if v == "affine_2" { r.bool()? } else { false };
    expect_shape(r, "beta", beta, gamma)?;
    let gamma_v = slice_params(r, &params, 0, tensor_len(gamma), "gamma")?;
    let beta_v = slice_params(r, &params, gamma_v.len(), tensor_len(beta), "beta")?;
    Ok(Affine {
        mode,
        shape: gamma,
        gamma: gamma_v,
        beta: beta_v,
        from_batch_norm: false,
        disabled,
    })
}

/// `affine_(const bn_<mode>&)`: `g = gamma / sqrt(running_var + eps)`,
/// `b = beta - g * running_mean`, in `float` like dlib (`eps` is cast to `float` by
/// `op_add_scalar`).
fn read_bn_as_affine(r: &mut Reader, mode: AffineMode) -> Result<Affine> {
    let (_, params) = r.context(|| "params", Reader::resizable_tensor)?;
    let gamma = r.context(|| "gamma", Reader::alias_tensor)?;
    let beta = r.context(|| "beta", Reader::alias_tensor)?;
    r.skip_resizable_tensor()?; // means
    r.skip_resizable_tensor()?; // invstds
    let (_, running_means) = r.context(|| "running_means", Reader::resizable_tensor)?;
    let (_, running_variances) = r.context(|| "running_variances", Reader::resizable_tensor)?;
    r.u64()?; // num_updates
    r.u64()?; // running_stats_window_size
    for _ in 0..4 {
        r.f64()?; // learning rate and weight decay multipliers
    }
    let eps = r.f64()? as f32;

    expect_shape(r, "beta", beta, gamma)?;
    let n = tensor_len(gamma);
    let sg = slice_params(r, &params, 0, n, "gamma")?;
    let sb = slice_params(r, &params, n, n, "beta")?;
    if running_means.len() != n || running_variances.len() != n {
        return Err(r.invalid(format!(
            "batch norm has {n} channels but {} running means and {} running variances",
            running_means.len(),
            running_variances.len()
        )));
    }
    let g: Vec<f32> = sg
        .iter()
        .zip(&running_variances)
        .map(|(&s, &var)| s / (var + eps).sqrt())
        .collect();
    let b: Vec<f32> = sb
        .iter()
        .zip(&g)
        .zip(&running_means)
        .map(|((&s, &g), &m)| s - g * m)
        .collect();
    Ok(Affine {
        mode,
        shape: gamma,
        gamma: g,
        beta: b,
        from_batch_norm: true,
        disabled: false,
    })
}

fn read_pool(
    r: &mut Reader,
    nr: usize,
    nc: usize,
    stride_y: usize,
    stride_x: usize,
) -> Result<Pool> {
    let f_nr = r.i64()?;
    let f_nc = r.i64()?;
    let f_stride_y = i64::from(r.i32()?);
    let f_stride_x = i64::from(r.i32()?);
    let f_padding_y = i64::from(r.i32()?);
    let f_padding_x = i64::from(r.i32()?);
    let (padding_y, padding_x) = (default_padding(nr, stride_y), default_padding(nc, stride_x));
    for (what, found, expected) in [
        ("nr", f_nr, nr),
        ("nc", f_nc, nc),
        ("stride_y", f_stride_y, stride_y),
        ("stride_x", f_stride_x, stride_x),
        ("padding_y", f_padding_y, padding_y),
        ("padding_x", f_padding_x, padding_x),
    ] {
        let found = to_usize(r, found, what)?;
        expect_eq(r, what, found, expected)?;
    }
    Ok(Pool {
        nr,
        nc,
        stride_y,
        stride_x,
        padding_y,
        padding_x,
    })
}

fn read_fc_no_bias(r: &mut Reader, num_outputs: usize) -> Result<Fc> {
    let v = r.string_version("dlib::fc_", &["fc_2", "fc_3"])?;
    let f_num_outputs = r.u64()?;
    let num_inputs = r.u64()?;
    let (_, params) = r.context(|| "params", Reader::resizable_tensor)?;
    let weights = r.context(|| "weights", Reader::alias_tensor)?;
    r.context(|| "biases", Reader::alias_tensor)?;
    let bias_mode = r.i32()?;
    for _ in 0..4 {
        r.f64()?; // learning rate and weight decay multipliers
    }
    if v == "fc_3" {
        r.bool()?; // use_bias, irrelevant in FC_NO_BIAS mode
    }
    // FC_NO_BIAS = 1 (layers_abstract.h).
    if bias_mode != 1 {
        return Err(r.invalid(format!("fc bias mode is {bias_mode}, fc_no_bias expects 1")));
    }
    let to = |v: u64| usize::try_from(v).map_err(|_| r.invalid(format!("fc size {v} too large")));
    expect_eq(r, "num_outputs", to(f_num_outputs)?, num_outputs)?;
    let num_inputs = to(num_inputs)?;
    expect_shape(r, "weights", weights, [num_inputs, num_outputs, 1, 1])?;
    let weights = slice_params(
        r,
        &params,
        0,
        num_inputs.saturating_mul(num_outputs),
        "weights",
    )?;
    Ok(Fc {
        num_inputs,
        num_outputs,
        weights,
        bias: None,
    })
}

fn read_input(r: &mut Reader, spec: InputSpec) -> Result<RgbInput> {
    let InputSpec::RgbImageSized { nr, nc } = spec;
    r.string_version("dlib::input_rgb_image_sized", &["input_rgb_image_sized"])?;
    let avg_red = r.f32()?;
    let avg_green = r.f32()?;
    let avg_blue = r.f32()?;
    let to = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
    let f_nr = to(r.u64()?);
    let f_nc = to(r.u64()?);
    expect_eq(r, "input rows", f_nr, nr)?;
    expect_eq(r, "input columns", f_nc, nc)?;
    Ok(RgbInput {
        nr,
        nc,
        avg_red,
        avg_green,
        avg_blue,
    })
}

fn read_loss(r: &mut Reader, spec: LossSpec) -> Result<Loss> {
    let LossSpec::Metric = spec;
    let v = r.string_version("dlib::loss_metric_", &["loss_metric_", "loss_metric_2"])?;
    if v == "loss_metric_" {
        // Hard-coded values of the first version (loss.h).
        return Ok(Loss::Metric {
            margin: 0.1,
            dist_thresh: 0.75,
        });
    }
    Ok(Loss::Metric {
        margin: r.f32()?,
        dist_thresh: r.f32()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::writer::Writer;

    /// `loss_metric<affine<input_rgb_image_sized<2>>>` with the affine slot holding a
    /// serialized `bn_con2`.
    fn bn_net_bytes() -> Vec<u8> {
        let mut w = Writer::default();
        w.i64(1); // add_loss_layer
        w.string("loss_metric_2");
        w.f32(0.04);
        w.f32(0.6);
        w.i64(3); // add_layer over the input
        w.string("input_rgb_image_sized");
        for v in [1.0, 2.0, 3.0] {
            w.f32(v);
        }
        w.u64(2);
        w.u64(2);
        w.string("bn_con2");
        w.resizable_tensor([1, 4, 1, 1], &[2.0, 3.0, 0.5, -1.0]); // gamma, beta
        w.alias_tensor([1, 2, 1, 1]);
        w.alias_tensor([1, 2, 1, 1]);
        w.resizable_tensor([1, 2, 1, 1], &[9.0, 9.0]); // means (unused)
        w.resizable_tensor([1, 2, 1, 1], &[9.0, 9.0]); // invstds (unused)
        w.resizable_tensor([1, 2, 1, 1], &[1.0, 2.0]); // running_means
        w.resizable_tensor([1, 2, 1, 1], &[3.0, 8.0]); // running_variances
        w.u64(0);
        w.u64(100);
        for v in [1.0, 1.0, 1.0, 0.0, 1.0] {
            w.f64(v); // multipliers, then eps = 1
        }
        for _ in 0..3 {
            w.bool(false);
        }
        for _ in 0..3 {
            w.resizable_tensor([0, 0, 0, 0], &[]);
        }
        w.u64(1); // _sample_expansion_factor
        w.buf
    }

    fn bn_spec() -> NetSpec {
        NetSpec {
            input: InputSpec::RgbImageSized { nr: 2, nc: 2 },
            layers: vec![LayerSpec::Affine],
            loss: LossSpec::Metric,
        }
    }

    #[test]
    fn batch_norm_is_converted_to_affine() {
        let bytes = bn_net_bytes();
        let mut r = Reader::new(&bytes);
        let net = parse(&mut r, &bn_spec()).unwrap();
        assert_eq!(r.remaining(), 0);
        let Layer::Affine(a) = &net.layers[0] else {
            panic!("{:?}", net.layers[0]);
        };
        assert_eq!(a.mode, AffineMode::Conv);
        assert!(a.from_batch_norm);
        // g = gamma / sqrt(var + eps), b = beta - g * mean
        assert_eq!(a.gamma, [1.0, 1.0]);
        assert_eq!(a.beta, [-0.5, -3.0]);
        assert_eq!(net.input.avg_blue, 3.0);
    }

    #[test]
    fn every_truncation_is_an_error() {
        let bytes = bn_net_bytes();
        for n in 0..bytes.len() {
            assert!(
                parse(&mut Reader::new(&bytes[..n]), &bn_spec()).is_err(),
                "{n}"
            );
        }
    }

    #[test]
    fn spec_mismatch_is_reported() {
        let bytes = bn_net_bytes();
        let spec = NetSpec {
            input: InputSpec::RgbImageSized { nr: 150, nc: 150 },
            ..bn_spec()
        };
        let e = parse(&mut Reader::new(&bytes), &spec)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("input rows is 2, the network type expects 150"),
            "{e}"
        );
    }

    #[test]
    fn dangling_tag_reference_is_rejected() {
        let r = Reader::new(&[]);
        let layers = [Layer::AddPrev { tag: 1 }];
        let e = check_tags(&r, &layers).unwrap_err().to_string();
        assert!(e.contains("refers to tag 1"), "{e}");
    }
}
