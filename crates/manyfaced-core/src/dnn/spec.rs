//! Network specifications: Rust values that mirror the C++ template type of a dlib network.
//!
//! dlib's DNN serialization is not self-describing (an `add_layer`, an `add_tag_layer` and an
//! `add_skip_layer` can start with the same bytes), so a file can only be read when the network
//! type is known. A [`NetSpec`] is that type: the layers in execution order, from the one applied
//! to the input up to the one under the loss.

/// One layer of a network spec, with the template parameters that dlib checks on load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSpec {
    /// `con<num_filters, nr, nc, stride_y, stride_x>` with dlib's default padding.
    Con {
        num_filters: usize,
        nr: usize,
        nc: usize,
        stride_y: usize,
        stride_x: usize,
    },
    /// `affine` (also accepts a serialized `bn_con`/`bn_fc`, converted on load like dlib).
    Affine,
    Relu,
    /// `add_prev<tag>`: adds the output of the nearest `tag<tag>` below.
    AddPrev {
        tag: u32,
    },
    /// `max_pool<nr, nc, stride_y, stride_x>` with dlib's default padding.
    MaxPool {
        nr: usize,
        nc: usize,
        stride_y: usize,
        stride_x: usize,
    },
    /// `avg_pool<nr, nc, stride_y, stride_x>`; `avg_pool_everything` is `avg_pool<0,0,1,1>`.
    AvgPool {
        nr: usize,
        nc: usize,
        stride_y: usize,
        stride_x: usize,
    },
    /// `fc_no_bias<num_outputs>`.
    FcNoBias {
        num_outputs: usize,
    },
    /// `tag<id>`: marks its input for a later `add_prev`/`skip`.
    Tag(u32),
    /// `skip<id>`: replaces the current output by the output of the nearest `tag<id>` below.
    Skip(u32),
}

/// The input layer of a network spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputSpec {
    /// `input_rgb_image_sized<nr, nc>`.
    RgbImageSized { nr: usize, nc: usize },
}

/// The loss layer of a network spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossSpec {
    /// `loss_metric`.
    Metric,
}

/// A network type. `layers[0]` is applied to the input, the last layer feeds the loss.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetSpec {
    pub input: InputSpec,
    pub layers: Vec<LayerSpec>,
    pub loss: LossSpec,
}

/// dlib's default padding for `con`, `max_pool` and `avg_pool`: `stride != 1 ? 0 : size / 2`.
pub fn default_padding(size: usize, stride: usize) -> usize {
    if stride != 1 { 0 } else { size / 2 }
}
