//! Image types, pixel conversions, interpolation and image pyramids.
//!
//! Images are 8-bit, row-major, with `C` interleaved channels: `C = 1` is grayscale and `C = 3`
//! is RGB, matching the numpy arrays dlib's Python bindings accept (`H×W` and `H×W×3`).

pub mod interpolation;
pub mod pyramid;

/// A borrowed 8-bit image with `C` channels. Rows may be padded (`row_stride >= width * C`).
#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a, const C: usize> {
    data: &'a [u8],
    width: usize,
    height: usize,
    /// Bytes between the starts of two consecutive rows.
    row_stride: usize,
}

pub type GrayView<'a> = ImageView<'a, 1>;
pub type RgbView<'a> = ImageView<'a, 3>;

/// Why an image buffer cannot be viewed with the requested geometry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "image buffer of {len} bytes is too small for {width}x{height}x{channels} with row stride {row_stride}"
)]
pub struct ImageLayoutError {
    pub len: usize,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub row_stride: usize,
}

impl<'a, const C: usize> ImageView<'a, C> {
    /// Views `data` as `height` rows of `width` pixels, `row_stride` bytes apart.
    pub fn new(
        data: &'a [u8],
        width: usize,
        height: usize,
        row_stride: usize,
    ) -> Result<Self, ImageLayoutError> {
        let err = || ImageLayoutError {
            len: data.len(),
            width,
            height,
            channels: C,
            row_stride,
        };
        let row_bytes = width.checked_mul(C).ok_or_else(err)?;
        let needed = if height == 0 || width == 0 {
            0
        } else {
            row_stride
                .checked_mul(height - 1)
                .and_then(|n| n.checked_add(row_bytes))
                .ok_or_else(err)?
        };
        if row_stride < row_bytes || needed > data.len() {
            return Err(err());
        }
        Ok(ImageView {
            data,
            width,
            height,
            row_stride,
        })
    }

    /// Views a tightly packed buffer.
    pub fn packed(data: &'a [u8], width: usize, height: usize) -> Result<Self, ImageLayoutError> {
        Self::new(data, width, height, width * C)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// The `width * C` bytes of row `y`.
    #[inline]
    pub fn row(&self, y: usize) -> &'a [u8] {
        let start = y * self.row_stride;
        &self.data[start..start + self.width * C]
    }

    /// The pixel at column `x`, row `y`.
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [u8; C] {
        let i = y * self.row_stride + x * C;
        let mut p = [0u8; C];
        p.copy_from_slice(&self.data[i..i + C]);
        p
    }

    /// dlib's `get_pixel_intensity` of the pixel at (`x`, `y`).
    #[inline]
    pub fn intensity(&self, x: usize, y: usize) -> u8 {
        intensity(self.pixel(x, y))
    }
}

/// dlib's `get_pixel_intensity`: the value of a grayscale pixel, and for RGB the unweighted
/// integer mean `(r + g + b) / 3` (`pixel.h`, `assign` from rgb to grayscale).
#[inline]
pub fn intensity<const C: usize>(p: [u8; C]) -> u8 {
    if C == 1 {
        p[0]
    } else {
        let sum: u32 = p.iter().take(3).map(|&v| u32::from(v)).sum();
        (sum / 3) as u8
    }
}

/// An owned, tightly packed 8-bit image with `C` channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image<const C: usize> {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

pub type GrayImage = Image<1>;
pub type RgbImage = Image<3>;

impl<const C: usize> Image<C> {
    /// A black image.
    pub fn zeros(width: usize, height: usize) -> Self {
        Image {
            data: vec![0; width * height * C],
            width,
            height,
        }
    }

    /// A black image, or `None` if its size overflows or the memory is not available.
    pub fn try_zeros(width: usize, height: usize) -> Option<Self> {
        let len = width.checked_mul(height)?.checked_mul(C)?;
        let mut data = Vec::new();
        data.try_reserve_exact(len).ok()?;
        data.resize(len, 0);
        Some(Image {
            data,
            width,
            height,
        })
    }

    pub fn from_vec(data: Vec<u8>, width: usize, height: usize) -> Result<Self, ImageLayoutError> {
        if data.len() != width * height * C {
            return Err(ImageLayoutError {
                len: data.len(),
                width,
                height,
                channels: C,
                row_stride: width * C,
            });
        }
        Ok(Image {
            data,
            width,
            height,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }

    pub fn view(&self) -> ImageView<'_, C> {
        ImageView {
            data: &self.data,
            width: self.width,
            height: self.height,
            row_stride: self.width * C,
        }
    }

    /// The `width * C` bytes of row `y`, mutable.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        let w = self.width * C;
        &mut self.data[y * w..(y + 1) * w]
    }

    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, p: [u8; C]) {
        let i = (y * self.width + x) * C;
        self.data[i..i + C].copy_from_slice(&p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_intensity_is_the_truncated_mean() {
        assert_eq!(intensity([255u8, 255, 254]), 254);
        assert_eq!(intensity([1u8, 1, 0]), 0);
        assert_eq!(intensity([10u8, 20, 31]), 20);
        assert_eq!(intensity([77u8]), 77);
    }

    #[test]
    fn views_with_padded_rows() {
        let data: Vec<u8> = (0..20).collect();
        // 2x3 RGB with a 10-byte stride: rows start at 0 and 10.
        let v = RgbView::new(&data, 2, 2, 10).unwrap();
        assert_eq!(v.pixel(1, 1), [13, 14, 15]);
        assert_eq!(v.row(1), [10, 11, 12, 13, 14, 15]);
        assert!(RgbView::new(&data, 4, 2, 10).is_err());
        assert!(RgbView::new(&data, 3, 3, 9).is_err());
        assert!(GrayView::packed(&[], 0, 0).unwrap().is_empty());
    }
}
