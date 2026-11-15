//! Reader for dlib's binary serialization format.
//!
//! Ported from `dlib/serialize.h` and `dlib/float_details.h` (dlib v20.0.1, Boost Software
//! License 1.0, see `THIRD_PARTY_NOTICES`).
//!
//! dlib's format is not self-describing: what is read depends on the C++ type being
//! deserialized. [`Reader`] therefore exposes one method per C++ type (`int`, `long`,
//! `float`, ...) because the type decides how many bytes an integer may use and how a float is
//! rounded.

mod float;
#[cfg(test)]
pub(crate) mod writer;

use std::fmt;

/// What went wrong while decoding a dlib stream.
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    /// The stream ended while `needed` more bytes were expected.
    UnexpectedEof { needed: usize, available: usize },
    /// An integer control byte that is invalid for the C++ type being read.
    InvalidIntegerControl { ty: &'static str, control: u8 },
    /// A well-formed integer whose value does not fit the C++ type being read.
    IntegerOutOfRange { ty: &'static str },
    /// A `bool` that is neither ASCII `'1'` nor `'0'`.
    InvalidBool(u8),
    /// A float stored in dlib's pre-2013 ASCII format, which manyfaced does not read.
    LegacyFloatFormat,
    /// A version tag that this reader does not know.
    UnexpectedVersion { what: &'static str, found: String },
    /// Structurally valid data that violates an invariant of the object being loaded.
    Invalid(String),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UnexpectedEof { needed, available } => write!(
                f,
                "unexpected end of data (needed {needed} more bytes, {available} available): \
                 the file is truncated or not a dlib file"
            ),
            ErrorKind::InvalidIntegerControl { ty, control } => {
                write!(f, "invalid integer control byte 0x{control:02x} for {ty}")
            }
            ErrorKind::IntegerOutOfRange { ty } => write!(f, "integer does not fit in {ty}"),
            ErrorKind::InvalidBool(b) => write!(f, "invalid bool byte 0x{b:02x}"),
            ErrorKind::LegacyFloatFormat => {
                write!(
                    f,
                    "float stored in dlib's legacy ASCII format (unsupported)"
                )
            }
            ErrorKind::UnexpectedVersion { what, found } => {
                write!(f, "unexpected version {found:?} while reading {what}")
            }
            ErrorKind::Invalid(msg) => f.write_str(msg),
        }
    }
}

/// A decoding error with the byte offset where it happened and the path of the object
/// being read, outermost first (e.g. `shape_predictor > forests[3][17] > leaf_values`).
#[derive(Debug, Clone, PartialEq)]
pub struct Error {
    pub kind: ErrorKind,
    pub offset: usize,
    /// Innermost context last pushed first; use [`Error::context_path`] for display order.
    context: Vec<String>,
}

impl Error {
    pub fn new(kind: ErrorKind, offset: usize) -> Self {
        Error {
            kind,
            offset,
            context: Vec::new(),
        }
    }

    /// The context path, outermost first.
    pub fn context_path(&self) -> impl Iterator<Item = &str> {
        self.context.iter().rev().map(String::as_str)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte offset {}", self.kind, self.offset)?;
        if !self.context.is_empty() {
            f.write_str(" (in ")?;
            for (i, c) in self.context_path().enumerate() {
                if i > 0 {
                    f.write_str(" > ")?;
                }
                f.write_str(c)?;
            }
            f.write_str(")")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// A dense row-major matrix as stored by `dlib::matrix`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

/// Shape of a dlib tensor: (num_samples, k, nr, nc).
pub type TensorShape = [usize; 4];

/// Number of elements of a tensor shape. Saturates at `usize::MAX`, so that a hostile shape
/// gives a length that no parameter buffer has (an error) instead of a wrapped one.
pub fn tensor_len(shape: TensorShape) -> usize {
    shape.iter().fold(1usize, |n, &d| n.saturating_mul(d))
}

/// A cursor over a dlib serialization stream.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// An error of `kind` at the current position.
    pub fn error(&self, kind: ErrorKind) -> Error {
        Error::new(kind, self.pos)
    }

    /// An [`ErrorKind::Invalid`] error at the current position.
    pub fn invalid(&self, msg: impl Into<String>) -> Error {
        self.error(ErrorKind::Invalid(msg.into()))
    }

    /// Runs `f`, adding `name()` to the context path of any error it returns. The name is only
    /// built on error, so this is free on the success path.
    pub fn context<T, N: fmt::Display>(
        &mut self,
        name: impl FnOnce() -> N,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        f(self).map_err(|mut e| {
            e.context.push(name().to_string());
            e
        })
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(self.error(ErrorKind::UnexpectedEof {
                needed: n,
                available: self.remaining(),
            }));
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn peek(&self) -> Result<u8> {
        self.data.get(self.pos).copied().ok_or_else(|| {
            self.error(ErrorKind::UnexpectedEof {
                needed: 1,
                available: 0,
            })
        })
    }

    fn le_magnitude(bytes: &[u8]) -> u64 {
        bytes
            .iter()
            .rev()
            .fold(0u64, |acc, &b| (acc << 8) | u64::from(b))
    }

    /// `ser_helper::unpack_int` for an unsigned type of `width` bytes.
    fn unsigned(&mut self, width: usize, ty: &'static str) -> Result<u64> {
        let start = self.pos;
        let control = self.byte()?;
        // dlib masks out the three reserved bits but keeps the sign bit, so a "negative"
        // unsigned value yields size > 8 and is rejected.
        let size = usize::from(control & 0x8F);
        if size == 0 || size > width {
            return Err(Error::new(
                ErrorKind::InvalidIntegerControl { ty, control },
                start,
            ));
        }
        let value = Self::le_magnitude(self.bytes(size)?);
        if width < 8 && value >> (8 * width) != 0 {
            return Err(Error::new(ErrorKind::IntegerOutOfRange { ty }, start));
        }
        Ok(value)
    }

    /// `ser_helper::unpack_int` for a signed type of `width` bytes.
    fn signed(&mut self, width: usize, ty: &'static str) -> Result<i64> {
        let start = self.pos;
        let control = self.byte()?;
        let negative = control & 0x80 != 0;
        let size = usize::from(control & 0x0F);
        if size == 0 || size > width {
            return Err(Error::new(
                ErrorKind::InvalidIntegerControl { ty, control },
                start,
            ));
        }
        let magnitude = Self::le_magnitude(self.bytes(size)?);
        let max = (1u64 << (8 * width - 1)) - 1;
        if magnitude > max {
            return Err(Error::new(ErrorKind::IntegerOutOfRange { ty }, start));
        }
        // magnitude <= i64::MAX here, so the cast is lossless.
        let value = magnitude as i64;
        Ok(if negative { -value } else { value })
    }

    /// C++ `short` / `int16`.
    pub fn i16(&mut self) -> Result<i16> {
        // `signed` checked the value fits in 2 bytes.
        Ok(self.signed(2, "int16")? as i16)
    }

    /// C++ `int`.
    pub fn i32(&mut self) -> Result<i32> {
        Ok(self.signed(4, "int")? as i32)
    }

    /// C++ `long` (LP64) / `long long` / `int64`.
    pub fn i64(&mut self) -> Result<i64> {
        self.signed(8, "int64")
    }

    /// C++ `unsigned int`.
    pub fn u32(&mut self) -> Result<u32> {
        Ok(self.unsigned(4, "unsigned int")? as u32)
    }

    /// C++ `unsigned long` (LP64) / `size_t` / `uint64`.
    pub fn u64(&mut self) -> Result<u64> {
        self.unsigned(8, "uint64")
    }

    /// C++ `bool`: ASCII `'1'` or `'0'`.
    pub fn bool(&mut self) -> Result<bool> {
        let start = self.pos;
        match self.byte()? {
            b'1' => Ok(true),
            b'0' => Ok(false),
            b => Err(Error::new(ErrorKind::InvalidBool(b), start)),
        }
    }

    fn float_details(&mut self) -> Result<(i64, i16)> {
        // dlib detects the legacy ASCII format because a binary integer control byte never has
        // any of the bits 0x70 set.
        if self.peek()? & 0x70 != 0 {
            return Err(self.error(ErrorKind::LegacyFloatFormat));
        }
        let mantissa = self.i64()?;
        let exponent = self.i16()?;
        Ok((mantissa, exponent))
    }

    /// C++ `float`.
    pub fn f32(&mut self) -> Result<f32> {
        let (m, e) = self.float_details()?;
        Ok(float::to_f32(m, e))
    }

    /// C++ `double`.
    pub fn f64(&mut self) -> Result<f64> {
        let (m, e) = self.float_details()?;
        Ok(float::to_f64(m, e))
    }

    /// A container length (`unsigned long`), checked against the remaining bytes assuming each
    /// element takes at least `min_elem_bytes` bytes, so corrupt lengths cannot cause huge
    /// allocations.
    pub fn len(&mut self, min_elem_bytes: usize) -> Result<usize> {
        let start = self.pos;
        let n = self.u64()?;
        let fits = usize::try_from(n)
            .ok()
            .filter(|&n| n.saturating_mul(min_elem_bytes.max(1)) <= self.remaining());
        fits.ok_or_else(|| {
            Error::new(
                ErrorKind::Invalid(format!(
                    "container length {n} exceeds the {} remaining bytes",
                    self.remaining()
                )),
                start,
            )
        })
    }

    /// `std::string`.
    pub fn string(&mut self) -> Result<String> {
        let n = self.len(1)?;
        let bytes = self.bytes(n)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// `std::vector<T>` where each element is read by `f(reader, index)`.
    pub fn vec<T>(
        &mut self,
        min_elem_bytes: usize,
        mut f: impl FnMut(&mut Self, usize) -> Result<T>,
    ) -> Result<Vec<T>> {
        let n = self.len(min_elem_bytes)?;
        // `n` is limited by the remaining bytes, but an element in memory can be many times
        // larger than its minimum size in the file. Limit the first reservation.
        let mut out = Vec::with_capacity(n.min(1 << 16));
        for i in 0..n {
            out.push(self.context(|| format!("[{i}]"), |r| f(r, i))?);
        }
        Ok(out)
    }

    fn matrix<T>(&mut self, mut elem: impl FnMut(&mut Self) -> Result<T>) -> Result<Matrix<T>> {
        let start = self.pos;
        let mut nr = self.i64()?;
        let mut nc = self.i64()?;
        // Negative dimensions mark the current format; positive ones the legacy format. Both
        // are negated together, exactly like dlib.
        if nr < 0 || nc < 0 {
            // i64::MIN stays negative and is rejected below.
            nr = nr.wrapping_neg();
            nc = nc.wrapping_neg();
        }
        let (rows, cols) = match (usize::try_from(nr), usize::try_from(nc)) {
            (Ok(r), Ok(c)) => (r, c),
            _ => {
                return Err(Error::new(
                    ErrorKind::Invalid(format!("invalid matrix dimensions {nr}x{nc}")),
                    start,
                ));
            }
        };
        // Each float takes at least 2 bytes (mantissa and exponent control bytes).
        let n = rows
            .checked_mul(cols)
            .filter(|n| n.saturating_mul(2) <= self.remaining())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Invalid(format!(
                        "matrix {rows}x{cols} exceeds the {} remaining bytes",
                        self.remaining()
                    )),
                    start,
                )
            })?;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(elem(self)?);
        }
        Ok(Matrix { rows, cols, data })
    }

    /// `dlib::matrix<float>`.
    pub fn matrix_f32(&mut self) -> Result<Matrix<f32>> {
        self.matrix(Self::f32)
    }

    /// `dlib::matrix<double>`.
    pub fn matrix_f64(&mut self) -> Result<Matrix<f64>> {
        self.matrix(Self::f64)
    }

    /// `dlib::matrix<float,0,1>`: a column vector (dlib rejects any other column count).
    pub fn col_vector_f32(&mut self) -> Result<Vec<f32>> {
        let start = self.pos;
        let m = self.matrix_f32()?;
        Self::check_column(m, start)
    }

    /// `dlib::matrix<double,0,1>`.
    pub fn col_vector_f64(&mut self) -> Result<Vec<f64>> {
        let start = self.pos;
        let m = self.matrix_f64()?;
        Self::check_column(m, start)
    }

    fn check_column<T>(m: Matrix<T>, start: usize) -> Result<Vec<T>> {
        // A 0x0 matrix is how dlib serializes an empty column vector.
        if m.cols != 1 && !m.data.is_empty() {
            return Err(Error::new(
                ErrorKind::Invalid(format!(
                    "expected a column vector, found a {}x{} matrix",
                    m.rows, m.cols
                )),
                start,
            ));
        }
        Ok(m.data)
    }

    /// An `int` version number that must lie in `accepted`.
    pub fn int_version(
        &mut self,
        what: &'static str,
        accepted: std::ops::RangeInclusive<i32>,
    ) -> Result<i32> {
        let start = self.pos;
        let v = self.i32()?;
        if !accepted.contains(&v) {
            return Err(Error::new(
                ErrorKind::UnexpectedVersion {
                    what,
                    found: v.to_string(),
                },
                start,
            ));
        }
        Ok(v)
    }

    /// A version string that must be one of `accepted`.
    pub fn string_version(&mut self, what: &'static str, accepted: &[&str]) -> Result<String> {
        let start = self.pos;
        let v = self.string()?;
        if !accepted.contains(&v.as_str()) {
            return Err(Error::new(
                ErrorKind::UnexpectedVersion { what, found: v },
                start,
            ));
        }
        Ok(v)
    }

    fn tensor_shape(&mut self) -> Result<TensorShape> {
        let start = self.pos;
        let mut shape = [0usize; 4];
        for d in &mut shape {
            let v = self.i64()?;
            *d = usize::try_from(v).map_err(|_| {
                Error::new(
                    ErrorKind::Invalid(format!("negative tensor dimension {v}")),
                    start,
                )
            })?;
        }
        Ok(shape)
    }

    fn resizable_tensor_header(&mut self) -> Result<(TensorShape, usize)> {
        self.int_version("dlib::resizable_tensor", 2..=2)?;
        let start = self.pos;
        let shape = self.tensor_shape()?;
        let bytes = shape
            .iter()
            .try_fold(4usize, |acc, &d| acc.checked_mul(d))
            .filter(|&b| b <= self.remaining())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Invalid(format!(
                        "tensor {shape:?} exceeds the {} remaining bytes",
                        self.remaining()
                    )),
                    start,
                )
            })?;
        Ok((shape, bytes))
    }

    /// `dlib::resizable_tensor`: shape and raw little-endian `f32` data.
    pub fn resizable_tensor(&mut self) -> Result<(TensorShape, Vec<f32>)> {
        let (shape, bytes) = self.resizable_tensor_header()?;
        let data = self
            .bytes(bytes)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((shape, data))
    }

    /// Reads past a `dlib::resizable_tensor` without decoding it (training-time state).
    pub fn skip_resizable_tensor(&mut self) -> Result<()> {
        let (_, bytes) = self.resizable_tensor_header()?;
        self.bytes(bytes)?;
        Ok(())
    }

    /// `dlib::alias_tensor`: only a shape.
    pub fn alias_tensor(&mut self) -> Result<TensorShape> {
        self.int_version("dlib::alias_tensor", 1..=1)?;
        self.tensor_shape()
    }
}

#[cfg(test)]
mod tests {
    use super::writer::Writer;
    use super::*;

    fn reader_err<T: std::fmt::Debug>(
        bytes: &[u8],
        f: impl FnOnce(&mut Reader) -> Result<T>,
    ) -> Error {
        f(&mut Reader::new(bytes)).unwrap_err()
    }

    #[test]
    fn integers_match_dlib_encoding() {
        let mut w = Writer::default();
        let signed = [
            0i64,
            1,
            -1,
            127,
            128,
            -255,
            256,
            65535,
            i32::MAX as i64,
            i32::MIN as i64 + 1,
            i64::MAX,
            -i64::MAX,
        ];
        let unsigned = [0u64, 1, 255, 256, u32::MAX as u64, u64::MAX];
        for v in signed {
            w.i64(v);
        }
        for v in unsigned {
            w.u64(v);
        }
        let mut r = Reader::new(&w.buf);
        for v in signed {
            assert_eq!(r.i64().unwrap(), v);
        }
        for v in unsigned {
            assert_eq!(r.u64().unwrap(), v);
        }
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn integer_byte_layout() {
        // Control byte = size | sign bit, then little-endian magnitude.
        assert_eq!(Reader::new(&[0x01, 0x00]).i32().unwrap(), 0);
        assert_eq!(Reader::new(&[0x02, 0xe8, 0x03]).u64().unwrap(), 1000);
        assert_eq!(Reader::new(&[0x81, 0x01]).i32().unwrap(), -1);
        // Reserved bits 0x70 are ignored by dlib.
        assert_eq!(Reader::new(&[0x71, 0x05]).i32().unwrap(), 5);
    }

    #[test]
    fn integer_errors() {
        let e = reader_err(&[0x00], |r| r.i32());
        assert!(matches!(
            e.kind,
            ErrorKind::InvalidIntegerControl { control: 0, .. }
        ));
        // 5 bytes do not fit an int.
        let e = reader_err(&[0x05, 1, 2, 3, 4, 5], |r| r.i32());
        assert!(matches!(e.kind, ErrorKind::InvalidIntegerControl { .. }));
        // A sign bit on an unsigned value is invalid.
        let e = reader_err(&[0x81, 0x01], |r| r.u64());
        assert!(matches!(e.kind, ErrorKind::InvalidIntegerControl { .. }));
        // 4 bytes but larger than i32::MAX.
        let e = reader_err(&[0x04, 0xff, 0xff, 0xff, 0xff], |r| r.i32());
        assert_eq!(e.kind, ErrorKind::IntegerOutOfRange { ty: "int" });
        let e = reader_err(&[0x02, 0x01], |r| r.i32());
        assert_eq!(
            e.kind,
            ErrorKind::UnexpectedEof {
                needed: 2,
                available: 1
            }
        );
        assert_eq!(e.offset, 1);
    }

    #[test]
    fn bools() {
        let mut r = Reader::new(b"10x");
        assert!(r.bool().unwrap());
        assert!(!r.bool().unwrap());
        assert_eq!(r.bool().unwrap_err().kind, ErrorKind::InvalidBool(b'x'));
    }

    #[test]
    fn floats_round_trip() {
        let doubles = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            9.0,
            1e-5,
            1e300,
            -1e-300,
            5e-324,
            2.2250738585072014e-308,
            f64::MAX,
            f64::MIN_POSITIVE,
            std::f64::consts::PI,
        ];
        let floats = [
            0.0f32,
            0.1,
            -0.75,
            1e-45,
            f32::MIN_POSITIVE,
            f32::MAX,
            123456.79,
        ];
        let mut w = Writer::default();
        for v in doubles {
            w.f64(v);
        }
        for v in floats {
            w.f32(v);
        }
        let mut r = Reader::new(&w.buf);
        for v in doubles {
            let got = r.f64().unwrap();
            assert_eq!(got.to_bits() & !(1 << 63), v.to_bits() & !(1 << 63), "{v}");
        }
        for v in floats {
            assert_eq!(r.f32().unwrap(), v, "{v}");
        }
    }

    #[test]
    fn float_special_values() {
        let mut w = Writer::default();
        w.f64(f64::INFINITY);
        w.f64(f64::NEG_INFINITY);
        w.f64(f64::NAN);
        let mut r = Reader::new(&w.buf);
        assert_eq!(r.f32().unwrap(), f32::INFINITY);
        assert_eq!(r.f32().unwrap(), f32::NEG_INFINITY);
        assert!(r.f32().unwrap().is_nan());
        // Any exponent above is_nan (32002) also decodes to NaN, like dlib.
        let mut w = Writer::default();
        w.i64(1);
        w.i64(32005);
        assert!(Reader::new(&w.buf).f64().unwrap().is_nan());
    }

    #[test]
    fn nuclear_norm_bytes_from_the_frontal_detector() {
        // 9.0 = 18 * 2^-1, as found in the embedded frontal face detector.
        assert_eq!(Reader::new(&[0x01, 0x12, 0x81, 0x01]).f64().unwrap(), 9.0);
    }

    #[test]
    fn double_decoded_as_float_rounds_the_mantissa_first() {
        // dlib computes ldexp((float)mantissa, exponent): one rounding of the mantissa to 24
        // bits, then an exact scaling (for normal results).
        for v in [0.1f64, 1.0 / 3.0, 1e-5, 123.456789123, -7.77e-20] {
            let mut w = Writer::default();
            w.f64(v);
            let got = Reader::new(&w.buf).f32().unwrap();
            assert_eq!(got, v as f32, "{v}");
        }
    }

    #[test]
    fn legacy_float_format_is_rejected() {
        let e = reader_err(b"1.5 ", |r| r.f32());
        assert_eq!(e.kind, ErrorKind::LegacyFloatFormat);
    }

    #[test]
    fn strings_and_vectors() {
        let mut w = Writer::default();
        w.string("con_6");
        w.u64(3);
        for v in [1u64, 2, 3] {
            w.u64(v);
        }
        let mut r = Reader::new(&w.buf);
        assert_eq!(r.string().unwrap(), "con_6");
        assert_eq!(r.vec(1, |r, _| r.u64()).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn huge_lengths_are_rejected_before_allocating() {
        let mut w = Writer::default();
        w.u64(u64::MAX / 2);
        let e = reader_err(&w.buf, |r| r.vec(1, |r, _| r.u64()));
        assert!(matches!(e.kind, ErrorKind::Invalid(_)), "{e}");
        let mut w = Writer::default();
        w.i64(-1_000_000);
        w.i64(-1_000_000);
        let e = reader_err(&w.buf, |r| r.matrix_f32());
        assert!(matches!(e.kind, ErrorKind::Invalid(_)), "{e}");
    }

    #[test]
    fn matrices() {
        let mut w = Writer::default();
        w.matrix_f32(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        w.matrix_f32(3, 1, &[0.5, -0.5, 0.25]);
        w.matrix_f32(2, 2, &[0.0; 4]);
        let mut r = Reader::new(&w.buf);
        let m = r.matrix_f32().unwrap();
        assert_eq!((m.rows, m.cols), (2, 3));
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(r.col_vector_f32().unwrap(), vec![0.5, -0.5, 0.25]);
        assert!(matches!(
            r.col_vector_f32().unwrap_err().kind,
            ErrorKind::Invalid(_)
        ));
        // Legacy positive dimensions are still accepted.
        let mut w = Writer::default();
        w.i64(1);
        w.i64(2);
        w.f32(1.0);
        w.f32(2.0);
        assert_eq!(
            Reader::new(&w.buf).matrix_f32().unwrap().data,
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn tensors() {
        let mut w = Writer::default();
        w.resizable_tensor([1, 2, 1, 2], &[1.0, -2.0, 3.5, 0.0]);
        w.alias_tensor([4, 3, 2, 1]);
        let mut r = Reader::new(&w.buf);
        let (shape, data) = r.resizable_tensor().unwrap();
        assert_eq!(shape, [1, 2, 1, 2]);
        assert_eq!(data, vec![1.0, -2.0, 3.5, 0.0]);
        assert_eq!(r.alias_tensor().unwrap(), [4, 3, 2, 1]);
    }

    #[test]
    fn error_context_path() {
        let mut w = Writer::default();
        w.u64(2);
        w.u64(7);
        let mut r = Reader::new(&w.buf);
        let e = r
            .context(
                || "model",
                |r| r.context(|| "items", |r| r.vec(1, |r, _| r.u64())),
            )
            .unwrap_err();
        assert_eq!(
            e.context_path().collect::<Vec<_>>(),
            ["model", "items", "[1]"]
        );
        assert!(e.to_string().contains("(in model > items > [1])"), "{e}");
    }
}
