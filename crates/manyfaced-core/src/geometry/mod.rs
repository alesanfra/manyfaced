//! Rectangles, points and similarity/affine point transforms.
//!
//! Ported from `dlib/geometry/rectangle.h`, `vector.h` and `point_transforms.h` (dlib v20.0.1,
//! Boost Software License 1.0, see `THIRD_PARTY_NOTICES`).

pub mod drect;
mod transform;

pub use drect::DRect;
pub use transform::{PointTransformAffine, find_affine_transform, find_similarity_transform};

/// dlib's integer `point` (`vector<long,2>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Point {
    pub x: i64,
    pub y: i64,
}

impl Point {
    pub const fn new(x: i64, y: i64) -> Self {
        Point { x, y }
    }
}

/// dlib's `dpoint` (`vector<double,2>`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DPoint {
    pub x: f64,
    pub y: f64,
}

impl DPoint {
    pub const fn new(x: f64, y: f64) -> Self {
        DPoint { x, y }
    }

    /// dlib's conversion of a floating point vector to an integer one: `floor(v + 0.5)` per
    /// coordinate (`geometry/vector.h`).
    pub fn round(self) -> Point {
        Point::new((self.x + 0.5).floor() as i64, (self.y + 0.5).floor() as i64)
    }

    /// `length_squared()`: `x*x + y*y`.
    pub fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }
}

impl From<Point> for DPoint {
    fn from(p: Point) -> Self {
        DPoint::new(p.x as f64, p.y as f64)
    }
}

impl std::ops::Add for DPoint {
    type Output = DPoint;
    fn add(self, o: DPoint) -> DPoint {
        DPoint::new(self.x + o.x, self.y + o.y)
    }
}

impl std::ops::Sub for DPoint {
    type Output = DPoint;
    fn sub(self, o: DPoint) -> DPoint {
        DPoint::new(self.x - o.x, self.y - o.y)
    }
}

/// dlib's `rectangle`: inclusive integer bounds. Empty when `left > right` or `top > bottom`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rect {
    pub left: i64,
    pub top: i64,
    pub right: i64,
    pub bottom: i64,
}

impl Rect {
    pub const fn new(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        Rect {
            left,
            top,
            right,
            bottom,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.top > self.bottom || self.left > self.right
    }

    /// dlib's `rectangle::operator+`: the bounding box of both. An empty operand is ignored.
    pub fn union(self, other: Rect) -> Rect {
        if other.is_empty() {
            self
        } else if self.is_empty() {
            other
        } else {
            Rect::new(
                self.left.min(other.left),
                self.top.min(other.top),
                self.right.max(other.right),
                self.bottom.max(other.bottom),
            )
        }
    }

    /// `right - left + 1`, or 0 if empty. Saturates at `u64::MAX` for the full `i64` range.
    pub fn width(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            self.right.abs_diff(self.left).saturating_add(1)
        }
    }

    /// `bottom - top + 1`, or 0 if empty. Saturates at `u64::MAX` for the full `i64` range.
    pub fn height(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            self.bottom.abs_diff(self.top).saturating_add(1)
        }
    }

    /// `width * height`, saturating at `u64::MAX`.
    pub fn area(&self) -> u64 {
        self.width().saturating_mul(self.height())
    }

    pub fn tl_corner(&self) -> Point {
        Point::new(self.left, self.top)
    }

    pub fn tr_corner(&self) -> Point {
        Point::new(self.right, self.top)
    }

    pub fn bl_corner(&self) -> Point {
        Point::new(self.left, self.bottom)
    }

    pub fn br_corner(&self) -> Point {
        Point::new(self.right, self.bottom)
    }

    pub fn contains(&self, p: Point) -> bool {
        !(p.x < self.left || p.x > self.right || p.y < self.top || p.y > self.bottom)
    }

    /// dlib's `center(rect)`: the integer centre, rounding towards negative infinity.
    ///
    /// The sums are in `i128`, so that they cannot overflow. The centre of two `i64` values is
    /// always an `i64`.
    pub fn center(&self) -> Point {
        let half = |a: i64, b: i64| {
            let mut s = i128::from(a) + i128::from(b) + 1;
            if s < 0 {
                s -= 1;
            }
            (s / 2) as i64
        };
        Point::new(half(self.left, self.right), half(self.top, self.bottom))
    }

    /// dlib's `dcenter(rect)`: `((left + right) / 2, (top + bottom) / 2)` in `f64`. The sums
    /// are in `i128`, so that they cannot overflow.
    pub fn dcenter(&self) -> DPoint {
        DPoint::new(
            (i128::from(self.left) + i128::from(self.right)) as f64 / 2.0,
            (i128::from(self.top) + i128::from(self.bottom)) as f64 / 2.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_semantics_match_dlib() {
        let r = Rect::new(10, 20, 29, 59);
        assert_eq!((r.width(), r.height(), r.area()), (20, 40, 800));
        assert_eq!(r.center(), Point::new(20, 40));
        assert_eq!(r.dcenter(), DPoint::new(19.5, 39.5));
        assert!(r.contains(Point::new(10, 59)) && !r.contains(Point::new(30, 20)));
        let e = Rect::new(5, 5, 4, 10);
        assert!(e.is_empty());
        assert_eq!((e.width(), e.area()), (0, 0));
        // Negative centres round towards -inf like dlib.
        assert_eq!(Rect::new(-3, -3, 0, 0).center(), Point::new(-1, -1));
        assert_eq!(Rect::new(-4, -4, 0, 0).center(), Point::new(-2, -2));
    }

    #[test]
    fn rounding_is_floor_plus_half() {
        assert_eq!(DPoint::new(1.5, -1.5).round(), Point::new(2, -1));
        assert_eq!(DPoint::new(2.49, -2.51).round(), Point::new(2, -3));
    }
}
