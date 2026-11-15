//! dlib's `drectangle` and `rotate_point`, ported from `dlib/geometry/drectangle.h` and
//! `point_transforms.h` (dlib v20.0.1, Boost Software License 1.0, see `THIRD_PARTY_NOTICES`).

use super::{DPoint, Rect};

/// A rectangle with `f64` bounds. Empty when `left > right` or `top > bottom`; widths and
/// heights follow dlib's inclusive convention (`right - left + 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DRect {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

impl Default for DRect {
    /// dlib's default `drectangle`: `(0, 0, -1, -1)`, which is empty.
    fn default() -> Self {
        DRect::new(0.0, 0.0, -1.0, -1.0)
    }
}

impl From<Rect> for DRect {
    fn from(r: Rect) -> Self {
        DRect::new(r.left as f64, r.top as f64, r.right as f64, r.bottom as f64)
    }
}

impl DRect {
    pub const fn new(left: f64, top: f64, right: f64, bottom: f64) -> Self {
        DRect {
            left,
            top,
            right,
            bottom,
        }
    }

    /// The single-point rectangle `drectangle(p)`.
    pub fn from_point(p: DPoint) -> Self {
        DRect::new(p.x, p.y, p.x, p.y)
    }

    /// `drectangle(p1, p2)`: the union of the two single-point rectangles.
    pub fn from_corners(p1: DPoint, p2: DPoint) -> Self {
        DRect::from_point(p1).union(&DRect::from_point(p2))
    }

    /// dlib's `operator rectangle()`: each bound rounded with `std::lround` (half away from
    /// zero). Values outside the `i64` range saturate.
    pub fn to_rect(&self) -> Rect {
        let lround = |v: f64| v.round() as i64;
        Rect::new(
            lround(self.left),
            lround(self.top),
            lround(self.right),
            lround(self.bottom),
        )
    }

    pub fn is_empty(&self) -> bool {
        self.top > self.bottom || self.left > self.right
    }

    pub fn width(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.right - self.left + 1.0
        }
    }

    pub fn height(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.bottom - self.top + 1.0
        }
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    pub fn tl_corner(&self) -> DPoint {
        DPoint::new(self.left, self.top)
    }

    pub fn tr_corner(&self) -> DPoint {
        DPoint::new(self.right, self.top)
    }

    pub fn bl_corner(&self) -> DPoint {
        DPoint::new(self.left, self.bottom)
    }

    pub fn br_corner(&self) -> DPoint {
        DPoint::new(self.right, self.bottom)
    }

    /// dlib's `center(drectangle)`: `((left + right) / 2, (top + bottom) / 2)`.
    pub fn center(&self) -> DPoint {
        DPoint::new(
            (self.left + self.right) / 2.0,
            (self.top + self.bottom) / 2.0,
        )
    }

    /// `operator+`: the smallest rectangle containing both (an empty operand is ignored).
    pub fn union(&self, rhs: &DRect) -> DRect {
        if rhs.is_empty() {
            *self
        } else if self.is_empty() {
            *rhs
        } else {
            DRect::new(
                self.left.min(rhs.left),
                self.top.min(rhs.top),
                self.right.max(rhs.right),
                self.bottom.max(rhs.bottom),
            )
        }
    }

    /// `operator+=(point)`.
    pub fn include(&mut self, p: DPoint) {
        *self = self.union(&DRect::from_point(p));
    }

    pub fn intersect(&self, rhs: &DRect) -> DRect {
        DRect::new(
            self.left.max(rhs.left),
            self.top.max(rhs.top),
            self.right.min(rhs.right),
            self.bottom.min(rhs.bottom),
        )
    }

    /// `translate_rect(rect, (dx, dy))`.
    pub fn translate(&self, dx: f64, dy: f64) -> DRect {
        DRect::new(
            self.left + dx,
            self.top + dy,
            self.right + dx,
            self.bottom + dy,
        )
    }

    /// `grow_rect(rect, num)` (implemented by dlib as `shrink_rect(rect, -num)`).
    pub fn grow(&self, num: f64) -> DRect {
        let num = -num;
        DRect::new(
            self.left + num,
            self.top + num,
            self.right - num,
            self.bottom - num,
        )
    }

    /// `centered_drect(p, width, height)`.
    pub fn centered(p: DPoint, width: f64, height: f64) -> DRect {
        let width = width - 1.0;
        let height = height - 1.0;
        DRect::new(
            p.x - width / 2.0,
            p.y - height / 2.0,
            p.x + width / 2.0,
            p.y + height / 2.0,
        )
    }
}

/// dlib's `rotate_point(center, p, angle)`: `point_rotator(angle)(p - center) + center`.
pub fn rotate_point(center: DPoint, p: DPoint, angle: f64) -> DPoint {
    let (sin, cos) = (angle.sin(), angle.cos());
    let d = p - center;
    let x = cos * d.x - sin * d.y;
    let y = sin * d.x + cos * d.y;
    DPoint::new(x, y) + center
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drect_semantics_match_dlib() {
        let r = DRect::new(1.5, 2.0, 10.5, 4.0);
        assert_eq!((r.width(), r.height(), r.area()), (10.0, 3.0, 30.0));
        assert_eq!(r.center(), DPoint::new(6.0, 3.0));
        assert!(DRect::default().is_empty());
        assert_eq!(DRect::default().area(), 0.0);
        // lround rounds halves away from zero.
        assert_eq!(
            DRect::new(-0.5, 0.5, 2.5, -2.5).to_rect(),
            Rect::new(-1, 1, 3, -3)
        );
        let c = DRect::centered(DPoint::new(10.0, 10.0), 5.0, 3.0);
        assert_eq!(c, DRect::new(8.0, 9.0, 12.0, 11.0));
        assert_eq!(c.width(), 5.0);
        let mut u = DRect::default();
        u.include(DPoint::new(3.0, 4.0));
        u.include(DPoint::new(-1.0, 6.0));
        assert_eq!(u, DRect::new(-1.0, 4.0, 3.0, 6.0));
        assert_eq!(u.grow(1.0), DRect::new(-2.0, 3.0, 4.0, 7.0));
    }

    #[test]
    fn rotation_about_a_center() {
        let p = rotate_point(
            DPoint::new(1.0, 1.0),
            DPoint::new(2.0, 1.0),
            std::f64::consts::FRAC_PI_2,
        );
        assert!((p.x - 1.0).abs() < 1e-15 && (p.y - 2.0).abs() < 1e-15);
    }
}
