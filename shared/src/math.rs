use std::{
    f32::EPSILON,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub const VECTOR_X: Vector = Vector::from_vector(1.0, 0.0, 0.0);
pub const VECTOR_Y: Vector = Vector::from_vector(0.0, 1.0, 0.0);
pub const VECTOR_Z: Vector = Vector::from_vector(0.0, 0.0, 1.0);

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Transform {
    pub x: Vector,
    pub y: Vector,
    pub z: Vector,
    pub w: Vector,
}

impl Vector {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vector { x, y, z, w }
    }

    pub fn from_point(x: f32, y: f32, z: f32) -> Self {
        Vector { x, y, z, w: 1.0 }
    }

    pub fn from_vector(x: f32, y: f32, z: f32) -> Self {
        Vector { x, y, z, w: 0.0 }
    }

    pub fn to_point(self) -> Vector {
        Vector {
            x: self.x,
            y: self.y,
            z: self.z,
            w: 1.0,
        }
    }

    pub fn to_vector(self) -> Vector {
        Vector {
            x: self.x,
            y: self.y,
            z: self.z,
            w: 0.0,
        }
    }

    pub fn zero_vector() -> Self {
        Vector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    pub fn zero_point() -> Self {
        Vector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    pub fn dot(self, other: Vector) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn cross(self, other: Vector) -> Vector {
        Vector::from_vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn canonicalize(self) -> Vector {
        self / self.w
    }

    pub fn normalize(self) -> Vector {
        self / self.length()
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn hadamard(self, other: Vector) -> Vector {
        Vector {
            w: self.w * other.w,
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    pub fn slerp(self, other: Vector, t: f32) -> Vector {
        let mut dot = self.dot(other);

        let mut other_adjusted = other;
        if dot < 0.0 {
            other_adjusted = -other_adjusted;
            dot = -dot;
        }

        let dot = dot.clamp(-1.0, 1.0);

        if dot > 0.9995 {
            // Use linear interpolation for very close quaternions
            return (self + (other_adjusted - self) * t).normalize();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();

        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        self * a + other_adjusted * b
    }

    pub fn project_onto(self, other: Vector) -> Vector {
        other.dot(self) * other
    }
}

impl Transform {
    pub fn identity() -> Self {
        Transform {
            x: Vector::new(1.0, 0.0, 0.0, 0.0),
            y: Vector::new(0.0, 1.0, 0.0, 0.0),
            z: Vector::new(0.0, 0.0, 1.0, 0.0),
            w: Vector::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        Transform {
            x: Vector::new(1.0, 0.0, 0.0, 0.0),
            y: Vector::new(0.0, 1.0, 0.0, 0.0),
            z: Vector::new(0.0, 0.0, 1.0, 0.0),
            w: Vector::new(x, y, z, 1.0),
        }
    }

    pub fn scaling(x: f32, y: f32, z: f32) -> Self {
        Transform {
            x: Vector::new(x, 0.0, 0.0, 0.0),
            y: Vector::new(0.0, y, 0.0, 0.0),
            z: Vector::new(0.0, 0.0, z, 0.0),
            w: Vector::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    pub fn uniform_scaling(scale: f32) -> Self {
        Self::scaling(scale, scale, scale)
    }

    pub fn rotation_x(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Transform {
            x: Vector::new(1.0, 0.0, 0.0, 0.0),
            y: Vector::new(0.0, c, s, 0.0),
            z: Vector::new(0.0, -s, c, 0.0),
            w: Vector::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    pub fn rotation_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Transform {
            x: Vector::new(c, 0.0, -s, 0.0),
            y: Vector::new(0.0, 1.0, 0.0, 0.0),
            z: Vector::new(s, 0.0, c, 0.0),
            w: Vector::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    pub fn rotation_z(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Transform {
            x: Vector::new(c, s, 0.0, 0.0),
            y: Vector::new(-s, c, 0.0, 0.0),
            z: Vector::new(0.0, 0.0, 1.0, 0.0),
            w: Vector::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    pub fn look_at(eye: Vector, target: Vector, up: Vector) -> Self {
        let f = (eye - target).normalize();
        let r = up.cross(f).normalize();
        let u = f.cross(r);

        Transform {
            x: Vector::new(r.x, u.x, f.x, 0.0),
            y: Vector::new(r.y, u.y, f.y, 0.0),
            z: Vector::new(r.z, u.z, f.z, 0.0),

            w: Vector::new(-r.dot(eye), -u.dot(eye), -f.dot(eye), 1.0),
        }
    }

    pub fn perspective(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let f = 1.0 / (fov_y_radians / 2.0).tan();
        let range_inv = 1.0 / (z_near - z_far);

        Transform {
            x: Vector::new(f / aspect_ratio, 0.0, 0.0, 0.0),
            y: Vector::new(0.0, f, 0.0, 0.0),
            z: Vector::new(0.0, 0.0, (z_near + z_far) * range_inv, -1.0),
            w: Vector::new(0.0, 0.0, z_near * z_far * range_inv * 2.0, 0.0),
        }
    }

    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let w = right - left;
        let h = top - bottom;
        let d = near - far;

        Transform {
            x: Vector::new(2.0 / w, 0.0, 0.0, 0.0),
            y: Vector::new(0.0, 2.0 / h, 0.0, 0.0),
            z: Vector::new(0.0, 0.0, 2.0 / d, 0.0),
            w: Vector::new(
                -(right + left) / w,
                -(top + bottom) / h,
                (near + far) / d,
                1.0,
            ),
        }
    }

    pub fn transpose(self) -> Self {
        Transform {
            x: Vector::new(self.x.x, self.y.x, self.z.x, self.w.x),
            y: Vector::new(self.x.y, self.y.y, self.z.y, self.w.y),
            z: Vector::new(self.x.z, self.y.z, self.z.z, self.w.z),
            w: Vector::new(self.x.w, self.y.w, self.z.w, self.w.w),
        }
    }

    pub fn get(&self, col: usize, row: usize) -> f32 {
        match col {
            0 => self.x[row],
            1 => self.y[row],
            2 => self.z[row],
            3 => self.w[row],
            _ => panic!("Column index out of bounds"),
        }
    }

    pub fn right_vector(&self) -> Vector {
        self.x.to_vector()
    }
    pub fn up_vector(&self) -> Vector {
        self.y.to_vector()
    }
    pub fn forward_vector(&self) -> Vector {
        -self.z.to_vector()
    }
    pub fn position(&self) -> Vector {
        self.w.to_point()
    }

    pub fn inverse_rotation(self) -> Self {
        Transform {
            x: Vector::new(self.x.x, self.y.x, self.z.x, 0.0),
            y: Vector::new(self.x.y, self.y.y, self.z.y, 0.0),
            z: Vector::new(self.x.z, self.y.z, self.z.z, 0.0),
            w: Vector::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    pub fn inverse_affine(self) -> Self {
        let rotation_transpose = Transform {
            x: Vector::new(self.x.x, self.y.x, self.z.x, 0.0),
            y: Vector::new(self.x.y, self.y.y, self.z.y, 0.0),
            z: Vector::new(self.x.z, self.y.z, self.z.z, 0.0),
            w: Vector::new(0.0, 0.0, 0.0, 1.0),
        };

        let mut t_inv = Vector::new(-self.w.x, -self.w.y, -self.w.z, 1.0);
        t_inv = rotation_transpose * t_inv;

        Transform {
            x: rotation_transpose.x.to_vector(),
            y: rotation_transpose.y.to_vector(),
            z: rotation_transpose.z.to_vector(),
            w: Vector::new(t_inv.x, t_inv.y, t_inv.z, 1.0),
        }
    }

    /// Please note that this function is rather __expensive__, and that for most cases `inverse_affine` is enough
    pub fn inverse(&self) -> Option<Self> {
        let c0 = self.x;
        let c1 = self.y;
        let c2 = self.z;
        let c3 = self.w;

        let s0 = c0.x * c1.y - c1.x * c0.y;
        let s1 = c0.x * c1.z - c1.x * c0.z;
        let s2 = c0.x * c1.w - c1.x * c0.w;
        let s3 = c0.y * c1.z - c1.y * c0.z;
        let s4 = c0.y * c1.w - c1.y * c0.w;
        let s5 = c0.z * c1.w - c1.z * c0.w;

        let c5 = c2.z * c3.w - c3.z * c2.w;
        let c4 = c2.y * c3.w - c3.y * c2.w;
        let c3_val = c2.y * c3.z - c3.y * c2.z;
        let c2_val = c2.x * c3.w - c3.x * c2.w;
        let c1_val = c2.x * c3.z - c3.x * c2.z;
        let c0_val = c2.x * c3.y - c3.x * c2.y;

        let det = s0 * c5 - s1 * c4 + s2 * c3_val + s3 * c2_val - s4 * c1_val + s5 * c0_val;

        if det.abs() < EPSILON {
            return None;
        }

        let inv_det = 1.0 / det;

        let x_x = (c1.y * c5 - c1.z * c4 + c1.w * c3_val) * inv_det;
        let x_y = (-c1.x * c5 + c1.z * c2_val - c1.w * c1_val) * inv_det;
        let x_z = (c1.x * c4 - c1.y * c2_val + c1.w * c0_val) * inv_det;
        let x_w = (-c1.x * c3_val + c1.y * c1_val - c1.z * c0_val) * inv_det;

        let y_x = (-c0.y * c5 + c0.z * c4 - c0.w * c3_val) * inv_det;
        let y_y = (c0.x * c5 - c0.z * c2_val + c0.w * c1_val) * inv_det;
        let y_z = (-c0.x * c4 + c0.y * c2_val - c0.w * c0_val) * inv_det;
        let y_w = (c0.x * c3_val - c0.y * c1_val + c0.z * c0_val) * inv_det;

        let z_x = (c3.y * s5 - c3.z * s4 + c3.w * s3) * inv_det;
        let z_y = (-c3.x * s5 + c3.z * s2 - c3.w * s1) * inv_det;
        let z_z = (c3.x * s4 - c3.y * s2 + c3.w * s0) * inv_det;
        let z_w = (-c3.x * s3 + c3.y * s1 - c3.z * s0) * inv_det;

        let w_x = (-c2.y * s5 + c2.z * s4 - c2.w * s3) * inv_det;
        let w_y = (c2.x * s5 - c2.z * s2 + c2.w * s1) * inv_det;
        let w_z = (-c2.x * s4 + c2.y * s2 - c2.w * s0) * inv_det;
        let w_w = (c2.x * s3 - c2.y * s1 + c2.z * s0) * inv_det;

        Some(Transform {
            x: Vector::new(x_x, x_y, x_z, x_w),
            y: Vector::new(y_x, y_y, y_z, y_w),
            z: Vector::new(z_x, z_y, z_z, z_w),
            w: Vector::new(w_x, w_y, w_z, w_w),
        })
    }
}

impl Mul<Vector> for Transform {
    type Output = Vector;
    fn mul(self, v: Vector) -> Vector {
        self.x * v.x + self.y * v.y + self.z * v.z + self.w * v.w
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;
    fn mul(self, other: Transform) -> Transform {
        Transform {
            x: self * other.x,
            y: self * other.y,
            z: self * other.z,
            w: self * other.w,
        }
    }
}

impl Add for Vector {
    type Output = Vector;
    fn add(self, other: Vector) -> Vector {
        Vector {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vector {
    type Output = Vector;
    fn sub(self, other: Vector) -> Vector {
        Vector {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Vector {
    type Output = Vector;
    fn mul(self, scalar: f32) -> Vector {
        Vector {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }
}

impl MulAssign<f32> for Vector {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}

impl Mul<Vector> for f32 {
    type Output = Vector;
    fn mul(self, vector: Vector) -> Vector {
        Vector {
            x: vector.x * self,
            y: vector.y * self,
            z: vector.z * self,
            w: vector.w * self,
        }
    }
}

impl Div<f32> for Vector {
    type Output = Vector;
    fn div(self, scalar: f32) -> Vector {
        Vector {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
            w: self.w / scalar,
        }
    }
}

impl DivAssign<f32> for Vector {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Neg for Vector {
    type Output = Vector;
    fn neg(self) -> Vector {
        Vector {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl Index<usize> for Vector {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Index out of bounds"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    // normals point inward
    pub planes: [Vector; 6],
}

impl Frustum {
    pub fn from_matrix(m: Transform) -> Self {
        let t = m.transpose();

        let row0 = t.x;
        let row1 = t.y;
        let row2 = t.z;
        let row3 = t.w;

        let normalize_plane = |p: Vector| {
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if len > EPSILON { p / len } else { p }
        };

        Frustum {
            planes: [
                normalize_plane(row3 + row0), // left
                normalize_plane(row3 - row0), // right
                normalize_plane(row3 + row1), // bottom
                normalize_plane(row3 - row1), // top
                normalize_plane(row3 + row2), // near
                normalize_plane(row3 - row2), // far
            ],
        }
    }

    pub fn intersects_aabb(&self, bounds: BoundingBox) -> bool {
        for plane in &self.planes {
            let p_x = if plane.x > 0.0 {
                bounds.max.0
            } else {
                bounds.min.0
            };
            let p_y = if plane.y > 0.0 {
                bounds.max.1
            } else {
                bounds.min.1
            };
            let p_z = if plane.z > 0.0 {
                bounds.max.2
            } else {
                bounds.min.2
            };

            let p_vertex = Vector::new(p_x, p_y, p_z, 1.0);

            if plane.dot(p_vertex) < 0.0 {
                return false;
            }
        }

        true
    }
}

#[derive(Clone, Copy)]
pub struct BoundingBox {
    min: (f32, f32, f32),
    max: (f32, f32, f32),
}

impl BoundingBox {
    pub fn new(min: Vector, max: Vector) -> Self {
        Self {
            min: (min.x, min.y, min.z),
            max: (max.x, max.y, max.z),
        }
    }

    pub fn contains(&self, other: &BoundingBox) -> bool {
        self.min.0 < other.min.0
            && self.min.1 < other.min.1
            && self.min.2 < other.min.2
            && self.max.0 > other.max.0
            && self.max.1 > other.max.1
            && self.max.2 > other.max.2
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        (self.min.0 <= other.max.0 && self.max.0 >= other.min.0)
            && (self.min.1 <= other.max.1 && self.max.1 >= other.min.1)
            && (self.min.2 <= other.max.2 && self.max.2 >= other.min.2)
    }

    pub fn union(self, other: BoundingBox) -> BoundingBox {
        Self {
            min: (
                self.min.0.min(other.min.0),
                self.min.1.min(other.min.1),
                self.min.2.min(other.min.2),
            ),
            max: (
                self.max.0.max(other.max.0),
                self.max.1.max(other.max.1),
                self.max.2.max(other.max.2),
            ),
        }
    }
}
