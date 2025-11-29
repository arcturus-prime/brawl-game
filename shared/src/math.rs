use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

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
            if len > 1e-6 { p / len } else { p }
        };

        Frustum {
            planes: [
                normalize_plane(row3 + row0), // Left
                normalize_plane(row3 - row0), // Right
                normalize_plane(row3 + row1), // Bottom
                normalize_plane(row3 - row1), // Top
                normalize_plane(row3 + row2), // Near
                normalize_plane(row3 - row2), // Far
            ],
        }
    }

    pub fn intersects_aabb(&self, min: Vector, max: Vector) -> bool {
        for plane in &self.planes {
            let p_x = if plane.x > 0.0 { max.x } else { min.x };
            let p_y = if plane.y > 0.0 { max.y } else { min.y };
            let p_z = if plane.z > 0.0 { max.z } else { min.z };

            let p_vertex = Vector::new(p_x, p_y, p_z, 1.0);

            if plane.dot(p_vertex) < 0.0 {
                return false;
            }
        }

        true
    }
}
