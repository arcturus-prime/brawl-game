use std::{
    f32::EPSILON,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub const VECTOR_X: Vector3 = Vector3::new(1.0, 0.0, 0.0);
pub const VECTOR_Y: Vector3 = Vector3::new(0.0, 1.0, 0.0);
pub const VECTOR_Z: Vector3 = Vector3::new(0.0, 0.0, 1.0);

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vector3 { x, y, z }
    }

    pub fn zero() -> Self {
        Vector3::new(0.0, 0.0, 0.0)
    }

    pub fn dot(self, other: Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Vector3) -> Vector3 {
        Vector3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(self) -> Vector3 {
        let len = self.length();
        if len > EPSILON {
            self / len
        } else {
            Vector3::zero()
        }
    }

    pub fn hadamard(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }

    pub fn project_onto(self, other: Vector3) -> Vector3 {
        let dot = other.dot(self);
        let other_len_sq = other.length_squared();
        if other_len_sq > EPSILON {
            other * (dot / other_len_sq)
        } else {
            Vector3::zero()
        }
    }

    pub fn lerp(self, other: Vector3, t: f32) -> Vector3 {
        self + (other - self) * t
    }
}

impl Add for Vector3 {
    type Output = Vector3;
    fn add(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vector3 {
    type Output = Vector3;
    fn sub(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl SubAssign for Vector3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Vector3 {
    type Output = Vector3;
    fn mul(self, scalar: f32) -> Vector3 {
        Vector3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl MulAssign<f32> for Vector3 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}

impl Mul<Vector3> for f32 {
    type Output = Vector3;
    fn mul(self, vector: Vector3) -> Vector3 {
        Vector3::new(vector.x * self, vector.y * self, vector.z * self)
    }
}

impl Div<f32> for Vector3 {
    type Output = Vector3;
    fn div(self, scalar: f32) -> Vector3 {
        Vector3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl DivAssign<f32> for Vector3 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Neg for Vector3 {
    type Output = Vector3;
    fn neg(self) -> Vector3 {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quaternion {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Quaternion { x, y, z, w }
    }

    pub fn identity() -> Self {
        Quaternion::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn from_axis_angle(axis: Vector3, angle: f32) -> Self {
        let half_angle = angle * 0.5;
        let s = half_angle.sin();
        let c = half_angle.cos();

        let normalized_axis = axis.normalize();
        Quaternion::new(
            normalized_axis.x * s,
            normalized_axis.y * s,
            normalized_axis.z * s,
            c,
        )
    }

    pub fn to_axis_angle(self) -> (Vector3, f32) {
        let quat = self.normalize();

        if quat.w.abs() >= 1.0 - EPSILON {
            return (VECTOR_X, 0.0);
        }

        let angle = 2.0 * quat.w.acos();
        let s = (1.0 - quat.w * quat.w).sqrt();

        let axis = if s < EPSILON {
            // If s is too small, return arbitrary axis
            VECTOR_X
        } else {
            Vector3::new(quat.x / s, quat.y / s, quat.z / s)
        };

        (axis, angle)
    }

    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        let (sr, cr) = (roll * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sy, cy) = (yaw * 0.5).sin_cos();

        Quaternion::new(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    }

    pub fn dot(self, other: Quaternion) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    pub fn normalize(self) -> Quaternion {
        let len = self.length();
        if len > EPSILON {
            self / len
        } else {
            Quaternion::identity()
        }
    }

    pub fn conjugate(self) -> Quaternion {
        Quaternion::new(-self.x, -self.y, -self.z, self.w)
    }

    pub fn inverse(self) -> Quaternion {
        let len_sq = self.length_squared();
        if len_sq > EPSILON {
            let inv_len_sq = 1.0 / len_sq;
            Quaternion::new(
                -self.x * inv_len_sq,
                -self.y * inv_len_sq,
                -self.z * inv_len_sq,
                self.w * inv_len_sq,
            )
        } else {
            Quaternion::identity()
        }
    }

    pub fn multiply(self, other: Quaternion) -> Quaternion {
        Quaternion::new(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )
    }

    pub fn rotate_vector(self, v: Vector3) -> Vector3 {
        // Optimized implementation: v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
        let q_xyz = Vector3::new(self.x, self.y, self.z);
        let t = q_xyz.cross(v) * 2.0;
        v + (t * self.w) + q_xyz.cross(t)
    }

    pub fn slerp(self, other: Quaternion, t: f32) -> Quaternion {
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

impl Add for Quaternion {
    type Output = Quaternion;
    fn add(self, other: Quaternion) -> Quaternion {
        Quaternion::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }
}

impl AddAssign for Quaternion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Quaternion {
    type Output = Quaternion;
    fn sub(self, other: Quaternion) -> Quaternion {
        Quaternion::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }
}

impl SubAssign for Quaternion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Quaternion {
    type Output = Quaternion;
    fn mul(self, scalar: f32) -> Quaternion {
        Quaternion::new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            self.w * scalar,
        )
    }
}

impl MulAssign<f32> for Quaternion {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}

impl Mul<Quaternion> for f32 {
    type Output = Quaternion;
    fn mul(self, quat: Quaternion) -> Quaternion {
        Quaternion::new(quat.x * self, quat.y * self, quat.z * self, quat.w * self)
    }
}

impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;
    fn mul(self, other: Quaternion) -> Quaternion {
        self.multiply(other)
    }
}

impl MulAssign for Quaternion {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div<f32> for Quaternion {
    type Output = Quaternion;
    fn div(self, scalar: f32) -> Quaternion {
        Quaternion::new(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
            self.w / scalar,
        )
    }
}

impl DivAssign<f32> for Quaternion {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Quaternion {
        Quaternion::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Index<usize> for Quaternion {
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

impl IndexMut<usize> for Quaternion {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: Vector3,
    pub distance: f32,
}

impl Plane {
    pub fn new(normal: Vector3, distance: f32) -> Self {
        let normalized = normal.normalize();
        Plane {
            normal: normalized,
            distance,
        }
    }

    pub fn from_point_normal(point: Vector3, normal: Vector3) -> Self {
        let normalized = normal.normalize();
        let distance = normalized.dot(point);
        Plane {
            normal: normalized,
            distance,
        }
    }

    pub fn from_points(p1: Vector3, p2: Vector3, p3: Vector3) -> Self {
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let normal = v1.cross(v2).normalize();
        let distance = normal.dot(p1);
        Plane { normal, distance }
    }

    pub fn distance_to_point(self, point: Vector3) -> f32 {
        self.normal.dot(point) - self.distance
    }

    pub fn project_point(self, point: Vector3) -> Vector3 {
        let dist = self.distance_to_point(point);
        point - self.normal * dist
    }

    pub fn closest_point(self, point: Vector3) -> Vector3 {
        self.project_point(point)
    }

    pub fn intersects_line(self, start: Vector3, end: Vector3) -> Option<Vector3> {
        let direction = end - start;
        let denom = self.normal.dot(direction);

        if denom.abs() < EPSILON {
            return None; // Line is parallel to plane
        }

        let t = (self.distance - self.normal.dot(start)) / denom;

        if t >= 0.0 && t <= 1.0 {
            Some(start + direction * t)
        } else {
            None // Intersection is outside the line segment
        }
    }

    pub fn flip(self) -> Plane {
        Plane {
            normal: -self.normal,
            distance: -self.distance,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BoundingBox {
    pub min: Vector3,
    pub max: Vector3,
}

impl BoundingBox {
    pub fn new(min: Vector3, max: Vector3) -> Self {
        BoundingBox { min, max }
    }

    pub fn contains(&self, other: &BoundingBox) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.min.z <= other.min.z
            && self.max.x >= other.max.x
            && self.max.y >= other.max.y
            && self.max.z >= other.max.z
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        (self.min.x <= other.max.x && self.max.x >= other.min.x)
            && (self.min.y <= other.max.y && self.max.y >= other.min.y)
            && (self.min.z <= other.max.z && self.max.z >= other.min.z)
    }

    pub fn union(self, other: BoundingBox) -> BoundingBox {
        Self::new(
            Vector3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            Vector3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub position: Vector3,
    pub rotation: Quaternion,
}

impl Transform {
    pub fn new(position: Vector3, rotation: Quaternion) -> Self {
        Transform { position, rotation }
    }

    pub fn identity() -> Self {
        Transform {
            position: Vector3::zero(),
            rotation: Quaternion::identity(),
        }
    }

    pub fn from_position(position: Vector3) -> Self {
        Transform {
            position,
            rotation: Quaternion::identity(),
        }
    }

    pub fn from_rotation(rotation: Quaternion) -> Self {
        Transform {
            position: Vector3::zero(),
            rotation,
        }
    }

    pub fn transform_vector(self, point: Vector3) -> Vector3 {
        self.rotation.rotate_vector(point) + self.position
    }

    pub fn inverse(self) -> Transform {
        let inv_rotation = self.rotation.inverse();
        let inv_position = inv_rotation.rotate_vector(-self.position);
        Transform {
            position: inv_position,
            rotation: inv_rotation,
        }
    }

    pub fn combine(self, other: Transform) -> Transform {
        Transform {
            position: self.transform_vector(other.position),
            rotation: self.rotation * other.rotation,
        }
    }

    pub fn lerp(self, other: Transform, t: f32) -> Transform {
        Transform {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.slerp(other.rotation, t),
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mul for Transform {
    type Output = Transform;
    fn mul(self, other: Transform) -> Transform {
        self.combine(other)
    }
}

impl MulAssign for Transform {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
