use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vector {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vector { x, y, z, w }
    }

    pub fn from_position(x: f32, y: f32, z: f32) -> Self {
        Vector { x, y, z, w: 0.0 }
    }

    pub fn zero() -> Self {
        Vector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    pub fn one() -> Self {
        Vector {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: 1.0,
        }
    }

    pub fn identity() -> Self {
        Vector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    pub fn dot(&self, other: &Vector) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    pub fn normalize(&self) -> Vector {
        let len = self.length();
        if len > 0.0 {
            Vector {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
                w: self.w / len,
            }
        } else {
            *self
        }
    }

    pub fn from_axis_angle(axis: Vector, angle_radians: f32) -> Self {
        let half_angle = angle_radians / 2.0;
        let sin_half = half_angle.sin();
        let axis_normalized = axis.normalize();

        Vector {
            x: axis_normalized.x * sin_half,
            y: axis_normalized.y * sin_half,
            z: axis_normalized.z * sin_half,
            w: half_angle.cos(),
        }
    }

    pub fn from_euler(pitch: f32, yaw: f32, roll: f32) -> Self {
        let cy = (yaw * 0.5).cos();
        let sy = (yaw * 0.5).sin();
        let cp = (pitch * 0.5).cos();
        let sp = (pitch * 0.5).sin();
        let cr = (roll * 0.5).cos();
        let sr = (roll * 0.5).sin();

        Vector {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }

    pub fn hadamard(&self, other: &Vector) -> Vector {
        Vector {
            w: self.w * other.w,
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    pub fn geometric(&self, other: &Vector) -> Vector {
        Vector {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    pub fn conjugate(&self) -> Vector {
        Vector {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    pub fn inverse(&self) -> Vector {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let conj = self.conjugate();
            Vector {
                x: conj.x / len_sq,
                y: conj.y / len_sq,
                z: conj.z / len_sq,
                w: conj.w / len_sq,
            }
        } else {
            *self
        }
    }

    pub fn slerp(&self, other: &Vector, t: f32) -> Vector {
        let mut dot = self.dot(other);

        let mut other_adjusted = *other;
        if dot < 0.0 {
            other_adjusted = -other_adjusted;
            dot = -dot;
        }

        let dot = dot.clamp(-1.0, 1.0);

        if dot > 0.9995 {
            // Use linear interpolation for very close quaternions
            return (*self + (other_adjusted - *self) * t).normalize();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();

        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        *self * a + other_adjusted * b
    }

    // Returns (pitch, yaw, roll)
    pub fn to_euler(&self) -> (f32, f32, f32) {
        let sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z);
        let cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.z + self.x * self.y);
        let cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (pitch, yaw, roll)
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
