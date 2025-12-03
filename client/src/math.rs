use raylib::math::Vector3;
use shared::math::Vector;

pub fn to_raylib(vector: Vector) -> Vector3 {
    Vector3::new(vector.x, vector.y, vector.z)
}
