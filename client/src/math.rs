use shared::math::{Quaternion, Transform, Vector3};

pub fn vec_to_raylib(vector: Vector3) -> raylib::math::Vector3 {
    raylib::math::Vector3::new(vector.x, vector.y, vector.z)
}

pub fn quat_to_raylib(quaternion: Quaternion) -> raylib::math::Quaternion {
    raylib::math::Quaternion::new(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
}

pub fn transform_to_raylib(transform: Transform) -> raylib::math::Matrix {
    let (axis, angle) = transform.rotation.to_axis_angle();

    raylib::math::Matrix::translate(
        transform.position.x,
        transform.position.y,
        transform.position.z,
    ) + raylib::math::Matrix::rotate(vec_to_raylib(axis), angle)
}
