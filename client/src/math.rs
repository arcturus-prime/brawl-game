use raylib::camera::Camera;
use shared::math::{Quaternion, VECTOR_X, VECTOR_Z, Vector3};

pub fn vec_to_raylib(vector: Vector3) -> raylib::math::Vector3 {
    raylib::math::Vector3::new(vector.x, vector.y, vector.z)
}

pub fn quat_to_raylib(quaternion: Quaternion) -> raylib::math::Quaternion {
    raylib::math::Quaternion::new(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
}

pub fn camera_from_position_rotation(
    position: Vector3,
    rotation: Quaternion,
    fov_y: f32,
) -> Camera {
    Camera::perspective(
        vec_to_raylib(position),
        vec_to_raylib(rotation.rotate_vector(VECTOR_X)),
        vec_to_raylib(rotation.rotate_vector(VECTOR_Z)),
        fov_y,
    )
}
