use raylib::{camera::Camera, math::Vector3};
use shared::math::{VECTOR_X, VECTOR_Z, Vector};

pub fn to_raylib(vector: Vector) -> Vector3 {
    Vector3::new(vector.x, vector.y, vector.z)
}

pub fn camera_from_position_rotation(position: Vector, rotation: Vector, fov_y: f32) -> Camera {
    Camera::perspective(
        to_raylib(position),
        to_raylib(rotation.geometric(VECTOR_X)),
        to_raylib(rotation.geometric(VECTOR_Z)),
        fov_y,
    )
}
