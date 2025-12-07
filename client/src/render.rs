use raylib::{RaylibHandle, RaylibThread, camera::Camera};
use shared::{
    math::{Quaternion, Transform, VECTOR_X, VECTOR_Z, Vector3},
    utility::{EntityReserver, SparseSet},
};

use crate::math::vec_to_raylib;

pub struct RaylibContext {
    pub handle: RaylibHandle,
    pub thread: RaylibThread,
}

impl RaylibContext {
    pub fn new(handle: RaylibHandle, thread: RaylibThread) -> Self {
        Self { handle, thread }
    }
}

pub enum CameraMode {
    Orbit {
        theta: f32,
        azimuth: f32,
        distance: f32,
        target: usize,
    },
    Fixed,
}

pub struct CameraData {
    pub mode: CameraMode,
    pub fov_y: f32,
}

pub struct CameraInput {
    pub delta_x: f32,
    pub delta_y: f32,
    pub delta_scroll: f32,
}

impl CameraData {
    pub fn new() -> Self {
        Self {
            mode: CameraMode::Fixed,
            fov_y: 60.0,
        }
    }

    pub fn switch_mode(&mut self, mode: CameraMode) {
        self.mode = mode
    }

    pub fn handle_input(&mut self, input: CameraInput) {
        match &mut self.mode {
            CameraMode::Orbit {
                theta,
                azimuth,
                distance,
                target,
            } => {
                *theta += input.delta_x;
                *azimuth += input.delta_y;
                *distance += input.delta_scroll;
            }
            CameraMode::Fixed => {}
        }
    }

    pub fn update_tranform(&self, transforms: &mut SparseSet<Transform>, id: usize) {
        match self.mode {
            CameraMode::Orbit {
                theta,
                azimuth,
                distance,
                target,
            } => {
                let rotation = Quaternion::from_euler(0.0, azimuth, theta);

                transforms[id].position =
                    transforms[target].position - rotation.rotate_vector(VECTOR_X) * distance;
                transforms[id].rotation = rotation
            }
            CameraMode::Fixed => {}
        }
    }

    pub fn to_raylib(&self, transform: &Transform) -> Camera {
        let position = transform.position;
        let rotation = transform.rotation;

        Camera::perspective(
            vec_to_raylib(position),
            vec_to_raylib(rotation.rotate_vector(VECTOR_X) + position),
            vec_to_raylib(rotation.rotate_vector(VECTOR_Z)),
            self.fov_y,
        )
    }
}
