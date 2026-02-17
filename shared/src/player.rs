use std::collections::BTreeMap;

use crate::{
    math::{Quaternion, Transform3, Vector3},
    physics::Moment,
};

const PLAYER_ROTATION_SPEED: f32 = 0.01;
const PLAYER_ACCELERATION: f32 = 0.1;
const PLAYER_MAX_SPEED: f32 = 100.0;

#[derive(Default, Clone)]
pub struct PlayerInputState {
    pub want_direction: Vector3,
    pub throttle: f32,
}

impl PlayerInputState {
    pub fn apply(&mut self, moment: &mut Moment, transform: &mut Transform3) {
        self.want_direction = self.want_direction.normalize();

        let movement_direction = transform.rotate_vector(Vector3::Z);
        let current_velocity = moment.velocity.dot(movement_direction);
        let remaining = (self.throttle * PLAYER_MAX_SPEED - current_velocity)
            .clamp(-PLAYER_MAX_SPEED, PLAYER_MAX_SPEED);

        moment.apply_impulse(movement_direction * remaining * PLAYER_ACCELERATION);

        if self.want_direction == Vector3::ZERO {
            return;
        }

        let target_rotation = Quaternion::look_at(self.want_direction, Vector3::Z);

        transform.rotation = transform
            .rotation
            .slerp(target_rotation, PLAYER_ROTATION_SPEED)
    }
}

#[derive(Default)]
pub struct PlayerData {
    pub name: String,
    pub inputs: BTreeMap<u32, PlayerInputState>,
}
