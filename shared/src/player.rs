use std::collections::BTreeMap;

use crate::{
    math::{Quaternion, Transform3, Vector3},
    physics::Moment,
};

const PLAYER_ACCELERATION: f32 = 0.1;
const PLAYER_MAX_SPEED: f32 = 100.0;

#[derive(Default, Clone)]
pub struct PlayerInputState {
    pub want_direction: Vector3,
    pub look_direction: Vector3,
    pub throttle: f32,
}

impl PlayerInputState {
    pub fn apply(&mut self, moment: &mut Moment, transform: &mut Transform3) {
        // note(arcprime): should always be normalized, but just in case
        self.want_direction = self.want_direction.normalize();

        let current_velocity = moment.velocity.dot(self.want_direction);
        let remaining =
            (self.throttle * PLAYER_MAX_SPEED - current_velocity).clamp(0.0, PLAYER_MAX_SPEED);

        moment.apply_impulse(self.want_direction * remaining * PLAYER_ACCELERATION);
        transform.rotation = Quaternion::look_at(self.look_direction, Vector3::Z);
    }
}

#[derive(Default)]
pub struct PlayerData {
    pub name: String,
    pub inputs: BTreeMap<u32, PlayerInputState>,
}
