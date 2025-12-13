use std::collections::BTreeMap;

use crate::{
    math::{Quaternion, Transform, Vector3},
    physics::Moment,
};

const PLAYER_ACCELERATION: f32 = 0.1;
const PLAYER_MAX_SPEED: f32 = 100.0;

#[derive(Default, Clone)]
pub struct InputState {
    pub want_direction: Vector3,
    pub look_direction: Vector3,
    pub throttle: f32,
}

#[derive(Default)]
pub struct PlayerData {
    pub health: f32,
    pub input_history: BTreeMap<u32, InputState>,
}

impl PlayerData {
    pub fn apply_input(
        &mut self,
        tick: u32,
        moment: &mut Moment,
        transform: &mut Transform,
    ) -> Result<(), ()> {
        let inputs = self.input_history.get_mut(&tick).ok_or(())?;

        // note(arcprime): should always be normalized, but just in case
        inputs.want_direction = inputs.want_direction.normalize();

        let current_velocity = moment.velocity.dot(inputs.want_direction);
        let remaining =
            (inputs.throttle * PLAYER_MAX_SPEED - current_velocity).clamp(0.0, PLAYER_MAX_SPEED);

        moment.apply_impulse(inputs.want_direction * remaining * PLAYER_ACCELERATION);
        transform.rotation = Quaternion::look_at(inputs.look_direction, Vector3::Z);

        Ok(())
    }

    pub fn clear_input(&mut self, tick: u32) {
        self.input_history.remove(&tick);
    }

    pub fn set_input(&mut self, tick: u32, input: InputState) {
        self.input_history.insert(tick, input);
    }
}
