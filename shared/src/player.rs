use std::collections::BTreeMap;

use crate::math::Vector3;

#[derive(Default, Clone)]
pub struct InputState {
    pub want_direction: Vector3,
    pub throttle: f32,
}

#[derive(Default)]
pub struct PlayerData {
    pub health: f32,
    pub input_history: BTreeMap<u32, InputState>,
}

impl PlayerData {}
