use std::collections::BTreeMap;

use crate::{math::Vector, utility::SparseSet};

#[derive(Default, Clone)]
pub struct InputState {
    pub want_direction: Vector,
    pub throttle: f32,
}

#[derive(Default)]
pub struct PlayerData {
    pub health: f32,
    pub input_history: BTreeMap<u32, InputState>,
}
