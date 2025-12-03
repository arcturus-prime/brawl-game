use crate::{
    math::Vector,
    physics::{ConvexHull, DynamicBody},
};

#[derive(Default, Clone)]
pub struct InputState {
    pub want_direction: Vector,
    pub throttle: f32,
}

pub struct Player {
    pub health: f32,
    pub body: DynamicBody<ConvexHull>,
}
