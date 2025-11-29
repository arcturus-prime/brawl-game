use crate::{
    math::Vector,
    physics::{Collider, Moving, Sphere},
};

pub const PLAYER_MAX_SPEED: f32 = 1.0;
pub const PLAYER_DRAG: f32 = 0.01;
pub const PLAYER_ACCELERATION: f32 = 2.0;

pub struct Input {
    forward: (u8, u8),
    throttle: u8,
}

pub struct Player {
    pub moving: Moving,
    pub collider: Sphere,

    pub forward: Vector,
    pub throttle: f32,
}

impl Player {
    pub fn update(&mut self, dt: f32, collision: &dyn Collider) {
        let speed = self.moving.velocity.dot(self.forward);
        let remaining = (PLAYER_MAX_SPEED - speed).clamp(0.0, PLAYER_MAX_SPEED);
        let impulse = self.forward * (dt * PLAYER_ACCELERATION * self.throttle);

        self.moving.apply_impulse(remaining * impulse);
        self.moving
            .apply_impulse(-self.moving.velocity * PLAYER_DRAG);
    }

    pub fn apply_input(&mut self, input: Input) {}
}
