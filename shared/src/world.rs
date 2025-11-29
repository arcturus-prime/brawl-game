use crate::{physics::CollisionWorld, player::Player};

pub struct World {
    pub players: Vec<Player>,
    pub map: CollisionWorld,

    tick: u32,

    warp: f32,
    step_size: f32,

    accumulator: f32,
}

impl World {
    pub fn run(&mut self, dt: f32) {
        self.accumulator += dt;

        while self.accumulator > self.step_size + self.warp {
            self.accumulator -= self.step_size + self.warp;
            self.tick += 1;

            for x in &mut self.players {
                x.update(dt, &self.map)
            }
        }
    }

    pub fn set_tick(&mut self, tick: u32) {
        self.tick = tick;
    }

    pub fn set_warp(&mut self, warp: f32) {
        self.warp = -warp;
    }
}
