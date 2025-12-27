use std::time::Instant;

use shared::{
    math::{GeometryTree, Transform3},
    physics::Moment,
    player::PlayerData,
    tick::Ticker,
    utility::{IdReserver, SparseSet},
};

mod net;

pub struct Game {
    last_update: Instant,

    transforms: SparseSet<Transform3>,
    players: SparseSet<PlayerData>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,

    reserver: IdReserver,
    ticker: Ticker,
}

impl Game {
    pub fn update(&mut self) {
        let new_update_time = Instant::now();
        let dt = (new_update_time - self.last_update).as_secs_f32();
        self.last_update = new_update_time;

        self.ticker.update(dt, |tick, dt| {})
    }
}

fn main() {
    loop {}
}
