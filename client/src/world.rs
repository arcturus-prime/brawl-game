use raylib::{
    RaylibHandle,
    camera::Camera,
    color::Color,
    models::Model,
    prelude::{RaylibDraw3D, RaylibDrawHandle, RaylibMode3DExt},
};
use shared::{
    math::{VECTOR_X, Vector},
    physics::{Cuboid, PhysicsWorld},
    player::PlayerData,
    tick::Ticker,
    utility::SparseSet,
};

use crate::math::{camera_from_position_rotation, to_raylib};

pub const CAMERA_DISTANCE: f32 = 10.0;

pub struct ClientWorld {
    local_player: Option<usize>,

    world: PhysicsWorld<Cuboid, Cuboid>,

    players: SparseSet<PlayerData>,
    models: SparseSet<Model>,

    camera: Camera,

    ticker: Ticker,
}

impl ClientWorld {
    pub fn new() -> Self {
        Self {
            local_player: None,
            players: SparseSet::default(),
            world: PhysicsWorld::default(),
            models: SparseSet::default(),
            camera: camera_from_position_rotation(
                Vector::zero_point(),
                Vector::identity_quaternion(),
                60.0,
            ),
            ticker: Ticker::default(),
        }
    }

    pub fn render(&self, mut drawer: RaylibDrawHandle) {
        let mut three_d = drawer.begin_mode3D(self.camera);

        for (id, body) in self.world.dynamic_bodies.iter() {
            three_d.draw_model(
                &self.models[*id],
                to_raylib(body.body.position),
                1.0,
                Color::WHITE,
            );
        }

        for (id, body) in self.world.static_bodies.iter() {
            three_d.draw_model(
                &self.models[*id],
                to_raylib(body.position),
                1.0,
                Color::WHITE,
            );
        }
    }

    pub fn update(&mut self, handle: &RaylibHandle) {
        let dt = handle.get_frame_time();

        self.world.update(dt);

        if let Some(id) = self.local_player {
            let body = &self.world.dynamic_bodies[id];

            self.camera.position = to_raylib(
                body.body.position - body.body.rotation.geometric(VECTOR_X) * CAMERA_DISTANCE,
            );
            self.camera.target = to_raylib(body.body.position);
        }
    }
}
