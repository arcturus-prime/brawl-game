use brawl_game_shared::tick::Ticker;
use raylib::{
    RaylibHandle, camera::Camera, color::Color, input, math::Vector3, models::Model,
    prelude::RaylibDraw3D,
};

use crate::math::to_raylib;

pub const CAMERA_DISTANCE: f32 = 10.0;

pub struct ClientWorld {
    player_index: Option<usize>,

    camera: Camera,
}

impl ClientWorld {
    pub fn render<T: RaylibDraw3D>(&self, mut drawer: T) {
        drawer.draw_model(&self.map_model, Vector3::zero(), 1.0, Color::RED);

        for (x, i) in self.player_models.iter().zip(0..) {
            let player_position = self.world.players[i].physical.position;
            drawer.draw_model(x, to_raylib(player_position), 1.0, Color::BLUE);
        }
    }

    pub fn update(&mut self, handle: &RaylibHandle) {
        let dt = handle.get_frame_time();

        self.world.update(dt);

        if let Some(index) = self.player_index {
            let player = &mut self.world.players[index];

            self.camera.position =
                to_raylib(player.physical.position - player.forward * CAMERA_DISTANCE);
            self.camera.target = to_raylib(player.physical.position);
        }
    }
}
