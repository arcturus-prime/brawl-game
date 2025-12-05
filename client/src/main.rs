use raylib::{
    camera::Camera,
    color::Color,
    models::Model,
    prelude::{RaylibDraw, RaylibDraw3D, RaylibMode3DExt},
};
use shared::{
    math::{VECTOR_X, Vector},
    physics::{Cuboid, PhysicsWorld},
    player::PlayerData,
    tick::Ticker,
    utility::SparseSet,
};

use crate::math::{camera_from_position_rotation, to_raylib};

mod math;

const CAMERA_DISTANCE: f32 = 10.0;

fn main() {
    let mut local_player: Option<usize> = None;
    let mut world: PhysicsWorld<Cuboid, Cuboid> = PhysicsWorld::default();

    let mut players: SparseSet<PlayerData> = SparseSet::default();
    let mut models: SparseSet<Model> = SparseSet::default();

    let mut camera: Camera =
        camera_from_position_rotation(Vector::zero_point(), Vector::identity_quaternion(), 60.0);
    let mut ticker = Ticker::default();

    let (mut rl, thread) = raylib::init().size(640, 480).title("Brawl Game").build();

    while !rl.window_should_close() {
        let dt = rl.get_frame_time();

        // UPDATE

        ticker.update(dt, |tick, dt| world.update(dt));

        if let Some(id) = local_player {
            let body = &world.dynamic_bodies[id];

            camera.position = to_raylib(
                body.body.position - body.body.rotation.geometric(VECTOR_X) * CAMERA_DISTANCE,
            );
            camera.target = to_raylib(body.body.position);
        }

        // RENDER

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);

        let mut three_d = d.begin_mode3D(camera);

        for (id, body) in world.dynamic_bodies.iter() {
            three_d.draw_model(
                &models[*id],
                to_raylib(body.body.position),
                1.0,
                Color::WHITE,
            );
        }

        for (id, body) in world.static_bodies.iter() {
            three_d.draw_model(&models[*id], to_raylib(body.position), 1.0, Color::WHITE);
        }
    }
}
