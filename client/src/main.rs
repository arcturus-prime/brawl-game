use raylib::{
    RaylibHandle, RaylibThread,
    camera::Camera,
    color::Color,
    models::{Mesh, Model, RaylibMesh},
    prelude::{RaylibDraw, RaylibDraw3D, RaylibMode3DExt},
};
use shared::{
    math::{VECTOR_X, VECTOR_Y, VECTOR_Z, Vector},
    physics::{Cuboid, DynamicBody, PhysicsWorld},
    player::PlayerData,
    tick::Ticker,
    utility::{EntityReserver, SparseSet},
};

use crate::math::{camera_from_position_rotation, to_raylib};

mod math;

const CAMERA_DISTANCE: f32 = 10.0;

fn create_player(
    handle: &mut RaylibHandle,
    thread: &RaylibThread,
    models: &mut SparseSet<Model>,
    players: &mut SparseSet<PlayerData>,
    world: &mut PhysicsWorld<Cuboid, Cuboid>,
    id: usize,
) {
    let mesh = Mesh::gen_mesh_cube(thread, 2.0, 2.0, 2.0);

    models.insert(
        id,
        handle
            .load_model_from_mesh(thread, unsafe { mesh.make_weak() })
            .expect("Could not load model"),
    );

    players.insert(id, PlayerData::default());
    world.dynamic_bodies.insert(
        id,
        DynamicBody::new(Cuboid::new(Vector::from_vector(2.0, 2.0, 2.0)), 5.0),
    );
}

fn main() {
    let mut reserver = EntityReserver::default();

    let local_player = reserver.reserve();

    let mut world: PhysicsWorld<Cuboid, Cuboid> = PhysicsWorld::default();
    let mut players: SparseSet<PlayerData> = SparseSet::default();
    let mut models: SparseSet<Model> = SparseSet::default();

    let mut camera: Camera =
        camera_from_position_rotation(Vector::zero_point(), Vector::identity_quaternion(), 60.0);
    let mut ticker = Ticker::default();

    let (mut rl, thread) = raylib::init().size(640, 480).title("Brawl Game").build();

    create_player(
        &mut rl,
        &thread,
        &mut models,
        &mut players,
        &mut world,
        local_player,
    );

    while !rl.window_should_close() {
        let dt = rl.get_frame_time();

        // UPDATE

        ticker.update(dt, |tick, dt| world.update(dt));

        let body = &world.dynamic_bodies[local_player];

        camera.position = to_raylib(
            body.body.position - body.body.rotation.geometric(VECTOR_X) * CAMERA_DISTANCE,
        );
        camera.target = to_raylib(body.body.position);

        // RENDER

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);

        let mut three_d = d.begin_mode3D(camera);
        for (id, body) in world.dynamic_bodies.iter() {
            three_d.draw_model(&models[*id], to_raylib(body.body.position), 1.0, Color::RED);
        }

        for (id, body) in world.static_bodies.iter() {
            three_d.draw_model(&models[*id], to_raylib(body.position), 1.0, Color::RED);
        }
    }
}
