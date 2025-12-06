use raylib::{
    RaylibHandle, RaylibThread,
    color::Color,
    models::{Mesh, Model, RaylibMesh, RaylibModel},
    prelude::{RaylibDraw, RaylibDraw3D, RaylibMode3DExt},
};
use shared::{
    math::{Quaternion, Transform, VECTOR_X, Vector3},
    physics::{Cuboid, Moment},
    player::PlayerData,
    tick::Ticker,
    utility::{EntityReserver, SparseSet},
};

use crate::math::{camera_from_position_rotation, quat_to_raylib, vec_to_raylib};

mod math;

struct RaylibDrawContext<'a> {
    handle: &'a mut RaylibHandle,
    thread: &'a RaylibThread,
}

impl<'a> RaylibDrawContext<'a> {
    pub fn new(handle: &'a mut RaylibHandle, thread: &'a RaylibThread) -> Self {
        Self { handle, thread }
    }
}

fn create_player(
    context: &mut RaylibDrawContext,
    transforms: &mut SparseSet<Transform>,
    models: &mut SparseSet<Model>,
    players: &mut SparseSet<PlayerData>,
    colliders: &mut SparseSet<Cuboid>,
    moment: &mut SparseSet<Moment>,
    id: usize,
) {
    let mesh = Mesh::gen_mesh_cube(context.thread, 2.0, 2.0, 2.0);

    models.insert(
        id,
        context
            .handle
            .load_model_from_mesh(context.thread, unsafe { mesh.make_weak() })
            .expect("Could not load model"),
    );
    players.insert(id, PlayerData::default());
    moment.insert(id, Moment::new(5.0));
    colliders.insert(id, Cuboid::new(2.0, 2.0, 2.0));
    transforms.insert(id, Transform::identity());
}

fn create_static_object(
    context: &mut RaylibDrawContext,
    models: &mut SparseSet<Model>,
    tranforms: &mut SparseSet<Transform>,
    colliders: &mut SparseSet<Cuboid>,
    id: usize,
) {
    let mesh = Mesh::gen_mesh_cube(context.thread, 2.0, 2.0, 6.0);

    models.insert(
        id,
        context
            .handle
            .load_model_from_mesh(context.thread, unsafe { mesh.make_weak() })
            .expect("Could not load model"),
    );
    tranforms.insert(id, Transform::identity());
    colliders.insert(id, Cuboid::new(2.0, 2.0, 6.0))
}

fn main() {
    let mut reserver = EntityReserver::default();

    let local_player = reserver.reserve();

    let mut colliders: SparseSet<Cuboid> = SparseSet::default();
    let mut players: SparseSet<PlayerData> = SparseSet::default();
    let mut models: SparseSet<Model> = SparseSet::default();
    let mut momenta: SparseSet<Moment> = SparseSet::default();
    let mut transforms: SparseSet<Transform> = SparseSet::default();

    let mut camera = camera_from_position_rotation(Vector3::zero(), Quaternion::identity(), 60.0);
    let mut ticker = Ticker::default();

    let (mut rl, thread) = raylib::init().size(640, 480).title("Brawl Game").build();

    let mut context = RaylibDrawContext::new(&mut rl, &thread);

    create_player(
        &mut context,
        &mut transforms,
        &mut models,
        &mut players,
        &mut colliders,
        &mut momenta,
        local_player,
    );

    let reference = reserver.reserve();
    create_static_object(
        &mut context,
        &mut models,
        &mut transforms,
        &mut colliders,
        reference,
    );

    transforms[reference].position = Vector3::new(10.0, 0.0, 0.0);

    let mut theta = 0.0;
    let mut azimuth = 0.0;

    while !rl.window_should_close() {
        let dt = rl.get_frame_time();
        let delta = rl.get_mouse_delta();

        theta += delta.x * 0.01;
        azimuth += delta.y * 0.01;

        transforms[local_player].rotation = Quaternion::from_euler(0.0, azimuth, theta);

        ticker.update(dt, |tick, dt| {
            for (id, moment) in momenta.iter_mut() {
                transforms[*id].position += moment.velocity * dt;
            }
        });

        let forward = transforms[local_player].rotation.rotate_vector(VECTOR_X);

        camera = camera_from_position_rotation(
            transforms[local_player].position - forward * 10.0,
            transforms[local_player].rotation,
            60.0,
        );

        // RENDER

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);

        let mut three_d = d.begin_mode3D(camera);
        for (id, model) in models.iter() {
            three_d.draw_model(
                model,
                vec_to_raylib(transforms[*id].position),
                1.0,
                Color::RED,
            );
        }
    }
}
