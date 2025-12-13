use std::f32::consts::PI;

use raylib::{
    color::Color,
    models::{Mesh, Model, RaylibMesh, RaylibModel},
    prelude::{RaylibDraw, RaylibDraw3D, RaylibMode3DExt},
};
use shared::{
    math::{GeometryTree, Quaternion, Transform, Vector3},
    physics::{Moment, get_collisions, step_world},
    player::{InputState, PlayerData},
    tick::Ticker,
    utility::{EntityReserver, SparseSet},
};

use crate::{
    math::vec_to_raylib,
    render::{CameraData, CameraInput, CameraMode, RaylibContext},
};

mod math;
mod render;

pub struct World {
    transforms: SparseSet<Transform>,
    models: SparseSet<Model>,
    players: SparseSet<PlayerData>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    cameras: SparseSet<CameraData>,
}

impl World {
    pub fn new() -> Self {
        Self {
            transforms: SparseSet::default(),
            models: SparseSet::default(),
            players: SparseSet::default(),
            colliders: SparseSet::default(),
            momenta: SparseSet::default(),
            cameras: SparseSet::default(),
        }
    }
}

fn create_player(id: usize, world: &mut World, context: &mut RaylibContext) -> usize {
    let mesh = Mesh::gen_mesh_cube(&context.thread, 2.0, 2.0, 2.0);

    world.models.insert(
        id,
        context
            .handle
            .load_model_from_mesh(&context.thread, unsafe { mesh.make_weak() })
            .expect("Could not load model"),
    );
    world.players.insert(id, PlayerData::default());
    world.momenta.insert(id, Moment::new(1.0));
    world
        .colliders
        .insert(id, GeometryTree::from_cube(2.0, 2.0, 2.0));
    world.transforms.insert(id, Transform::identity());

    id
}

fn create_static_object(id: usize, world: &mut World, context: &mut RaylibContext) -> usize {
    let mesh = Mesh::gen_mesh_cube(&context.thread, 5.0, 5.0, 6.0);

    world.models.insert(
        id,
        context
            .handle
            .load_model_from_mesh(&context.thread, unsafe { mesh.make_weak() })
            .expect("Could not load model"),
    );
    world.transforms.insert(id, Transform::identity());
    world
        .colliders
        .insert(id, GeometryTree::from_cube(5.0, 5.0, 6.0));

    id
}

fn create_dynamic_object(id: usize, world: &mut World, context: &mut RaylibContext) -> usize {
    let mesh = Mesh::gen_mesh_cube(&context.thread, 5.0, 5.0, 5.0);

    world.models.insert(
        id,
        context
            .handle
            .load_model_from_mesh(&context.thread, unsafe { mesh.make_weak() })
            .expect("Could not load model"),
    );
    world.transforms.insert(id, Transform::identity());
    world
        .colliders
        .insert(id, GeometryTree::from_cube(5.0, 5.0, 5.0));
    world.momenta.insert(id, Moment::new(50.0));

    id
}

fn create_orbit_camera(id: usize, target: usize, world: &mut World) {
    world.cameras.insert(id, CameraData::new());
    world.cameras[id].switch_mode(CameraMode::Orbit {
        theta: 0.0,
        azimuth: 0.0,
        distance: 10.0,
        target,
    });
    world.transforms.insert(id, Transform::identity());
}

fn get_current_input_state(context: &RaylibContext, camera_transform: &Transform) -> InputState {
    let mut direction = Vector3::zero();

    if context.handle.is_key_down(raylib::ffi::KeyboardKey::KEY_W) {
        direction += Vector3::X;
    }

    if context.handle.is_key_down(raylib::ffi::KeyboardKey::KEY_S) {
        direction -= Vector3::X;
    }

    if context.handle.is_key_down(raylib::ffi::KeyboardKey::KEY_A) {
        direction += Vector3::Y;
    }

    if context.handle.is_key_down(raylib::ffi::KeyboardKey::KEY_D) {
        direction -= Vector3::Y;
    }

    InputState {
        look_direction: camera_transform.rotation.rotate_vector(Vector3::X),
        want_direction: camera_transform
            .rotation
            .rotate_vector(direction.normalize()),
        throttle: 1.0,
    }
}

fn main() {
    let (rl, thread) = raylib::init().size(1280, 720).title("Brawl Game").build();

    let mut context = RaylibContext { handle: rl, thread };
    let mut world = World::new();
    let mut entity = EntityReserver::default();
    let mut ticker = Ticker::default();

    let local_player = entity.reserve();
    create_player(local_player, &mut world, &mut context);

    let object1 = entity.reserve();
    create_static_object(object1, &mut world, &mut context);

    let object2 = entity.reserve();
    create_dynamic_object(object2, &mut world, &mut context);

    let camera = entity.reserve();
    create_orbit_camera(camera, local_player, &mut world);

    world.transforms[object1].position = Vector3::new(10.0, 0.0, 0.0);
    world.transforms[object2].position = Vector3::new(5.0, -5.0, 5.0);

    while !context.handle.window_should_close() {
        let dt = context.handle.get_frame_time();
        let mouse_delta = context.handle.get_mouse_delta();

        world.cameras[camera].handle_input(CameraInput {
            delta_x: mouse_delta.x * 0.01,
            delta_y: mouse_delta.y * 0.01,
            delta_scroll: 0.0,
        });

        ticker.update(dt, |tick, dt| {
            let input = get_current_input_state(&context, &world.transforms[camera]);

            world.players[local_player].set_input(tick, input);
            world.players[local_player]
                .apply_input(
                    tick,
                    &mut world.momenta[local_player],
                    &mut world.transforms[local_player],
                )
                .expect("Somehow the input of the current tick was not set");

            let (colliding, non_colliding) =
                get_collisions(&world.momenta, &world.transforms, &world.colliders, dt);

            step_world(
                colliding,
                non_colliding,
                &mut world.momenta,
                &mut world.transforms,
                dt,
            );
        });

        world.cameras[camera].update_tranform(&mut world.transforms, camera);

        // RENDER

        let mut draw = context.handle.begin_drawing(&context.thread);
        draw.clear_background(Color::BLACK);

        let mut draw_3d =
            draw.begin_mode3D(world.cameras[camera].to_raylib(&world.transforms[camera]));

        for (id, model) in world.models.iter() {
            let pair = world.transforms[*id].rotation.to_axis_angle();

            let position = vec_to_raylib(world.transforms[*id].position);
            let rotation_axis = vec_to_raylib(pair.0);
            let rotation_angle = pair.1 / (2.0 * PI) * 360.0;
            let scale = vec_to_raylib(Vector3::new(1.0, 1.0, 1.0));

            draw_3d.draw_model_ex(
                model,
                position,
                rotation_axis,
                rotation_angle,
                scale,
                Color::RED,
            );
        }
    }
}
