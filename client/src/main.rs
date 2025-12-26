use std::time::Instant;

use shared::{
    math::{GeometryTree, Transform3, Vector3},
    physics::{Moment, step_world},
    player::{PlayerData, PlayerInputState},
    tick::Ticker,
    utility::{IdReserver, SparseSet},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, ControlFlow, EventLoop},
    keyboard::KeyCode,
};
use winit_input_helper::WinitInputHelper;

use crate::render::{CameraData, CameraInput, CameraMode, Renderable, Renderer};

mod net;
mod render;

pub struct Game {
    last_update: Instant,

    transforms: SparseSet<Transform3>,
    players: SparseSet<PlayerData>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    cameras: SparseSet<CameraData>,
    renderable: SparseSet<Renderable>,

    reserver: IdReserver,
    ticker: Ticker,

    camera_id: usize,
    local_player_id: usize,

    pub input: WinitInputHelper,
    pub renderer: Renderer,
}

pub struct App {
    game: Option<Game>,
}

impl App {
    pub fn new() -> Self {
        Self { game: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if self.game.is_none() {
            let renderer = Renderer::new(event_loop).unwrap();
            self.game = Some(Game::new(renderer));
        }
    }

    fn device_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let Some(game) = &mut self.game else { return };
        game.input.process_device_event(&event);
    }

    fn about_to_wait(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        let Some(game) = &mut self.game else { return };
        game.input.end_step();
    }

    fn new_events(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        let Some(game) = &mut self.game else { return };
        game.input.step();
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(game) = &mut self.game else { return };

        game.input.process_window_event(&event);
        game.update();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                game.renderer.set_recreate_swapchain();
            }
            WindowEvent::RedrawRequested => {
                game.render();
            }
            _ => {}
        }
    }
}

impl Game {
    pub fn new(renderer: Renderer) -> Self {
        let mut reserver = IdReserver::default();
        let camera_id = reserver.reserve();

        let mut cameras = SparseSet::default();
        cameras.insert(camera_id, CameraData::default());

        let mut transforms = SparseSet::default();
        transforms.insert(camera_id, Transform3::identity());

        let mut s = Self {
            last_update: Instant::now(),
            reserver,
            camera_id,
            local_player_id: usize::MAX,
            cameras,
            transforms,
            renderable: SparseSet::default(),
            colliders: SparseSet::default(),
            players: SparseSet::default(),
            momenta: SparseSet::default(),
            renderer,
            ticker: Ticker::default(),
            input: WinitInputHelper::new(),
        };

        s.temp_create_local_player();
        s.temp_add_object_static();
        s.temp_add_object_dynamic();

        s
    }

    fn temp_create_local_player(&mut self) {
        let id = self.reserver.reserve();

        self.transforms.insert(id, Transform3::identity());
        self.momenta.insert(id, Moment::new(5.0));
        self.players.insert(id, PlayerData::default());

        let mut collider = GeometryTree::from_cube(2.0, 2.0, 2.0, 1);
        let mut hole = GeometryTree::from_cube(1.0, 1.0, 3.0, 1);
        hole.invert();
        collider.intersection(hole);

        let mut renderable = self.renderer.create_renderable().unwrap();
        renderable.set_nodes(&collider).unwrap();

        self.renderable.insert(id, renderable);
        self.colliders.insert(id, collider);

        self.cameras[self.camera_id].mode = CameraMode::Orbit {
            theta: 0.0,
            azimuth: 0.0,
            distance: 10.0,
            target: id,
        };

        self.local_player_id = id;
    }

    fn temp_add_object_static(&mut self) {
        let id = self.reserver.reserve();

        let mut collider = GeometryTree::from_cube(10.0, 10.0, 10.0, 1);
        // let mut hole = GeometryTree::from_cube(17.5, 5.0, 5.0, 1);
        // hole.transform(Transform3::from_position(Vector3::Y * 5.0));
        // hole.invert();
        // collider.intersection(hole);

        // let mut hole = GeometryTree::from_cube(17.5, 5.0, 5.0, 1);
        // hole.transform(Transform3::from_position(Vector3::Y * -4.0));
        // hole.invert();
        // collider.intersection(hole);

        let mut renderable = self.renderer.create_renderable().unwrap();
        renderable.set_nodes(&collider).unwrap();

        self.colliders.insert(id, collider);
        self.renderable.insert(id, renderable);
        self.transforms
            .insert(id, Transform3::from_position(Vector3::X * 20.0));
    }

    fn temp_add_object_dynamic(&mut self) {
        let id = self.reserver.reserve();

        let mut collider = GeometryTree::from_cube(5.0, 5.0, 5.0, 1);
        let mut hole = GeometryTree::from_cube(10.0, 4.0, 4.0, 1);
        hole.transform(Transform3::from_position(Vector3::Y * 3.0));
        hole.invert();
        collider.intersection(hole);

        let mut renderable = self.renderer.create_renderable().unwrap();
        renderable.set_nodes(&collider).unwrap();

        self.momenta.insert(id, Moment::new(5.0));
        self.colliders.insert(id, collider);
        self.renderable.insert(id, renderable);
        self.transforms
            .insert(id, Transform3::from_position(Vector3::X * -10.0));
    }

    pub fn render(&mut self) {
        self.renderer
            .draw_scene(
                self.transforms[self.camera_id],
                self.cameras[self.camera_id].fov_y,
                &self.renderable,
                &self.transforms,
            )
            .unwrap();
    }

    pub fn update(&mut self) {
        let new_update_time = Instant::now();
        let dt = (new_update_time - self.last_update).as_secs_f32();
        self.last_update = new_update_time;

        let dampening = 0.01;

        self.cameras[self.camera_id].handle_input(CameraInput {
            delta_x: self.input.mouse_diff().0 * dampening,
            delta_y: self.input.mouse_diff().1 * dampening,
            delta_scroll: self.input.scroll_diff().1,
        });

        self.cameras[self.camera_id].update_tranform(&mut self.transforms, self.camera_id);

        self.ticker.update(dt, |tick, dt| {
            if let Some(player) = self.players.get_mut(self.local_player_id) {
                let camera_tranform = self.transforms[self.camera_id];
                let mut move_direction = Vector3::zero();

                if self.input.key_held(KeyCode::KeyW) {
                    move_direction += Vector3::X;
                }

                if self.input.key_held(KeyCode::KeyS) {
                    move_direction -= Vector3::X;
                }

                if self.input.key_held(KeyCode::KeyA) {
                    move_direction -= Vector3::Y;
                }

                if self.input.key_held(KeyCode::KeyD) {
                    move_direction += Vector3::Y;
                }

                player.set_input(
                    tick,
                    PlayerInputState {
                        want_direction: camera_tranform.rotate_vector(move_direction),
                        look_direction: camera_tranform.rotate_vector(Vector3::X),
                        throttle: 1.0,
                    },
                );

                player
                    .apply_input(
                        tick,
                        &mut self.momenta[self.local_player_id],
                        &mut self.transforms[self.local_player_id],
                    )
                    .unwrap();
            }

            step_world(&self.colliders, &mut self.momenta, &mut self.transforms, dt);
        });
    }
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).unwrap();
}
