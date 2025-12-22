use std::{
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::Instant,
};

use shared::{
    math::{GeometryTree, Transform},
    physics::{Moment, step_world},
    player::PlayerData,
    tick::Ticker,
    utility::{IdReserver, SparseSet},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, ControlFlow, EventLoop},
};

use crate::{
    input::InputStream,
    render::{CameraData, Renderable, Renderer},
};

mod input;
mod render;

pub struct Game {
    last_update: Instant,

    transforms: SparseSet<Transform>,
    players: SparseSet<PlayerData>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    cameras: SparseSet<CameraData>,
    renderable: SparseSet<Renderable>,

    reserver: IdReserver,
    ticker: Ticker,

    camera_id: usize,

    pub input_stream: InputStream,
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

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(game) = &mut self.game else { return };
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
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {}
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {}
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
        transforms.insert(camera_id, Transform::identity());

        Self {
            last_update: Instant::now(),
            reserver,
            camera_id,
            cameras,
            transforms,
            renderable: SparseSet::default(),
            colliders: SparseSet::default(),
            players: SparseSet::default(),
            momenta: SparseSet::default(),
            renderer,
            ticker: Ticker::default(),
            input_stream: InputStream::default(),
        }
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

        self.ticker.update(dt, |tick, dt| {
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
