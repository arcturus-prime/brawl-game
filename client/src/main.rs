mod math;
mod render;

use std::time::Instant;

use shared::{
    math::{GeometryTree, Transform},
    physics::Moment,
    player::PlayerData,
    utility::SparseSet,
};
use vulkano::{buffer::BufferContents, padded::Padded};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, EventLoop},
};

use crate::render::{CameraData, RenderContext, Renderer};

pub struct App {
    transforms: SparseSet<Transform>,
    players: SparseSet<PlayerData>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    cameras: SparseSet<CameraData>,

    renderer: Option<Renderer>,
    context: Option<RenderContext>,

    start_time: Instant,
}

impl App {
    pub fn new() -> Self {
        Self {
            transforms: SparseSet::default(),
            players: SparseSet::default(),
            colliders: SparseSet::default(),
            momenta: SparseSet::default(),
            cameras: SparseSet::default(),

            renderer: None,
            context: None,

            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if self.renderer.is_none() {
            let renderer = match Renderer::new(event_loop) {
                Ok(r) => r,
                Err(e) => panic!("Failed to create renderer: {}", e),
            };

            let shader = match render::compute_shader::load(renderer.device()) {
                Ok(s) => s,
                Err(e) => panic!("Failed to load shader: {}", e),
            };

            let context = match renderer.create_context(event_loop, shader) {
                Ok(c) => c,
                Err(e) => panic!("Failed to create context: {}", e),
            };

            self.renderer = Some(renderer);
            self.context = Some(context);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(ctx) = &mut self.context {
                    ctx.recreate_swapchain()
                        .expect("Could not recreate swapchain");
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(context) = &mut self.context {
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let window_size = context.window_size();

                    let input = render::compute_shader::InputData {
                        time: elapsed,
                        resolution_x: window_size.0,
                        resolution_y: Padded::from(window_size.1),
                        camera_position: Padded::from([0.0, 0.0, 0.0]),
                        camera_rotation: [0.0, 0.0, 0.0, 1.0],
                    };

                    if let Err(e) = context.render(input) {
                        eprintln!("Render error: {}", e);
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();

    event_loop.run_app(&mut app).unwrap();
}
