use std::f32::consts::PI;

use shared::{
    math::{GeometryTree, Quaternion, Transform, Vector3},
    physics::Moment,
    player::PlayerData,
    utility::{IdReserver, SparseSet},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, EventLoop},
    keyboard::PhysicalKey,
};

use crate::render::{CameraData, CameraInput, Renderable, Renderer};

mod render;

pub struct App {
    transforms: SparseSet<Transform>,
    players: SparseSet<PlayerData>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    cameras: SparseSet<CameraData>,
    renderable: SparseSet<Renderable>,

    renderer: Option<Renderer>,
    camera_id: usize,
    object_a: usize,
    reserver: IdReserver,
}

impl App {
    pub fn new() -> Self {
        let mut reserver = IdReserver::default();
        let camera_id = reserver.reserve();
        let object_a = reserver.reserve();

        let mut cameras = SparseSet::default();
        cameras.insert(
            camera_id,
            CameraData {
                mode: render::CameraMode::Orbit {
                    theta: 0.0,
                    azimuth: 0.0,
                    distance: 10.0,
                    target: object_a,
                },
                fov_y: 60.0,
            },
        );

        let mut transforms = SparseSet::default();
        transforms.insert(camera_id, Transform::identity());
        transforms.insert(
            object_a,
            Transform::from_position(Vector3::new(0.0, 0.0, -20.0)),
        );

        let mut colliders = SparseSet::default();
        colliders.insert(object_a, GeometryTree::from_cube(5.0, 5.0, 5.0, 0));

        let renderable = SparseSet::default();
        let players = SparseSet::default();

        Self {
            momenta: SparseSet::default(),
            renderable,

            cameras,
            transforms,
            colliders,
            players,

            renderer: None,

            object_a,
            camera_id,
            reserver,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if self.renderer.is_none() {
            let mut renderer = match Renderer::new(event_loop) {
                Ok(r) => r,
                Err(e) => panic!("Failed to create renderer: {}", e),
            };

            let mut renderable = renderer.create_renderable().unwrap();
            renderable
                .set_nodes(&self.colliders[self.object_a])
                .unwrap();

            self.renderable.insert(self.object_a, renderable);

            self.renderer = Some(renderer);
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
                if let Some(ctx) = &mut self.renderer {
                    ctx.recreate_swapchain()
                        .expect("Could not recreate swapchain");
                }
            }
            WindowEvent::RedrawRequested => {
                self.cameras[self.camera_id].update_tranform(&mut self.transforms, self.camera_id);

                if let Some(context) = &mut self.renderer {
                    if let Err(e) = context.draw_scene(
                        self.transforms[self.camera_id],
                        60.0,
                        &self.renderable,
                        &self.transforms,
                    ) {
                        eprintln!("Render error: {}", e);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {
                if event.physical_key == PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown) {
                    self.cameras[self.camera_id].handle_input(CameraInput {
                        delta_x: 0.001,
                        delta_y: 0.0,
                        delta_scroll: 0.0,
                    });
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
