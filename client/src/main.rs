use std::{collections::BTreeMap, net::SocketAddr, str::FromStr, time::Instant};

use shared::{
    math::{GeometryTree, Transform3, Vector3},
    net::{Network, Packet},
    physics::{Moment, step_world},
    player::{PlayerData, PlayerInputState},
    tick::Ticker,
    utility::{IdReserver, SingletonSet, SparseSet},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, ControlFlow, EventLoop},
    keyboard::KeyCode,
};
use winit_input_helper::WinitInputHelper;

use crate::render::{CameraData, CameraInput, Renderable, Renderer};

mod render;

pub struct Game {
    pub input: WinitInputHelper,
    pub renderer: Renderer,

    reserver: IdReserver,
    network: Network,
    last_update: Instant,
    ticker: Ticker,

    transforms: SparseSet<Transform3>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    renderable: SparseSet<Renderable>,
    camera: SingletonSet<CameraData>,

    players: SparseSet<PlayerData>,
    inputs: SingletonSet<BTreeMap<u32, PlayerInputState>>,
}

pub struct App {
    server_address: SocketAddr,
    game: Option<Game>,
}

impl App {
    pub fn new() -> Self {
        Self {
            game: None,
            server_address: SocketAddr::from_str("0.0.0.0:0").unwrap(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if self.game.is_none() {
            let renderer = Renderer::new(event_loop).unwrap();
            let network = Network::new(SocketAddr::from_str("0.0.0.0:0").unwrap()).unwrap();

            let mut game = Game::new(renderer, network);
            let id = game
                .network
                .add_client(self.server_address, &mut game.reserver);

            game.network.send(id, Packet::ClientHello).unwrap();

            self.game = Some(game);
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
    pub fn new(renderer: Renderer, network: Network) -> Self {
        Self {
            network,
            last_update: Instant::now(),
            reserver: IdReserver::default(),
            camera: SingletonSet::default(),
            transforms: SparseSet::default(),
            renderable: SparseSet::default(),
            colliders: SparseSet::default(),
            players: SparseSet::default(),
            inputs: SingletonSet::default(),
            momenta: SparseSet::default(),
            renderer,
            ticker: Ticker::default(),
            input: WinitInputHelper::new(),
        }
    }

    pub fn render(&mut self) {
        if let Some((id, camera)) = self.camera.obtain() {
            self.renderer
                .draw_scene(
                    self.transforms[*id],
                    camera.fov_y,
                    &self.renderable,
                    &self.transforms,
                )
                .unwrap();
        }
    }

    pub fn update(&mut self) {
        let new_update_time = Instant::now();
        let dt = (new_update_time - self.last_update).as_secs_f32();
        self.last_update = new_update_time;

        while let Ok((_, packet)) = self.network.receive(&mut self.reserver) {
            match packet {
                shared::net::Packet::ClientHello => {
                    eprintln!("Got unexpected ClientHello from server");
                }
                shared::net::Packet::PlayerJoin { id, is_you } => {
                    let id = self.network.reserve_real_entity(&mut self.reserver, id);

                    self.transforms.insert(id, Transform3::identity());
                    self.momenta.insert(id, Moment::new(5.0));
                    self.players.insert(id, PlayerData::default());

                    let tree = GeometryTree::from_cube(1.0, 1.0, 1.0, 0);
                    let mut renderable = self.renderer.create_renderable().unwrap();
                    renderable.set_nodes(&tree).unwrap();

                    self.renderable.insert(id, renderable);
                    self.colliders.insert(id, tree);

                    if is_you {
                        self.inputs.insert(id, BTreeMap::new());

                        let camera_id = self.reserver.reserve();

                        self.transforms.insert(camera_id, Transform3::identity());
                        self.camera.insert(camera_id, CameraData::orbit(id));
                    }
                }
                shared::net::Packet::PlayerLeave { id } => todo!(),
                shared::net::Packet::PlayerInput { input } => todo!(),
                shared::net::Packet::PlayerMovement {
                    transform,
                    velocity,
                    id,
                } => todo!(),
            }
        }

        self.ticker.update(dt, |tick, dt| {
            if let Some((inputs_id, inputs)) = self.inputs.obtain_mut()
                && let Some((camera_id, camera)) = self.camera.obtain()
            {
                let camera_tranform = self.transforms[*camera_id];
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

                let mut input = PlayerInputState {
                    want_direction: camera_tranform.rotate_vector(move_direction),
                    look_direction: camera_tranform.rotate_vector(Vector3::X),
                    throttle: 1.0,
                };

                input.apply(
                    &mut self.momenta[*inputs_id],
                    &mut self.transforms[*inputs_id],
                );

                inputs.insert(tick, input);
            }

            step_world(&self.colliders, &mut self.momenta, &mut self.transforms, dt);
        });

        let dampening = 0.01;

        if let Some((camera_id, camera)) = self.camera.obtain_mut() {
            camera.handle_input(CameraInput {
                delta_x: self.input.mouse_diff().0 * dampening,
                delta_y: self.input.mouse_diff().1 * dampening,
                delta_scroll: self.input.scroll_diff().1,
            });

            camera.update_tranform(&mut self.transforms, *camera_id);
        }
    }
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: client.exe SERVER_IP_ADDRESS:SERVER_PORT");
        return;
    }

    let Ok(address) = SocketAddr::from_str(&args[1]) else {
        eprintln!("Invalid IP address and port");
        return;
    };

    app.server_address = address;

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).unwrap();
}
