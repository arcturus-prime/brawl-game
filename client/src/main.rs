use std::{collections::BTreeMap, net::SocketAddr, str::FromStr, time::Instant};

use shared::{
    math::{GeometryTree, Mesh, Transform3, Vector3},
    net::{Network, Packet},
    physics::{Moment, step_world},
    player::{PlayerData, PlayerInputState},
    tick::Ticker,
    utility::{EntityReserver, SingletonSet, SparseSet},
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

    players: SparseSet<PlayerData>,
    local_player: SingletonSet<()>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    transforms: SparseSet<Transform3>,
    renderable: SparseSet<Renderable>,
    camera: SingletonSet<CameraData>,

    reserver: EntityReserver,
    network: Network,
    last_update: Instant,
    ticker: Ticker,
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
            reserver: EntityReserver::default(),
            players: SparseSet::default(),
            local_player: SingletonSet::default(),
            colliders: SparseSet::default(),
            momenta: SparseSet::default(),
            renderable: SparseSet::default(),
            transforms: SparseSet::default(),
            camera: SingletonSet::default(),
            renderer,
            ticker: Ticker::default(),
            input: WinitInputHelper::new(),
        }
    }

    pub fn render(&mut self) {
        let Some((camera_id, camera)) = self.camera.obtain() else {
            return;
        };

        self.renderer
            .draw_scene(
                self.transforms[*camera_id],
                camera.fov_y,
                &self.renderable,
                &self.transforms,
            )
            .unwrap();
    }

    pub fn update(&mut self) {
        let new_update_time = Instant::now();
        let dt = (new_update_time - self.last_update).as_secs_f32();
        self.last_update = new_update_time;

        while let Ok((_, packet)) = self.network.receive(&mut self.reserver) {
            match packet {
                Packet::ClientHello => {
                    eprintln!("Got unexpected ClientHello from server");
                }
                Packet::PlayerJoin { net_entity, is_you } => {
                    let entity = self
                        .network
                        .reserve_real_entity(&mut self.reserver, net_entity);

                    let collider = GeometryTree::load_from_mesh(&Mesh::create_cube_mesh());

                    self.transforms.insert(entity, Transform3::identity());
                    self.momenta.insert(entity, Moment::new(5.0));

                    let mut renderable = self.renderer.create_renderable().unwrap();

                    renderable.set_nodes(&collider).unwrap();

                    self.colliders.insert(entity, collider);
                    self.renderable.insert(entity, renderable);
                    self.players.insert(entity, PlayerData::default());

                    if is_you {
                        let camera_entity = self.reserver.reserve();

                        self.camera.insert(camera_entity, CameraData::orbit(entity));
                        self.transforms
                            .insert(camera_entity, Transform3::identity());

                        self.local_player.insert(entity, ());
                    }
                }
                Packet::PlayerLeave { net_entity: id } => {}
                Packet::PlayerInput { input, tick } => {
                    eprintln!("Received unexpected player input packet from server");
                }
                Packet::PlayerMovement {
                    transform,
                    velocity,
                    net_entity,
                } => {
                    let Some(real_id) = self.network.get_real_entity(net_entity) else {
                        eprintln!("Received player id from server that is not yet registered");
                        continue;
                    };

                    self.transforms[*real_id] = transform;
                    self.momenta[*real_id].velocity = velocity;
                }
                Packet::TickSync { skip } => {
                    self.ticker.set_tick(skip + 1);
                }
            }
        }

        self.ticker.update(dt, |tick, dt| {
            if let Some((local_player_entity, _)) = self.local_player.obtain()
                && let Some((camera_id, _)) = self.camera.obtain()
                && let Some(data) = self.players.get_mut(*local_player_entity)
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
                    &mut self.momenta[*local_player_entity],
                    &mut self.transforms[*local_player_entity],
                );

                self.network
                    .send_all(Packet::PlayerInput {
                        input: input.clone(),
                        tick,
                    })
                    .unwrap();

                data.inputs.insert(tick, input);
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
