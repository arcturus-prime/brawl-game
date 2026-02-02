use std::{
    net::{IpAddr, SocketAddr},
    str::FromStr,
    time::Instant,
};

use shared::{
    math::{GeometryTree, HalfspaceMetadata, Transform3},
    net::{Network, NetworkError, Packet},
    physics::{Moment, step_world},
    player::PlayerData,
    tick::Ticker,
    utility::{EntityReserver, SparseSet},
};

mod assets;

pub struct Game {
    players: SparseSet<PlayerData>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    transforms: SparseSet<Transform3>,

    network: Network,
    last_update: Instant,
    reserver: EntityReserver,
    ticker: Ticker,
}

impl Game {
    pub fn update(&mut self) {
        let new_update_time = Instant::now();
        let dt = (new_update_time - self.last_update).as_secs_f32();
        self.last_update = new_update_time;

        while let Ok((client_entity, packet)) = self.network.receive(&mut self.reserver) {
            match packet {
                Packet::ClientHello => {
                    let collider =
                        GeometryTree::from_cube(1.0, 1.0, 1.0, HalfspaceMetadata::default());

                    self.transforms
                        .insert(client_entity, Transform3::identity());
                    self.momenta.insert(client_entity, Moment::new(5.0));
                    self.colliders.insert(client_entity, collider);
                    self.players.insert(client_entity, PlayerData::default());

                    let net_entity = self.network.reserve_network_entity(client_entity);

                    self.network
                        .send(
                            client_entity,
                            Packet::TickSync {
                                skip: self.ticker.tick,
                            },
                        )
                        .unwrap();

                    self.network
                        .send_all_except(
                            client_entity,
                            Packet::PlayerJoin {
                                net_entity,
                                is_you: false,
                            },
                        )
                        .unwrap();

                    for x in self.network.get_clients() {
                        let net_entity = self.network.get_network_entity(x).unwrap();

                        self.network
                            .send(
                                client_entity,
                                Packet::PlayerJoin {
                                    net_entity: *net_entity,
                                    is_you: x == client_entity,
                                },
                            )
                            .unwrap();
                    }
                }
                Packet::PlayerJoin {
                    net_entity: id,
                    is_you,
                } => {
                    eprintln!("Unexpected player join packet received");
                }
                Packet::PlayerLeave { net_entity: id } => {
                    eprintln!("Unexpected player leave packet received");
                }
                Packet::PlayerInput { input, tick } => {
                    self.players[client_entity].inputs.insert(tick, input);
                }
                Packet::PlayerMovement {
                    transform,
                    velocity,
                    net_entity: id,
                } => {
                    eprintln!("Unexpected player movement packet received")
                }
                Packet::TickSync { skip } => {
                    eprintln!("Unexpected tick sync packet received");
                }
            }
        }

        self.ticker.update(dt, |tick, dt| {
            for (id, data) in self.players.iter_mut() {
                let Some(mut current_input) = data.inputs.remove(&tick) else {
                    continue;
                };

                current_input.apply(&mut self.momenta[*id], &mut self.transforms[*id]);
            }

            step_world(&self.colliders, &mut self.momenta, &mut self.transforms, dt);

            for (entity, data) in self.players.iter() {
                let Some(net_entity) = self.network.get_network_entity(*entity) else {
                    continue;
                };

                self.network
                    .send_all(Packet::PlayerMovement {
                        transform: self.transforms[*entity],
                        velocity: self.momenta[*entity].velocity,
                        net_entity: *net_entity,
                    })
                    .unwrap()
            }
        })
    }

    pub fn host(address: SocketAddr) -> Result<Self, NetworkError> {
        Ok(Self {
            last_update: Instant::now(),
            colliders: SparseSet::default(),
            momenta: SparseSet::default(),
            transforms: SparseSet::default(),
            players: SparseSet::default(),
            reserver: EntityReserver::default(),
            network: Network::new(address)?,
            ticker: Ticker::default(),
        })
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: server.exe PORT_NUMBER");
        return;
    }

    let port = u16::from_str(&args[1]).unwrap();

    let address = SocketAddr::new(IpAddr::from_str("0.0.0.0").unwrap(), port);
    let mut game = Game::host(address).unwrap();

    loop {
        game.update();
    }
}
