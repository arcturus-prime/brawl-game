use core::net;
use std::{
    collections::BTreeMap,
    net::{IpAddr, SocketAddr, SocketAddrV4},
    str::FromStr,
    time::Instant,
};

use shared::{
    math::{GeometryTree, Transform3},
    net::{Network, NetworkError, Packet},
    physics::Moment,
    player::{PlayerData, PlayerInputState},
    tick::Ticker,
    utility::{IdReserver, SparseSet},
};

pub struct Game {
    transforms: SparseSet<Transform3>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,
    inputs: SparseSet<BTreeMap<u32, PlayerInputState>>,
    players: SparseSet<PlayerData>,

    last_update: Instant,
    network: Network,
    reserver: IdReserver,
    ticker: Ticker,
}

impl Game {
    pub fn update(&mut self) {
        let new_update_time = Instant::now();
        let dt = (new_update_time - self.last_update).as_secs_f32();
        self.last_update = new_update_time;

        while let Ok((client_id, packet)) = self.network.receive(&mut self.reserver) {
            match packet {
                shared::net::Packet::ClientHello => {
                    self.players.insert(client_id, PlayerData::default());
                    self.inputs.insert(client_id, BTreeMap::new());
                    self.transforms.insert(client_id, Transform3::identity());
                    self.momenta.insert(client_id, Moment::new(5.0));

                    let tree = GeometryTree::from_cube(1.0, 1.0, 1.0, 0);
                    self.colliders.insert(client_id, tree);

                    let net_id = self.network.reserve_network_entity(client_id);

                    self.network
                        .send_all_except(
                            client_id,
                            Packet::PlayerJoin {
                                id: net_id,
                                is_you: false,
                            },
                        )
                        .unwrap();

                    self.network
                        .send(
                            client_id,
                            Packet::PlayerJoin {
                                id: net_id,
                                is_you: true,
                            },
                        )
                        .unwrap();
                }
                shared::net::Packet::PlayerJoin { id, is_you } => {
                    eprintln!("Unexpected player join packet received");
                }
                shared::net::Packet::PlayerLeave { id } => {
                    eprintln!("Unexpected player leave packet received");
                }
                shared::net::Packet::PlayerInput { input } => {}
                shared::net::Packet::PlayerMovement {
                    transform,
                    velocity,
                    id,
                } => todo!(),
            }
        }

        self.ticker.update(dt, |tick, dt| {})
    }

    pub fn host(address: SocketAddr) -> Result<Self, NetworkError> {
        Ok(Self {
            last_update: Instant::now(),
            transforms: SparseSet::default(),
            inputs: SparseSet::default(),
            players: SparseSet::default(),
            colliders: SparseSet::default(),
            momenta: SparseSet::default(),
            reserver: IdReserver::default(),
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
