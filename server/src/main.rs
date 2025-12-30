use std::{
    net::{IpAddr, SocketAddr, SocketAddrV4},
    str::FromStr,
    time::Instant,
};

use shared::{
    math::{GeometryTree, Transform3},
    net::{Network, NetworkError, Packet},
    physics::Moment,
    player::PlayerData,
    tick::Ticker,
    utility::{IdReserver, SparseSet},
};

pub struct Game {
    last_update: Instant,

    connections: SparseSet<SocketAddr>,
    transforms: SparseSet<Transform3>,
    players: SparseSet<PlayerData>,
    colliders: SparseSet<GeometryTree>,
    momenta: SparseSet<Moment>,

    network: Network,
    reserver: IdReserver,
    ticker: Ticker,
}

impl Game {
    pub fn update(&mut self) {
        let new_update_time = Instant::now();
        let dt = (new_update_time - self.last_update).as_secs_f32();
        self.last_update = new_update_time;

        for (address, packet) in self.network.rx.get_incoming() {
            match packet {
                shared::net::Packet::ClientHello => {
                    self.network.tx.send(address, Packet::ServerHello).unwrap();
                }
                shared::net::Packet::ServerHello => todo!(),
                shared::net::Packet::PlayerJoin { id } => todo!(),
                shared::net::Packet::PlayerLeave { id } => todo!(),
                shared::net::Packet::PlayerInput { input } => todo!(),
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
            connections: SparseSet::default(),
            last_update: Instant::now(),
            transforms: SparseSet::default(),
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
