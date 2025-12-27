use std::net::{IpAddr, UdpSocket};

pub struct PlayerClient {
    address: IpAddr,
}

pub struct NetworkServer {
    socket: UdpSocket,
}

impl NetworkServer {}
