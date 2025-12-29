use std::{
    error::Error,
    net::{SocketAddr, UdpSocket},
    sync::mpsc::{Receiver, Sender},
};

use shared::{
    net::Packet,
    utility::{ByteStream, IdReserver, SparseSet},
};

pub struct NetworkClient {
    receive: std::sync::mpsc::Receiver<Packet>,
    send: std::sync::mpsc::Sender<Packet>,

    entity_map: SparseSet<usize>,
}

impl NetworkClient {
    pub fn connect(
        recieve_addr: SocketAddr,
        send_addr: SocketAddr,
    ) -> Result<Self, Box<dyn Error>> {
        let send_socket = UdpSocket::bind(recieve_addr)?;
        send_socket.connect(send_addr)?;

        let (send_tx, send_rx): (Sender<Packet>, Receiver<Packet>) = std::sync::mpsc::channel();
        let (receive_tx, receive_rx) = std::sync::mpsc::channel();

        let receive_socket = send_socket.try_clone()?;
        let mut receive_buffer = vec![0; 1024];
        std::thread::spawn(move || {
            loop {
                if let Err(e) = receive_socket.recv(&mut receive_buffer) {
                    println!("Failed to receive packet from socket, reason: {}", e);
                    continue;
                };

                let mut stream = ByteStream::new(&mut receive_buffer);
                let packet = match Packet::deserialize(&mut stream) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("Failed to deserialize packet, reason: {}", e);
                        continue;
                    }
                };

                if let Err(e) = receive_tx.send(packet) {
                    println!("Failed to send packet to mpsc, reason: {}", e);
                    continue;
                }
            }
        });

        let mut send_buffer = vec![0; 1024];
        std::thread::spawn(move || {
            loop {
                let mut stream = ByteStream::new(&mut send_buffer);

                let Ok(packet) = send_rx.recv() else {
                    println!("Failed to get packet from mspc");
                    continue;
                };

                if let Err(e) = packet.serialize(&mut stream) {
                    println!("Failed to serialize packet, reason: {}", e);
                    continue;
                }

                if let Err(e) = send_socket.send(&send_buffer) {
                    println!("Failed to send packet to socket, reason: {}", e);
                    continue;
                }
            }
        });

        Ok(Self {
            receive: receive_rx,
            send: send_tx,
            entity_map: SparseSet::default(),
        })
    }

    pub fn translate_entity(&self, network_entity: usize) -> usize {
        self.entity_map[network_entity]
    }

    pub fn reserve_entity(&mut self, network_entity: usize, reserver: &mut IdReserver) -> usize {
        let id = reserver.reserve();

        self.entity_map.insert(network_entity, id);

        id
    }

    pub fn send(&mut self, packet: Packet) -> Result<(), Box<dyn Error>> {
        self.send.send(packet)?;

        Ok(())
    }

    pub fn receive(&mut self) -> Result<Packet, Box<dyn Error>> {
        let packet = self.receive.try_recv()?;

        Ok(packet)
    }
}
