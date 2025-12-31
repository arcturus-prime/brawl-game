use std::{
    collections::HashMap,
    hash::Hash,
    io,
    net::{SocketAddr, SocketAddrV4, UdpSocket},
    sync::mpsc::{Receiver, SendError, Sender, TryRecvError, TrySendError},
};

use enum_discriminant::discriminant;
use thiserror::Error;

use crate::{
    math::{Transform3, Vector3},
    player::PlayerInputState,
    utility::{ByteStream, ByteStreamError, IdReserver, SparseSet},
};

#[discriminant(u8)]
#[derive(Clone)]
pub enum Packet {
    ClientHello = 0,
    PlayerJoin {
        id: usize,
        is_you: bool,
    } = 1,
    PlayerLeave {
        id: usize,
    } = 2,
    PlayerInput {
        input: PlayerInputState,
    } = 3,
    PlayerMovement {
        transform: Transform3,
        velocity: Vector3,
        id: usize,
    } = 4,
}

#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("Invalid packet ID found")]
    InvalidPacketId,
    #[error("Stream read/write error")]
    ByteStreamError(#[from] ByteStreamError),
    #[error("Socket error occurred")]
    SocketError(#[from] io::Error),
    #[error("There was a MPSC send channel error")]
    MpscSendError(#[from] SendError<(SocketAddr, Packet)>),
    #[error("There was a MPSC receive channel error")]
    MpscReceiveError(#[from] TryRecvError),
}

impl Packet {
    pub fn serialize(self, stream: &mut ByteStream) -> Result<(), NetworkError> {
        stream.write_u8(self.discriminant())?;

        match self {
            Packet::ClientHello => return Ok(()),
            Packet::PlayerMovement {
                transform,
                velocity,
                id,
            } => {
                stream.write_u64(id as u64)?;
                stream.write_transform3(transform)?;
                stream.write_vec3(velocity)?;
            }
            Packet::PlayerInput { input } => {
                stream.write_vec3(input.look_direction)?;
                stream.write_vec3(input.want_direction)?;
                stream.write_f32(input.throttle)?;
            }
            Packet::PlayerJoin { id, is_you } => {
                stream.write_u64(id as u64)?;
                stream.write_u8(is_you as u8)?;
            }
            Packet::PlayerLeave { id } => {
                stream.write_u64(id as u64)?;
            }
        };

        Ok(())
    }

    pub fn deserialize(stream: &mut ByteStream) -> Result<Self, NetworkError> {
        let id = stream.read_u8()?;

        Ok(match id {
            0 => Packet::ClientHello,
            1 => Packet::PlayerJoin {
                id: stream.read_u64()? as usize,
                is_you: stream.read_u8()? == 1,
            },
            2 => Self::PlayerLeave {
                id: stream.read_u64()? as usize,
            },
            3 => Self::PlayerInput {
                input: PlayerInputState {
                    look_direction: stream.read_vec3()?,
                    want_direction: stream.read_vec3()?,
                    throttle: stream.read_f32()?,
                },
            },
            4 => Self::PlayerMovement {
                id: stream.read_u64()? as usize,
                transform: stream.read_transform3()?,
                velocity: stream.read_vec3()?,
            },
            _ => return Err(NetworkError::InvalidPacketId),
        })
    }
}

pub struct Network {
    tx: Sender<(SocketAddr, Packet)>,
    rx: Receiver<(SocketAddr, Packet)>,

    registered_addresses: SparseSet<SocketAddr>,
    registered_entities: HashMap<SocketAddr, usize>,

    entity_to_network: SparseSet<usize>,
    network_to_entity: SparseSet<usize>,
    network_reserver: IdReserver,
}

impl Network {
    pub fn new(self_address: SocketAddr) -> Result<Self, NetworkError> {
        let socket = UdpSocket::bind(self_address)?;

        let (send_tx, send_rx): (Sender<(SocketAddr, Packet)>, Receiver<(SocketAddr, Packet)>) =
            std::sync::mpsc::channel();
        let (receive_tx, receive_rx) = std::sync::mpsc::channel();

        let receive_socket = socket.try_clone()?;
        let mut receive_buffer = vec![0; 1024];
        std::thread::spawn(move || {
            loop {
                let Ok((_, address)) = receive_socket.recv_from(&mut receive_buffer) else {
                    continue;
                };

                let mut stream = ByteStream::new(&mut receive_buffer);

                let packet = match Packet::deserialize(&mut stream) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("Deserialization error {}", e);
                        continue;
                    }
                };

                if let Err(e) = receive_tx.send((address, packet)) {
                    return;
                }
            }
        });

        let mut send_buffer = vec![0; 1024];
        std::thread::spawn(move || {
            loop {
                let Ok((address, packet)) = send_rx.recv() else {
                    return;
                };

                let mut stream = ByteStream::new(&mut send_buffer);

                if let Err(e) = packet.serialize(&mut stream) {
                    println!("Packet serialization error {}", e);
                    continue;
                }

                if let Err(e) = socket.send_to(&send_buffer, address) {
                    println!("Send error {}", e);
                    continue;
                }
            }
        });

        Ok(Self {
            tx: send_tx,
            rx: receive_rx,
            registered_addresses: SparseSet::default(),
            registered_entities: HashMap::new(),
            entity_to_network: SparseSet::default(),
            network_reserver: IdReserver::default(),
            network_to_entity: SparseSet::default(),
        })
    }

    pub fn receive(&mut self, reserver: &mut IdReserver) -> Result<(usize, Packet), NetworkError> {
        let (address, packet) = self.rx.try_recv()?;

        if !self.registered_entities.contains_key(&address) {
            let id = reserver.reserve();

            self.registered_entities.insert(address, id);
            self.registered_addresses.insert(id, address);
        }

        let id = self.registered_entities[&address];

        Ok((id, packet))
    }

    pub fn send(&mut self, entity: usize, packet: Packet) -> Result<(), NetworkError> {
        let address = self.registered_addresses[entity];

        self.tx.send((address, packet))?;

        Ok(())
    }

    pub fn send_all(&mut self, packet: Packet) -> Result<(), NetworkError> {
        for (_, address) in self.registered_addresses.iter() {
            self.tx.send((*address, packet.clone()))?;
        }

        Ok(())
    }

    pub fn send_all_except(&mut self, except: usize, packet: Packet) -> Result<(), NetworkError> {
        for (id, address) in self.registered_addresses.iter() {
            if *id == except {
                continue;
            }

            self.tx.send((*address, packet.clone()))?;
        }

        Ok(())
    }

    pub fn add_client(&mut self, address: SocketAddr, reserver: &mut IdReserver) -> usize {
        if !self.registered_entities.contains_key(&address) {
            let id = reserver.reserve();

            self.registered_entities.insert(address, id);
            self.registered_addresses.insert(id, address);
        }

        let id = self.registered_entities[&address];

        id
    }

    pub fn delete_client(&mut self, entity: usize) {
        let address = self.registered_addresses[entity];

        self.registered_addresses.delete(entity);
        self.registered_entities.remove(&address);
    }

    pub fn reserve_real_entity(
        &mut self,
        reserver: &mut IdReserver,
        network_entity: usize,
    ) -> usize {
        let id = reserver.reserve();

        self.entity_to_network.insert(id, network_entity);
        self.network_to_entity.insert(network_entity, id);

        id
    }

    pub fn reserve_network_entity(&mut self, real_entity: usize) -> usize {
        let network_id = self.network_reserver.reserve();

        self.entity_to_network.insert(real_entity, network_id);
        self.network_to_entity.insert(network_id, real_entity);

        network_id
    }

    pub fn delete_real_entity(&mut self, entity: usize) {
        let network_id = self.entity_to_network[entity];

        self.entity_to_network.delete(entity);
        self.network_to_entity.delete(network_id);
    }

    pub fn delete_network_entity(&mut self, entity: usize) {
        let real_id = self.network_to_entity[entity];

        self.entity_to_network.delete(real_id);
        self.network_to_entity.delete(entity);
    }

    pub fn get_real_entity(&self, network_entity: usize) -> Option<&usize> {
        self.network_to_entity.get(network_entity)
    }

    pub fn get_network_entity(&self, real_entity: usize) -> Option<&usize> {
        self.entity_to_network.get(real_entity)
    }
}
