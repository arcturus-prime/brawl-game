use std::{
    collections::HashMap,
    io,
    net::{SocketAddr, SocketAddrV4, UdpSocket},
    sync::mpsc::{Receiver, SendError, Sender, TryRecvError, TrySendError},
};

use enum_discriminant::discriminant;
use thiserror::Error;

use crate::{
    math::{Transform3, Vector3},
    player::PlayerInputState,
    utility::{ByteStream, ByteStreamError, Entity, EntityReserver, SparseSet},
};

#[discriminant(u8)]
#[derive(Clone)]
pub enum Packet {
    ClientHello = 0,
    PlayerJoin {
        net_entity: usize,
        is_you: bool,
    } = 1,
    PlayerLeave {
        net_entity: usize,
    } = 2,
    PlayerInput {
        input: PlayerInputState,
        tick: u32,
    } = 3,
    PlayerMovement {
        transform: Transform3,
        velocity: Vector3,
        net_entity: usize,
    } = 4,
    TickSync {
        skip: u32,
    } = 5,
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
                net_entity: id,
            } => {
                stream.write_u64(id as u64)?;
                stream.write_transform3(transform)?;
                stream.write_vec3(velocity)?;
            }
            Packet::PlayerInput { input, tick } => {
                stream.write_vec3(input.look_direction)?;
                stream.write_vec3(input.want_direction)?;
                stream.write_f32(input.throttle)?;
                stream.write_u32(tick)?;
            }
            Packet::PlayerJoin {
                net_entity: id,
                is_you,
            } => {
                stream.write_u64(id as u64)?;
                stream.write_u8(is_you as u8)?;
            }
            Packet::PlayerLeave { net_entity: id } => {
                stream.write_u64(id as u64)?;
            }
            Packet::TickSync { skip } => {
                stream.write_u32(skip)?;
            }
        };

        Ok(())
    }

    pub fn deserialize(stream: &mut ByteStream) -> Result<Self, NetworkError> {
        let id = stream.read_u8()?;

        Ok(match id {
            0 => Packet::ClientHello,
            1 => Packet::PlayerJoin {
                net_entity: stream.read_u64()? as usize,
                is_you: stream.read_u8()? == 1,
            },
            2 => Self::PlayerLeave {
                net_entity: stream.read_u64()? as usize,
            },
            3 => Self::PlayerInput {
                input: PlayerInputState {
                    look_direction: stream.read_vec3()?,
                    want_direction: stream.read_vec3()?,
                    throttle: stream.read_f32()?,
                },
                tick: stream.read_u32()?,
            },
            4 => Self::PlayerMovement {
                net_entity: stream.read_u64()? as usize,
                transform: stream.read_transform3()?,
                velocity: stream.read_vec3()?,
            },
            5 => Self::TickSync {
                skip: stream.read_u32()?,
            },
            _ => return Err(NetworkError::InvalidPacketId),
        })
    }
}

pub struct Network {
    tx: Sender<(SocketAddr, Packet)>,
    rx: Receiver<(SocketAddr, Packet)>,

    client_addresses: SparseSet<SocketAddr>,
    client_entities: HashMap<SocketAddr, Entity>,

    entity_to_network: SparseSet<Entity>,
    network_to_entity: SparseSet<Entity>,
    network_reserver: EntityReserver,
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
            client_addresses: SparseSet::default(),
            client_entities: HashMap::new(),
            entity_to_network: SparseSet::default(),
            network_reserver: EntityReserver::default(),
            network_to_entity: SparseSet::default(),
        })
    }

    pub fn receive(
        &mut self,
        reserver: &mut EntityReserver,
    ) -> Result<(Entity, Packet), NetworkError> {
        let (address, packet) = self.rx.try_recv()?;

        if !self.client_entities.contains_key(&address) {
            let id = reserver.reserve();

            self.client_entities.insert(address, id);
            self.client_addresses.insert(id, address);
        }

        let entity = self.client_entities[&address];

        Ok((entity, packet))
    }

    pub fn send(&mut self, entity: Entity, packet: Packet) -> Result<(), NetworkError> {
        let address = self.client_addresses[entity];

        self.tx.send((address, packet))?;

        Ok(())
    }

    pub fn send_all(&mut self, packet: Packet) -> Result<(), NetworkError> {
        for (_, address) in self.client_addresses.iter() {
            self.tx.send((*address, packet.clone()))?;
        }

        Ok(())
    }

    pub fn send_all_except(&mut self, except: Entity, packet: Packet) -> Result<(), NetworkError> {
        for (entity, address) in self.client_addresses.iter() {
            if *entity == except {
                continue;
            }

            self.tx.send((*address, packet.clone()))?;
        }

        Ok(())
    }

    pub fn get_clients(&self) -> Vec<Entity> {
        self.client_addresses.iter().map(|i| *i.0).collect()
    }

    pub fn add_client(&mut self, address: SocketAddr, reserver: &mut EntityReserver) -> Entity {
        if !self.client_entities.contains_key(&address) {
            let id = reserver.reserve();

            self.client_entities.insert(address, id);
            self.client_addresses.insert(id, address);
        }

        let id = self.client_entities[&address];

        id
    }

    pub fn delete_client(&mut self, entity: Entity) {
        let address = self.client_addresses[entity];

        self.client_addresses.delete(entity);
        self.client_entities.remove(&address);
    }

    pub fn reserve_real_entity(
        &mut self,
        reserver: &mut EntityReserver,
        network_entity: Entity,
    ) -> Entity {
        let entity = reserver.reserve();

        self.entity_to_network.insert(entity, network_entity);
        self.network_to_entity.insert(network_entity, entity);

        entity
    }

    pub fn reserve_network_entity(&mut self, real_entity: Entity) -> Entity {
        let network_entity = self.network_reserver.reserve();

        self.entity_to_network.insert(real_entity, network_entity);
        self.network_to_entity.insert(network_entity, real_entity);

        network_entity
    }

    pub fn delete_from_real_entity(&mut self, entity: Entity) {
        let network_entity = self.entity_to_network[entity];

        self.entity_to_network.delete(entity);
        self.network_to_entity.delete(network_entity);
    }

    pub fn delete_from_network_entity(&mut self, entity: Entity) {
        let real_entity = self.network_to_entity[entity];

        self.entity_to_network.delete(real_entity);
        self.network_to_entity.delete(entity);
    }

    pub fn get_real_entity(&self, network_entity: Entity) -> Option<&Entity> {
        self.network_to_entity.get(network_entity)
    }

    pub fn get_network_entity(&self, real_entity: Entity) -> Option<&Entity> {
        self.entity_to_network.get(real_entity)
    }
}
