use std::{
    io,
    net::{SocketAddr, UdpSocket},
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
    ServerHello = 1,
    PlayerJoin {
        id: usize,
    } = 2,
    PlayerLeave {
        id: usize,
    } = 3,
    PlayerInput {
        input: PlayerInputState,
    } = 4,
    PlayerMovement {
        transform: Transform3,
        velocity: Vector3,
        id: usize,
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
            Packet::ServerHello => return Ok(()),
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
            Packet::PlayerJoin { id } => {
                stream.write_u64(id as u64)?;
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
            1 => Packet::ServerHello,
            2 => Packet::PlayerJoin {
                id: stream.read_u64()? as usize,
            },
            3 => Self::PlayerLeave {
                id: stream.read_u64()? as usize,
            },
            4 => Self::PlayerInput {
                input: PlayerInputState {
                    look_direction: stream.read_vec3()?,
                    want_direction: stream.read_vec3()?,
                    throttle: stream.read_f32()?,
                },
            },
            5 => Self::PlayerMovement {
                id: stream.read_u64()? as usize,
                transform: stream.read_transform3()?,
                velocity: stream.read_vec3()?,
            },
            _ => return Err(NetworkError::InvalidPacketId),
        })
    }
}

pub struct NetworkSender(Sender<(SocketAddr, Packet)>);

impl NetworkSender {
    pub fn send(&mut self, address: SocketAddr, packet: Packet) -> Result<(), NetworkError> {
        self.0.send((address, packet))?;

        Ok(())
    }

    pub fn send_all(
        &mut self,
        addresses: &SparseSet<SocketAddr>,
        packet: Packet,
    ) -> Result<(), NetworkError> {
        for (_, address) in addresses.iter() {
            self.send(*address, packet.clone())?
        }

        Ok(())
    }
}

pub struct NetworkReceiver(Receiver<(SocketAddr, Packet)>);

impl NetworkReceiver {
    pub fn get_incoming(&self) -> impl Iterator<Item = (SocketAddr, Packet)> {
        self.0.try_iter()
    }
}

pub struct Network {
    pub tx: NetworkSender,
    pub rx: NetworkReceiver,
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
            tx: NetworkSender(send_tx),
            rx: NetworkReceiver(receive_rx),
        })
    }
}
