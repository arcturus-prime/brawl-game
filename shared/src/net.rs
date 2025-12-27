use enum_discriminant::discriminant;
use thiserror::Error;

use crate::{
    math::{Transform3, Vector3},
    player::PlayerInputState,
    utility::{ByteStream, ByteStreamError},
};

#[discriminant(u8)]
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
pub enum PacketError {
    #[error("Invalid packet ID found")]
    InvalidId,
    #[error("Stream read/write error")]
    ByteStreamError(#[from] ByteStreamError),
}

impl Packet {
    pub fn serialize(self, stream: &mut ByteStream) -> Result<(), PacketError> {
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

    pub fn deserialize(stream: &mut ByteStream) -> Result<Self, PacketError> {
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
            _ => return Err(PacketError::InvalidId),
        })
    }
}
