use crate::{math::Transform3, physics::Moment, player::PlayerInputState};

#[repr(C, u8)]
pub enum Packet {
    ClientHello = 0,
    ServerHello,
    PlayerMovement {
        transform: Transform3,
        moment: Moment,
        id: usize,
    },
    PlayerInput {
        input: PlayerInputState,
    },
    PlayerJoin {
        id: usize,
    },
    PlayerLeave {
        id: usize,
    },
}

impl Packet {
    pub fn id(&self) -> u8 {
        unsafe { *<*const _>::from(self).cast::<u8>() }
    }

    pub fn serialize(&self, buffer: &mut Vec<u8>) {}

    pub fn deserialize(buffer: &[u8]) -> Self {
        Self::ClientHello
    }
}
