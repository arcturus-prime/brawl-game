use std::collections::VecDeque;

use crate::{math::Transform3, physics::Moment, player::PlayerInputState};

pub enum Packet {
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

pub struct NetworkQueue {
    send: VecDeque<Packet>,
    receive: VecDeque<Packet>,
}
