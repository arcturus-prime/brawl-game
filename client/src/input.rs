use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use winit::keyboard::{Key, KeyCode};

pub enum Input {
    Forward,
    Backward,
    Left,
    Right,
    LookDelta(f32, f32),
}

pub struct InputConfig {
    forward: KeyCode,
    backward: KeyCode,
    left: KeyCode,
    right: KeyCode,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            forward: KeyCode::KeyW,
            backward: KeyCode::KeyS,
            left: KeyCode::KeyA,
            right: KeyCode::KeyD,
        }
    }
}

#[derive(Default)]
pub struct InputStream {
    queue: VecDeque<Input>,
    config: InputConfig,
}
