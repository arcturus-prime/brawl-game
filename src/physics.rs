use std::f32::NAN;

use crate::math::{Vector};

const SOLID: Vector = Vector::new(NAN, NAN, NAN, NAN);
const EMPTY: Vector = Vector::new(-NAN, -NAN, -NAN, -NAN);

pub struct World {
    planes: Vec<Vector>
}

#[derive(Default)]
pub struct RigidCapsule {
    position: Vector,
    velocity: Vector,
    rotation: Vector,

    bounds: Vector,
}

pub struct CollisionData {
    collided: bool,
    positions: Vector,
    normal: Vector,
}