use crate::math::Vector;
use crate::physics::RigidCapsule;

pub struct Character {
    physics: RigidCapsule,

    look_direction: Vector,
    move_direction: Vector,
}