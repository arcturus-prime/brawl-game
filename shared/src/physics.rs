use crate::{
    geometry::{GeometryTree, RaycastData},
    math::{Transform3, Vector3},
    utility::SparseSet,
};

pub const LINEAR_DAMPENING: f32 = 0.99;

pub struct Moment {
    pub velocity: Vector3,
    pub mass: f32,
}

impl Moment {
    pub fn new(mass: f32) -> Self {
        Self {
            velocity: Vector3::zero(),
            mass,
        }
    }

    pub fn update(&mut self, body: &mut Transform3, dt: f32) {
        body.position += self.velocity * dt;
    }

    pub fn apply_impulse(&mut self, impulse: Vector3) {
        self.velocity += impulse / self.mass;
    }
}

pub fn step_world(
    colliders: &SparseSet<GeometryTree>,
    momenta: &mut SparseSet<Moment>,
    transforms: &mut SparseSet<Transform3>,
    dt: f32,
) {
    for (id, moment) in momenta.iter_mut() {
        transforms[*id].position += moment.velocity * dt;
        moment.velocity *= LINEAR_DAMPENING;
    }
}
