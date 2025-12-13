use crate::{
    math::{GeometryTree, SpherecastData, Transform, Vector3},
    utility::SparseSet,
};

pub const COLLISION_RESTITUTION: f32 = 0.5;
pub const LINEAR_DAMPENING: f32 = 0.95;

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

    pub fn update(&mut self, body: &mut Transform, dt: f32) {
        body.position += self.velocity * dt;
    }

    pub fn apply_impulse(&mut self, impulse: Vector3) {
        self.velocity += impulse / self.mass;
    }
}

pub fn step_world(
    momenta: &mut SparseSet<Moment>,
    transforms: &mut SparseSet<Transform>,
    colliders: &SparseSet<GeometryTree>,
    dt: f32,
) {
    let mut collisions = vec![];
    for (id_a, body_a) in colliders.iter() {
        let mut earliest_collision: Option<(SpherecastData, usize, usize)> = None;

        for (id_b, body_b) in colliders.iter() {
            if id_a == id_b {
                continue;
            }

            let relative_velocity = if let Some(moment_b) = momenta.get(*id_b)
                && let Some(moment_a) = momenta.get(*id_a)
            {
                moment_a.velocity - moment_b.velocity
            } else if let Some(moment_a) = momenta.get(*id_a) {
                moment_a.velocity
            } else {
                continue;
            };

            let relative_position = transforms[*id_a].position - transforms[*id_b].position;

            let Some(collision) = body_b.spherecast(
                body_a.get_bounds_radius(),
                relative_position,
                relative_velocity * dt,
            ) else {
                continue;
            };

            if let Some((earliest, id_a, id_b)) = earliest_collision
                && earliest.t > collision.t
            {
                earliest_collision = Some((collision, id_a, id_b))
            } else if earliest_collision.is_none() {
                earliest_collision = Some((collision, *id_a, *id_b))
            }
        }

        if let Some(collision) = earliest_collision {
            collisions.push(collision);
        }
    }

    for (collision, id_a, id_b) in collisions {
        let mut mass = 1.0 / momenta[id_a].mass;
        let mut velocity = momenta[id_a].velocity;

        if let Some(moment) = momenta.get(id_b) {
            mass += 1.0 / moment.mass;
            velocity -= moment.velocity;
        }

        let velocity_along = collision.normal.dot(velocity);
        let impulse = collision.normal * -(1.0 + COLLISION_RESTITUTION) * velocity_along / mass;

        momenta[id_a].apply_impulse(impulse);
    }

    for x in momenta.iter_mut() {
        transforms[*x.0].position += x.1.velocity * dt;
        x.1.velocity *= LINEAR_DAMPENING;
    }
}
