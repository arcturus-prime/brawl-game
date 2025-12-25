use crate::{
    math::{GeometryTree, RaycastData, Transform3, Vector3},
    utility::SparseSet,
};

pub const COLLISION_RESTITUTION: f32 = 1.0;
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
    let mut colliding = vec![];
    let mut non_colliding = vec![];

    for (id_a, moment_a) in momenta.iter() {
        let Some(collider) = colliders.get(*id_a) else {
            non_colliding.push(*id_a);

            continue;
        };

        let mut earliest_collision: Option<(RaycastData, usize)> = None;

        for (id_b, body_b) in colliders.iter() {
            if id_a == id_b {
                continue;
            }

            let world_velocity = if let Some(moment_b) = momenta.get(*id_b) {
                moment_a.velocity - moment_b.velocity
            } else {
                moment_a.velocity
            };

            let b_inverse = transforms[*id_b].inverse();

            let Some(mut collision) = body_b.treecast(
                collider,
                b_inverse * transforms[*id_a],
                b_inverse.rotate_vector(world_velocity * dt),
            ) else {
                continue;
            };

            collision.position = transforms[*id_b].transform_vector(collision.position);
            collision.normal = transforms[*id_b].rotate_vector(collision.normal);

            if let Some((earliest, _)) = &earliest_collision
                && earliest.t > collision.t
            {
                earliest_collision = Some((collision, *id_b))
            } else if earliest_collision.is_none() {
                earliest_collision = Some((collision, *id_b))
            }
        }

        if let Some((collision, id_b)) = earliest_collision {
            colliding.push((collision, *id_a));
        } else {
            non_colliding.push(*id_a);
        }
    }

    for (collision, id_a) in colliding {
        let velocity = momenta[id_a].velocity;
        let mass = momenta[id_a].mass;

        transforms[id_a].position += velocity * dt * (collision.t - 1e-2).min(0.0);

        let velocity_along = collision.normal.dot(velocity);
        let impulse = collision.normal * -(1.0 + COLLISION_RESTITUTION) * velocity_along
            / (1.0 / mass + 1.0 / 50.0);

        momenta[id_a].apply_impulse(impulse);

        let t_remaining = (1.0 - collision.t + 1e-2).max(0.0);

        transforms[id_a].position += velocity * dt * t_remaining;
        momenta[id_a].velocity *= LINEAR_DAMPENING;
    }

    for id_a in non_colliding {
        transforms[id_a].position += momenta[id_a].velocity * dt;
        momenta[id_a].velocity *= LINEAR_DAMPENING;
    }
}
