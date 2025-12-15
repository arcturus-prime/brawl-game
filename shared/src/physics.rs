use crate::{
    math::{GeometryTree, SpherecastData, Transform, Vector3},
    utility::SparseSet,
};

pub const COLLISION_RESTITUTION: f32 = 0.5;
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

    pub fn update(&mut self, body: &mut Transform, dt: f32) {
        body.position += self.velocity * dt;
    }

    pub fn apply_impulse(&mut self, impulse: Vector3) {
        self.velocity += impulse / self.mass;
    }
}

pub fn step_world(
    colliders: &SparseSet<GeometryTree>,
    momenta: &mut SparseSet<Moment>,
    transforms: &mut SparseSet<Transform>,
    dt: f32,
) {
    let mut colliding = vec![];
    let mut non_colliding = vec![];

    for (id_a, moment_a) in momenta.iter() {
        let Some(collider) = colliders.get(*id_a) else {
            non_colliding.push(*id_a);

            continue;
        };

        let mut earliest_collision: Option<(SpherecastData, usize)> = None;

        for (id_b, body_b) in colliders.iter() {
            if id_a == id_b {
                continue;
            }

            let velocity = if let Some(moment_b) = momenta.get(*id_b) {
                moment_a.velocity - moment_b.velocity
            } else {
                moment_a.velocity
            };

            let position = transforms[*id_a].position - transforms[*id_b].position;

            let Some(mut collision) =
                body_b.spherecast(collider.get_bounds_radius(), position, velocity * dt)
            else {
                continue;
            };

            collision.position += transforms[*id_b].position;

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

        let velocity_along = collision.normal.dot(velocity);
        let impulse = collision.normal * -(1.0 + COLLISION_RESTITUTION) * velocity_along / mass;

        momenta[id_a].apply_impulse(impulse);
        transforms[id_a].position = collision.position;
    }

    for id_a in non_colliding {
        transforms[id_a].position += momenta[id_a].velocity * dt;
        momenta[id_a].velocity *= LINEAR_DAMPENING;
    }
}
