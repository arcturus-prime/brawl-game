use crate::{
    math::{GeometryTree, SpherecastData, Transform, Vector3},
    utility::SparseSet,
};

pub const COLLISION_RESTITUTION: f32 = 0.5;
pub const LINEAR_DAMPENING: f32 = 0.75;

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

pub fn get_collisions(
    momenta: &SparseSet<Moment>,
    transforms: &SparseSet<Transform>,
    colliders: &SparseSet<GeometryTree>,
    dt: f32,
) -> (Vec<(SpherecastData, usize)>, Vec<usize>) {
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

            let Some(collision) =
                body_b.spherecast(collider.get_bounds_radius(), position, velocity * dt)
            else {
                continue;
            };

            if let Some((earliest, _)) = earliest_collision
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

    (colliding, non_colliding)
}

pub fn step_world(
    colliding: Vec<(SpherecastData, usize)>,
    non_colliding: Vec<usize>,
    momenta: &mut SparseSet<Moment>,
    transforms: &mut SparseSet<Transform>,
    dt: f32,
) {
    if colliding.len() != 0 {
        println!("Collision");
    } else {
        println!("No collision");
    }

    for (collision, id_a) in colliding {
        let velocity = momenta[id_a].velocity;
        let mass = momenta[id_a].mass;

        let velocity_along = collision.normal.dot(velocity);
        let impulse = collision.normal * -(1.0 + COLLISION_RESTITUTION) * velocity_along / mass;

        momenta[id_a].apply_impulse(impulse);

        transforms[id_a].position += momenta[id_a].velocity * dt;
        momenta[id_a].velocity *= LINEAR_DAMPENING;
    }

    for id_a in non_colliding {
        transforms[id_a].position += momenta[id_a].velocity * dt;
        momenta[id_a].velocity *= LINEAR_DAMPENING;
    }
}
