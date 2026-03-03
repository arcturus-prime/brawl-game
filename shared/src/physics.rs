use std::{collections::HashSet, f32};

use crate::{
    math::{BoundingBox3, Transform3, Vector3},
    utility::SparseSet,
};

const MAX_GJK_ITERATIONS: usize = 128;
const MAX_EPA_ITERATIONS: usize = 128;
const EPA_TOLERANCE: f32 = 1e-6;
const RESTITUTION: f32 = 0.5;

pub trait Collidable {
    fn support(&self, direction: Vector3) -> Vector3;
    fn aabb(&self, transform: &Transform3) -> BoundingBox3;
}

pub struct ConvexHull {
    points: Vec<Vector3>,
}

impl ConvexHull {
    pub fn new(points: Vec<Vector3>) -> Self {
        Self { points }
    }
}

impl Collidable for ConvexHull {
    fn support(&self, direction: Vector3) -> Vector3 {
        let mut max_dot = f32::NEG_INFINITY;
        let mut max_vertex = Vector3::ZERO;

        for x in &self.points {
            if x.dot(direction) > max_dot {
                max_dot = x.dot(direction);
                max_vertex = *x;
            }
        }

        return max_vertex;
    }

    fn aabb(&self, transform: &Transform3) -> BoundingBox3 {
        BoundingBox3::new(Vector3::ZERO, Vector3::ONE)
    }
}

pub struct Cuboid {
    size: Vector3,
}

impl Cuboid {
    pub fn new(x_size: f32, y_size: f32, z_size: f32) -> Self {
        Self {
            size: Vector3::new(x_size, y_size, z_size),
        }
    }
}

impl Collidable for Cuboid {
    fn support(&self, direction: Vector3) -> Vector3 {
        let vertices = &[
            Vector3::new(self.size.x, self.size.y, self.size.z) / 2.0,
            Vector3::new(self.size.x, -self.size.y, self.size.z) / 2.0,
            Vector3::new(self.size.x, -self.size.y, -self.size.z) / 2.0,
            Vector3::new(self.size.x, self.size.y, -self.size.z) / 2.0,
            Vector3::new(-self.size.x, self.size.y, self.size.z) / 2.0,
            Vector3::new(-self.size.x, -self.size.y, self.size.z) / 2.0,
            Vector3::new(-self.size.x, -self.size.y, -self.size.z) / 2.0,
            Vector3::new(-self.size.x, self.size.y, -self.size.z) / 2.0,
        ];

        let mut max_dot = f32::NEG_INFINITY;
        let mut max_vertex = Vector3::ZERO;
        for x in vertices {
            if direction.dot(*x) > max_dot {
                max_dot = direction.dot(*x);
                max_vertex = *x;
            }
        }

        max_vertex
    }

    fn aabb(&self, transform: &Transform3) -> BoundingBox3 {
        BoundingBox3::new(Vector3::ZERO, Vector3::ONE)
    }
}

enum Simplex {
    Point(Vector3),
    Line(Vector3, Vector3),
    Triangle(Vector3, Vector3, Vector3),
    Tetrahedron(Vector3, Vector3, Vector3, Vector3),
}

fn gjk<T: Collidable, U: Collidable>(
    first_transform: &Transform3,
    first_collider: &T,
    second_transform: &Transform3,
    second_collider: &U,
) -> Option<[Vector3; 4]> {
    let direction = {
        let d = first_transform.position - second_transform.position;

        if d.length_squared() < f32::EPSILON {
            Vector3::X
        } else {
            d
        }
    };

    let support = |direction| {
        first_transform.transform_vector(
            first_collider.support(first_transform.rotation.inverse().rotate_vector(direction)),
        ) - second_transform.transform_vector(
            second_collider.support(
                second_transform
                    .rotation
                    .inverse()
                    .rotate_vector(-direction),
            ),
        )
    };

    let new_point = support(direction);
    let mut simplex = Simplex::Point(new_point);

    for _ in 0..MAX_GJK_ITERATIONS {
        match simplex {
            Simplex::Point(a) => {
                let a_to_origin = -a;
                let new_point = support(a_to_origin);

                if new_point.dot(a_to_origin) < 0.0 {
                    return None;
                }

                simplex = Simplex::Line(a, new_point);
            }
            Simplex::Line(b, a) => {
                let a_to_b = b - a;
                let a_to_origin = -a;

                if a_to_origin.dot(a_to_b) > 0.0 {
                    let mut direction = a_to_b.cross(a_to_origin).cross(a_to_b);

                    if direction.length_squared() < f32::EPSILON {
                        direction = if a_to_b.x.abs() < 0.9 * a_to_b.length() {
                            a_to_b.cross(Vector3::X)
                        } else {
                            a_to_b.cross(Vector3::Y)
                        };
                    }

                    let new_point = support(direction);

                    if new_point.dot(direction) < 0.0 {
                        return None;
                    }

                    simplex = Simplex::Triangle(b, a, new_point);
                } else {
                    simplex = Simplex::Point(a);
                }
            }
            Simplex::Triangle(c, b, a) => {
                let a_to_b = b - a;
                let a_to_c = c - a;
                let a_to_origin = -a;

                let normal = a_to_b.cross(a_to_c);
                if normal.length_squared() < f32::EPSILON {
                    simplex = Simplex::Line(b, a);
                    continue;
                }

                // is the origin outside the triangle on the AC side

                let a_to_c_perp = normal.cross(a_to_c);
                if a_to_c_perp.dot(a_to_origin) > 0.0 {
                    simplex = Simplex::Line(c, a);
                    continue;
                }

                // is the origin outside the triangle on the AB side

                let a_to_b_perp = a_to_b.cross(normal);
                if a_to_b_perp.dot(a_to_origin) > 0.0 {
                    simplex = Simplex::Line(b, a);
                    continue;
                }

                // is the origin above or below the triangle

                if normal.dot(a_to_origin) > 0.0 {
                    let new_point = support(normal);

                    if new_point.dot(normal) < 0.0 {
                        return None;
                    }

                    simplex = Simplex::Tetrahedron(c, b, a, new_point);
                } else {
                    let new_point = support(-normal);

                    if new_point.dot(-normal) < 0.0 {
                        return None;
                    }

                    simplex = Simplex::Tetrahedron(b, c, a, new_point);
                };
            }
            Simplex::Tetrahedron(d, c, b, a) => {
                let a_to_b = b - a;
                let a_to_c = c - a;
                let a_to_d = d - a;
                let a_to_origin = -a;

                let abc_face = a_to_b.cross(a_to_c);
                let acd_face = a_to_c.cross(a_to_d);
                let adb_face = a_to_d.cross(a_to_b);

                if abc_face.dot(a_to_origin) > 0.0 {
                    simplex = Simplex::Triangle(c, b, a);
                    continue;
                }

                if acd_face.dot(a_to_origin) > 0.0 {
                    simplex = Simplex::Triangle(d, c, a);
                    continue;
                }

                if adb_face.dot(a_to_origin) > 0.0 {
                    simplex = Simplex::Triangle(b, d, a);
                    continue;
                }

                return Some([d, c, b, a]);
            }
        }
    }

    None
}

pub struct CollisionData {
    pub normal: Vector3,
    pub depth: f32,
}

pub fn epa<T: Collidable, U: Collidable>(
    first_transform: &Transform3,
    first_collider: &T,
    second_transform: &Transform3,
    second_collider: &U,
) -> Option<CollisionData> {
    let tetra = gjk(
        first_transform,
        first_collider,
        second_transform,
        second_collider,
    )?;

    let support = |direction| {
        first_transform.transform_vector(
            first_collider.support(first_transform.rotation.inverse().rotate_vector(direction)),
        ) - second_transform.transform_vector(
            second_collider.support(
                second_transform
                    .rotation
                    .inverse()
                    .rotate_vector(-direction),
            ),
        )
    };

    let mut vertices: Vec<Vector3> = tetra.to_vec();
    let mut faces: Vec<(usize, usize, usize)> = vec![(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)];

    for i in 0..faces.len() {
        let (ai, bi, ci) = faces[i];
        let (a, b, c) = (vertices[ai], vertices[bi], vertices[ci]);

        let normal = (b - a).cross(c - a);
        if normal.dot(a) < 0.0 {
            faces[i] = (ai, ci, bi);
        }
    }

    for _ in 0..MAX_EPA_ITERATIONS {
        let mut closest_distance = f32::INFINITY;
        let mut closest_normal = Vector3::ZERO;

        for (ai, bi, ci) in &faces {
            let (a, b, c) = (vertices[*ai], vertices[*bi], vertices[*ci]);

            let normal = (b - a).cross(c - a).normalize();
            let distance = normal.dot(a);

            if distance < closest_distance {
                closest_distance = distance;
                closest_normal = normal;
            }
        }

        let new_point = support(closest_normal);
        let new_distance = closest_normal.dot(new_point);

        if new_distance - closest_distance < EPA_TOLERANCE {
            return Some(CollisionData {
                normal: closest_normal,
                depth: closest_distance,
            });
        }

        let new_index = vertices.len();
        vertices.push(new_point);

        let mut horizon_edges: HashSet<(usize, usize)> = HashSet::new();

        let mut i = 0;
        while i < faces.len() {
            let (ai, bi, ci) = faces[i];
            let (a, b, c) = (vertices[ai], vertices[bi], vertices[ci]);
            let normal = (b - a).cross(c - a);

            if normal.dot(new_point - a) > 0.0 {
                for (ei, ej) in [(ai, bi), (bi, ci), (ci, ai)] {
                    if !horizon_edges.remove(&(ej, ei)) {
                        horizon_edges.insert((ei, ej));
                    }
                }

                faces.swap_remove(i);
            } else {
                i += 1;
            }
        }

        for (ei, ej) in horizon_edges {
            faces.push((ei, ej, new_index));
        }
    }

    let mut closest_distance = f32::INFINITY;
    let mut closest_normal = Vector3::ZERO;

    for &(ai, bi, ci) in &faces {
        let (a, b, c) = (vertices[ai], vertices[bi], vertices[ci]);
        let normal = (b - a).cross(c - a).normalize();
        let distance = normal.dot(a);

        if distance < closest_distance {
            closest_distance = distance;
            closest_normal = normal;
        }
    }

    Some(CollisionData {
        normal: closest_normal,
        depth: closest_distance,
    })
}

pub struct Moment {
    pub velocity: Vector3,
    pub mass: f32,
    pub dampening: f32,
}

impl Moment {
    pub fn new(mass: f32, dampening: f32) -> Self {
        Self {
            velocity: Vector3::ZERO,
            mass,
            dampening,
        }
    }

    pub fn update(&mut self, body: &mut Transform3, dt: f32) {
        body.position += self.velocity * dt;
        self.velocity *= self.dampening;
    }

    pub fn apply_impulse(&mut self, impulse: Vector3) {
        self.velocity += impulse / self.mass;
    }
}

pub fn step_world<T: Collidable>(
    momenta: &mut SparseSet<Moment>,
    transforms: &mut SparseSet<Transform3>,
    colliders: &SparseSet<T>,
    dt: f32,
) {
    for (id, moment) in momenta.iter_mut() {
        moment.update(&mut transforms[*id], dt);
    }

    // naive solution for now
    for (id_a, body_a) in colliders.iter() {
        for (id_b, body_b) in colliders.iter() {
            if id_a == id_b {
                continue;
            }

            let Some(collision) = epa(&transforms[*id_a], body_a, &transforms[*id_b], body_b)
            else {
                continue;
            };

            if momenta.get(*id_a).is_some() && momenta.get(*id_b).is_some() {
                let mass_a = momenta[*id_a].mass;
                let mass_b = momenta[*id_b].mass;
                let total_mass = mass_a + mass_b;

                let velocity_a = momenta[*id_a].velocity;
                let velocity_b = momenta[*id_b].velocity;

                let relative_velocity = velocity_a - velocity_b;
                let velocity_along_normal = relative_velocity.dot(collision.normal);

                let j =
                    -(1.0 + RESTITUTION) * velocity_along_normal / (1.0 / mass_a + 1.0 / mass_b);

                momenta[*id_a].velocity += j * collision.normal / mass_a;
                momenta[*id_b].velocity -= j * collision.normal / mass_b;

                transforms[*id_a].position -=
                    collision.normal * collision.depth * mass_b / total_mass;
                transforms[*id_b].position +=
                    collision.normal * collision.depth * mass_a / total_mass;
            } else if let Some(moment) = momenta.get_mut(*id_a) {
                moment.velocity -= collision.normal.dot(moment.velocity) * collision.normal;
                transforms[*id_a].position -= collision.normal * collision.depth;
            } else if let Some(moment) = momenta.get_mut(*id_b) {
                moment.velocity -= collision.normal.dot(moment.velocity) * collision.normal;
                transforms[*id_b].position += collision.normal * collision.depth;
            }
        }
    }
}
