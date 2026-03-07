use std::{collections::HashSet, f32};

use crate::{
    math::{BoundingBox3, Transform3, Vector3},
    utility::SparseSet,
};

const MAX_GJK_ITERATIONS: usize = 128;
const MAX_EPA_ITERATIONS: usize = 128;
const EPA_TOLERANCE: f32 = 1e-6;
const RESTITUTION: f32 = 0.5;

#[derive(Clone)]
pub struct Collider {
    points: Vec<Vector3>,
}

impl Collider {
    pub fn new(points: Vec<Vector3>) -> Self {
        Self { points }
    }
}

impl Collider {
    pub fn support(&self, direction: Vector3) -> Vector3 {
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

    pub fn transform(&mut self, transform: &Transform3) {
        for x in &mut self.points {
            *x = transform.transform_vector(*x)
        }
    }

    pub fn aabb(&self) -> BoundingBox3 {
        let mut min = Vector3::ONE * f32::INFINITY;
        let mut max = Vector3::ONE * f32::NEG_INFINITY;

        for point in &self.points {
            min = Vector3::new(min.x.min(point.x), min.y.min(point.y), min.z.min(point.z));
            max = Vector3::new(max.x.max(point.x), max.y.max(point.y), max.z.max(point.z));
        }

        BoundingBox3::new(min, max)
    }
}

enum Simplex {
    Point(Vector3),
    Line(Vector3, Vector3),
    Triangle(Vector3, Vector3, Vector3),
    Tetrahedron(Vector3, Vector3, Vector3, Vector3),
}

fn gjk(
    first_collider: &Collider,
    second_collider: &Collider,
    initial_direction: Vector3,
) -> Option<[Vector3; 4]> {
    let support =
        |direction| first_collider.support(direction) - second_collider.support(-direction);

    let new_point = support(initial_direction);
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

pub fn epa(
    first_collider: &Collider,
    second_collider: &Collider,
    initial_direction: Vector3,
) -> Option<CollisionData> {
    let tetra = gjk(first_collider, second_collider, initial_direction)?;

    let support =
        |direction| first_collider.support(direction) - second_collider.support(-direction);

    let mut vertices: Vec<Vector3> = tetra.to_vec();
    let mut faces: Vec<(usize, usize, usize)> = vec![(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)];

    for i in 0..faces.len() {
        let (a_index, b_index, c_index) = faces[i];
        let (a, b, c) = (vertices[a_index], vertices[b_index], vertices[c_index]);

        let normal = (b - a).cross(c - a);
        if normal.dot(a) < 0.0 {
            faces[i] = (a_index, c_index, b_index);
        }
    }

    for _ in 0..MAX_EPA_ITERATIONS {
        let mut closest_distance = f32::INFINITY;
        let mut closest_normal = Vector3::ZERO;

        for (a_index, b_index, c_index) in &faces {
            let (a, b, c) = (vertices[*a_index], vertices[*b_index], vertices[*c_index]);

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
            let (a_index, b_index, c_index) = faces[i];
            let (a, b, c) = (vertices[a_index], vertices[b_index], vertices[c_index]);
            let normal = (b - a).cross(c - a);

            if normal.dot(new_point - a) > 0.0 {
                for (edge_index_a, edge_index_b) in
                    [(a_index, b_index), (b_index, c_index), (c_index, a_index)]
                {
                    if !horizon_edges.remove(&(edge_index_b, edge_index_a)) {
                        horizon_edges.insert((edge_index_a, edge_index_b));
                    }
                }

                faces.swap_remove(i);
            } else {
                i += 1;
            }
        }

        for (edge_index_a, edge_index_b) in horizon_edges {
            faces.push((edge_index_a, edge_index_b, new_index));
        }
    }

    let mut closest_distance = f32::INFINITY;
    let mut closest_normal = Vector3::ZERO;

    for &(a_index, b_index, c_index) in &faces {
        let (a, b, c) = (vertices[a_index], vertices[b_index], vertices[c_index]);
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

pub fn step_world(
    momenta: &mut SparseSet<Moment>,
    transforms: &mut SparseSet<Transform3>,
    colliders: &SparseSet<Collider>,
    dt: f32,
) {
    for (id, moment) in momenta.iter_mut() {
        moment.update(&mut transforms[*id], dt);
    }

    let mut world_colliders = SparseSet::default();
    let mut world_aabbs = SparseSet::default();
    for (id, data) in colliders.iter() {
        let mut world_collider = data.clone();
        world_collider.transform(&transforms[*id]);

        world_aabbs.insert(*id, world_collider.aabb());
        world_colliders.insert(*id, world_collider);
    }

    // naive solution for now
    for (id_a, body_a) in world_colliders.iter() {
        for (id_b, body_b) in world_colliders.iter() {
            if id_a == id_b {
                continue;
            }

            let direction = transforms[*id_a].position - transforms[*id_b].position;
            let Some(collision) = epa(body_a, body_b, direction) else {
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
