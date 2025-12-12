use std::f32;

use crate::{
    math::{BoundingBox, Transform, VECTOR_X, VECTOR_Y, VECTOR_Z, Vector3},
    utility::SparseSet,
};

const MAX_GJK_ITERATIONS: usize = 128;
const MAX_EPA_ITERATIONS: usize = 128;
const EPA_TOLERANCE: f32 = 1e-6;

pub trait Collidable {
    fn support(&self, direction: Vector3) -> Vector3;
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
        let mut max_vertex = Vector3::zero();

        for x in &self.points {
            if x.dot(direction) > max_dot {
                max_dot = x.dot(direction);
                max_vertex = *x;
            }
        }

        return max_vertex;
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
        let mut max_vertex = Vector3::zero();
        for x in vertices {
            if direction.dot(*x) > max_dot {
                max_dot = direction.dot(*x);
                max_vertex = *x;
            }
        }

        max_vertex
    }
}

pub fn get_aabb<T: Collidable>(transform: &Transform, collider: &T) -> BoundingBox {
    let directions = [
        VECTOR_X, -VECTOR_X, VECTOR_Y, -VECTOR_Y, VECTOR_Z, -VECTOR_Z,
    ];

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    for direction in directions {
        let point = transform.transform_vector(
            collider.support(transform.rotation.inverse().rotate_vector(-direction)),
        );

        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
        min_z = min_z.min(point.z);
        max_z = max_z.max(point.z);
    }

    BoundingBox::new(
        Vector3::new(min_x, min_y, min_z),
        Vector3::new(max_x, max_y, max_z),
    )
}

enum Simplex {
    Point(Vector3),
    Line(Vector3, Vector3),
    Triangle(Vector3, Vector3, Vector3),
    Tetrahedron(Vector3, Vector3, Vector3, Vector3),
}

fn gjk<T: Collidable, U: Collidable>(
    first_transform: &Transform,
    first_collider: &T,
    second_transform: &Transform,
    second_collider: &U,
) -> Option<[Vector3; 4]> {
    let direction = first_transform.position - second_transform.position;

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
                    let direction = a_to_b.cross(a_to_origin).cross(a_to_b);

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

pub fn collides_with<T: Collidable, U: Collidable>(
    first_transform: &Transform,
    first_collider: &T,
    second_transform: &Transform,
    second_collider: &U,
) -> Option<CollisionData> {
    let tetra = gjk(
        first_transform,
        first_collider,
        second_transform,
        second_collider,
    )?;

    let mut faces = vec![
        (tetra[0], tetra[1], tetra[2]),
        (tetra[0], tetra[3], tetra[1]),
        (tetra[0], tetra[2], tetra[3]),
        (tetra[1], tetra[3], tetra[2]),
    ];

    for i in 0..faces.len() {
        let (a, b, c) = faces[i];
        let normal = (b - a).cross(c - a);

        if normal.dot(a) < 0.0 {
            faces[i] = (a, c, b);
        }
    }

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

    for _ in 0..MAX_EPA_ITERATIONS {
        let mut closest_face_index = 0;
        let mut closest_normal = VECTOR_X;
        let mut closest_distance = f32::INFINITY;

        for (x, i) in faces.iter().zip(0..) {
            let normal = (x.1 - x.0).cross(x.2 - x.0).normalize();
            let distance = x.0.dot(normal);

            if distance < closest_distance {
                closest_distance = distance;
                closest_normal = normal;
                closest_face_index = i;
            }
        }

        let new_point = support(closest_normal);
        let distance = new_point.dot(closest_normal);

        if distance - closest_distance < EPA_TOLERANCE {
            return Some(CollisionData {
                normal: closest_normal,
                depth: closest_distance,
            });
        }

        let (a, b, c) = faces[closest_face_index];
        faces.swap_remove(closest_face_index);

        faces.push((a, b, new_point));
        faces.push((b, c, new_point));
        faces.push((c, a, new_point));

        for i in (faces.len() - 3)..faces.len() {
            let (a, b, c) = faces[i];
            let normal = (b - a).cross(c - a);

            if normal.dot(a) < 0.0 {
                faces[i] = (a, c, b);
            }
        }
    }

    None
}

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

pub fn resolve_collisions<T: Collidable>(
    momenta: &mut SparseSet<Moment>,
    transforms: &mut SparseSet<Transform>,
    colliders: &SparseSet<T>,
) {
    // naive solution for now
    for (id_a, body_a) in colliders.iter() {
        for (id_b, body_b) in colliders.iter() {
            if id_a == id_b {
                continue;
            }

            let Some(collision) =
                collides_with(&transforms[*id_a], body_a, &transforms[*id_b], body_b)
            else {
                continue;
            };

            if momenta.get(*id_a).is_some() && momenta.get(*id_b).is_some() {
                let mass_a = momenta[*id_a].mass;
                let mass_b = momenta[*id_b].mass;

                let velocity_a = momenta[*id_a].velocity;
                let velocity_b = momenta[*id_b].velocity;

                let total_mass = mass_a + mass_b;

                momenta[*id_a].velocity -= collision.normal.dot(velocity_a) * collision.normal;
                momenta[*id_b].velocity -= collision.normal.dot(velocity_b) * collision.normal;

                transforms[*id_a].position -=
                    collision.normal * collision.depth * mass_a / total_mass;
                transforms[*id_b].position +=
                    collision.normal * collision.depth * mass_b / total_mass;
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
