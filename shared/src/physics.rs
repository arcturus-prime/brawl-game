use crate::{
    math::{BoundingBox, Transform, VECTOR_X, VECTOR_Y, VECTOR_Z, Vector3},
    utility::SparseSet,
};

pub trait Collidable {
    fn support(&self, direction: Vector3) -> Vector3;
}

impl Collidable for &dyn Collidable {
    fn support(&self, direction: Vector3) -> Vector3 {
        (*self).support(direction)
    }
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
        todo!()
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
        todo!()
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

#[derive(Clone, Copy, Debug)]
pub struct CollisionData {
    pub position: Vector3,
    pub normal: Vector3,
    pub depth: f32,
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

    loop {
        match simplex {
            Simplex::Point(a) => {
                let new_point = support(a);

                if new_point.dot(-a) < 0.0 {
                    return None;
                }

                simplex = Simplex::Line(a, new_point);
            }
            Simplex::Line(b, a) => {
                let a_to_b = b - a;

                if a.dot(a_to_b) > 0.0 {
                    let direction = a_to_b.cross(a).cross(a_to_b);
                    let new_point = support(direction);

                    if new_point.dot(-direction) < 0.0 {
                        return None;
                    }

                    simplex = Simplex::Triangle(b, a, new_point);
                } else {
                    simplex = Simplex::Point(-a);
                }
            }
            Simplex::Triangle(c, b, a) => {
                let a_to_b = a - b;
                let a_to_c = a - c;
                let a_to_origin = -a;

                let normal = a_to_b.cross(a_to_c);

                // is the origin outside the triangle on the AB side

                let a_to_b_perp = a_to_c.cross(a_to_b).cross(a_to_b);
                if a_to_b_perp.dot(a_to_origin) > 0.0 {
                    simplex = Simplex::Line(b, a);
                    continue;
                }

                // is the origin outside the triangle on the AC side

                let a_to_c_perp = a_to_b.cross(a_to_c).cross(a_to_c);
                if a_to_c_perp.dot(a_to_origin) > 0.0 {
                    simplex = Simplex::Line(c, a);
                    continue;
                }

                // is the origin above or below the triangle

                let direction = if normal.dot(a_to_origin) > 0.0 {
                    normal
                } else {
                    -normal
                };

                let new_point = support(direction);

                simplex = Simplex::Tetrahedron(c, b, a, new_point);
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
}

// note that this function is an approximation that will work better when penetration depths are minimal
// may replace later with proper EPA
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

    let faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)];

    let mut min_dist = f32::MAX;
    let mut best_normal = Vector3::zero();

    for (i, j, k) in faces {
        let a = tetra[i];
        let b = tetra[j];
        let c = tetra[k];

        let ab = b - a;
        let ac = c - a;

        let mut normal = ab.cross(ac);
        let mut dist = normal.dot(a);

        if dist < 0.0 {
            normal = -normal;
            dist = -dist;
        }

        if dist < min_dist {
            min_dist = dist;
            best_normal = normal;
        }
    }

    let contact_point = first_collider.support(first_transform.rotation.rotate_vector(best_normal));

    Some(CollisionData {
        position: contact_point,
        normal: best_normal,
        depth: min_dist,
    })
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

pub fn get_collisions<T: Collidable>(
    transforms: &SparseSet<Transform>,
    colliders: &SparseSet<T>,
) -> Vec<(CollisionData, usize, usize)> {
    let mut collisions = vec![];
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

            collisions.push((collision, *id_a, *id_b))
        }
    }

    return collisions;
}

pub fn resolve_collisions<T: Collidable>(
    transforms: &mut SparseSet<Transform>,
    collisions: Vec<(CollisionData, usize, usize)>,
) {
    for (collision, id_a, id_b) in collisions {
        transforms[id_a].position += collision.normal * collision.depth / 2.0;
        transforms[id_b].position += collision.normal * -collision.depth / 2.0;
    }
}
