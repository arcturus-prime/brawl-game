use crate::math::{Transform, Vector};

pub trait Collider {
    fn support(&self, direction: Vector) -> Vector;
    fn position(&self) -> Vector;
}

enum Simplex {
    Point(Vector),
    Line(Vector, Vector),
    Triangle(Vector, Vector, Vector),
    Tetrahedron(Vector, Vector, Vector, Vector),
}

pub fn gjk<T: Collider, U: Collider>(
    collider1: &T,
    collider2: &U,
    mut direction: Vector,
) -> Option<[Vector; 4]> {
    if direction == Vector::zero_vector() {
        direction = Vector::from_vector(1.0, 0.0, 0.0);
    }

    let new_point = collider1.support(direction) - collider2.support(-direction);
    let mut simplex = Simplex::Point(new_point);

    loop {
        match simplex {
            Simplex::Point(a) => {
                let new_point = collider1.support(a) - collider2.support(-a);

                if new_point.dot(-a) < 0.0 {
                    return None;
                }

                simplex = Simplex::Line(a, new_point);
            }
            Simplex::Line(b, a) => {
                let a_to_b = b - a;

                if a.dot(a_to_b) > 0.0 {
                    let direction = a_to_b.cross(a).cross(a_to_b);
                    let new_point = collider1.support(direction) - collider2.support(-direction);

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

                let new_point = collider1.support(direction) - collider2.support(-direction);

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
pub fn collision_test<T: Collider, U: Collider>(
    collider1: &T,
    collider2: &U,
) -> Option<CollisionData> {
    let tetra = gjk(
        collider1,
        collider2,
        collider1.position() - collider2.position(),
    )?;

    let faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)];

    let mut min_dist = f32::MAX;
    let mut best_normal = Vector::zero_vector();

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

    let contact_point = collider1.support(best_normal);

    Some(CollisionData {
        position: contact_point,
        normal: best_normal,
        depth: min_dist,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct CollisionData {
    pub position: Vector,
    pub normal: Vector,
    pub depth: f32,
}

#[derive(Clone)]
pub struct Moving {
    pub velocity: Vector,
    pub mass: f32,
}

impl Moving {
    pub fn apply_impulse(&mut self, impulse: Vector) {
        self.velocity += impulse / self.mass;
    }
}

pub struct Sphere {
    pub position: Vector,
    pub radius: f32,
}

impl Collider for Sphere {
    fn support(&self, direction: Vector) -> Vector {
        // should never happen
        if direction == Vector::zero_vector() {
            return self.position;
        }

        self.position + (direction.normalize() * self.radius)
    }

    fn position(&self) -> Vector {
        self.position
    }
}

pub struct Cuboid {
    pub transform: Transform,
    pub half_extents: Vector,
}

impl Collider for Cuboid {
    fn support(&self, direction: Vector) -> Vector {
        let sign_x = if direction.x >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if direction.y >= 0.0 { 1.0 } else { -1.0 };
        let sign_z = if direction.z >= 0.0 { 1.0 } else { -1.0 };

        let support_point_local = Vector::from_vector(
            self.half_extents.x * sign_x,
            self.half_extents.y * sign_y,
            self.half_extents.z * sign_z,
        );

        self.transform.w + support_point_local
    }

    fn position(&self) -> Vector {
        self.transform.w
    }
}

pub enum BvhNode {
    Branch {
        children: [u32; 2],
        bounds: [Vector; 2],
    },
    Leaf {
        collider: Box<dyn Collider>,
    },
}

pub struct CollisionWorld {
    pub map: Vec<BvhNode>,
}
