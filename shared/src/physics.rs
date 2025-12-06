use crate::{
    math::{BoundingBox, VECTOR_X, VECTOR_Y, VECTOR_Z, Vector},
    utility::SparseSet,
};

pub trait Collidable {
    fn support(&self, position: Vector, rotation: Vector, direction: Vector) -> Vector;
}

pub struct ConvexHull {
    points: Vec<Vector>,
}

impl ConvexHull {
    pub fn new(points: Vec<Vector>) -> Self {
        Self { points }
    }
}

impl Collidable for ConvexHull {
    fn support(&self, position1: Vector, rotation1: Vector, direction: Vector) -> Vector {
        todo!()
    }
}

pub struct Cuboid {
    size: Vector,
}

impl Cuboid {
    pub fn new(size: Vector) -> Self {
        Self { size }
    }
}

impl Collidable for Cuboid {
    fn support(&self, position1: Vector, rotation1: Vector, direction: Vector) -> Vector {
        todo!()
    }
}

pub struct Body<T: Collidable> {
    pub collider: T,
    pub rotation: Vector,
    pub position: Vector,
}

impl<T: Collidable> Body<T> {
    pub fn new(collider: T) -> Self {
        Self {
            collider,
            rotation: Vector::identity_quaternion(),
            position: Vector::zero_point(),
        }
    }
}

impl<T: Collidable> Body<T> {
    pub fn get_aabb(&self) -> BoundingBox {
        let directions = [
            VECTOR_X, -VECTOR_X, VECTOR_Y, -VECTOR_Y, VECTOR_Z, -VECTOR_Z,
        ];

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for direction in &directions {
            let point = self
                .collider
                .support(self.position, self.rotation, *direction);

            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
            min_z = min_z.min(point.z);
            max_z = max_z.max(point.z);
        }

        BoundingBox {
            min: Vector::from_point(min_x, min_y, min_z),
            max: Vector::from_point(max_x, max_y, max_z),
        }
    }
}

enum Simplex {
    Point(Vector),
    Line(Vector, Vector),
    Triangle(Vector, Vector, Vector),
    Tetrahedron(Vector, Vector, Vector, Vector),
}

#[derive(Clone, Copy, Debug)]
pub struct CollisionData {
    pub position: Vector,
    pub normal: Vector,
    pub depth: f32,
}

impl<T: Collidable> Body<T> {
    fn gjk<U: Collidable>(&self, other: &Body<U>, mut direction: Vector) -> Option<[Vector; 4]> {
        if direction == Vector::zero_vector() {
            direction = Vector::from_vector(1.0, 0.0, 0.0);
        }

        let support = |direction| {
            self.collider
                .support(self.position, self.rotation, direction)
                - other
                    .collider
                    .support(other.position, other.rotation, -direction)
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
    pub fn collides_with<U: Collidable>(&self, other: &Body<U>) -> Option<CollisionData> {
        let tetra = self.gjk(other, self.position - other.position)?;

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

        let contact_point = self
            .collider
            .support(self.position, self.rotation, best_normal);

        Some(CollisionData {
            position: contact_point,
            normal: best_normal,
            depth: min_dist,
        })
    }
}

pub struct DynamicBody<T: Collidable> {
    pub body: Body<T>,
    pub velocity: Vector,
    pub mass: f32,
}

impl<T: Collidable> DynamicBody<T> {
    pub fn new(collider: T, mass: f32) -> Self {
        Self {
            body: Body::new(collider),
            velocity: Vector::zero_vector(),
            mass,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.body.position += self.velocity * dt;
    }

    pub fn apply_impulse(&mut self, impulse: Vector) {
        self.velocity += impulse / self.mass;
    }
}

pub struct PhysicsWorld<S: Collidable, D: Collidable> {
    pub static_bodies: SparseSet<Body<S>>,
    pub dynamic_bodies: SparseSet<DynamicBody<D>>,
}

impl<S: Collidable, D: Collidable> Default for PhysicsWorld<S, D> {
    fn default() -> Self {
        Self {
            static_bodies: SparseSet::default(),
            dynamic_bodies: SparseSet::default(),
        }
    }
}

impl<S: Collidable, D: Collidable> PhysicsWorld<S, D> {
    pub fn update(&mut self, dt: f32) {
        for (_, body) in self.dynamic_bodies.iter_mut() {
            body.update(dt);
        }

        self.narrow_phase();
    }

    fn narrow_phase(&mut self) {
        let mut collisions = vec![];
        // naive solution for now
        for (id, dynamic_body) in self.dynamic_bodies.iter() {
            for (other_id, static_body) in self.static_bodies.iter() {
                if id == other_id {
                    continue;
                }

                let Some(collision) = dynamic_body.body.collides_with(static_body) else {
                    continue;
                };

                collisions.push((collision, *id, *other_id))
            }

            for (other_id, other_dynamic_body) in self.dynamic_bodies.iter() {
                if id == other_id {
                    continue;
                }

                let Some(collision) = dynamic_body.body.collides_with(&other_dynamic_body.body)
                else {
                    continue;
                };

                collisions.push((collision, *id, *other_id))
            }
        }

        for (collision, id, other_id) in collisions {
            let body = &mut self.dynamic_bodies[id];

            body.body.position -= collision.normal * collision.depth;
            body.velocity -= 2.0 * collision.normal * collision.normal.dot(body.velocity);
        }
    }
}
