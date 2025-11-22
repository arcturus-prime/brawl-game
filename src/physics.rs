use crate::math::Vector;

const EPSILON: f32 = 1e-6;
const SOLID_INDEX: u32 = u32::MAX - 1;
const EMPTY_INDEX: u32 = u32::MAX - 2;

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub plane: Vector,
    pub front: u32,
    pub back: u32,
}

impl Node {
    pub fn new(plane: Vector, front: u32, back: u32) -> Self {
        Self { plane, front, back }
    }
}

#[inline]
fn plane_normal(plane: Vector) -> Vector {
    Vector::new(plane.x, plane.y, plane.z, 0.0)
}

#[derive(Clone, Debug, Default)]
pub struct CollisionObject {
    nodes: Vec<Node>,
}

impl CollisionObject {
    pub fn invert(&mut self) {
        for node in &mut self.nodes {
            node.front = match node.front {
                SOLID_INDEX => EMPTY_INDEX,
                EMPTY_INDEX => SOLID_INDEX,
                index => index,
            };
            node.back = match node.back {
                SOLID_INDEX => EMPTY_INDEX,
                EMPTY_INDEX => SOLID_INDEX,
                index => index,
            };
        }
    }

    pub fn merge(&mut self, other: CollisionObject) {
        if other.nodes.is_empty() {
            return;
        }

        if self.nodes.is_empty() {
            self.nodes = other.nodes;
            return;
        }

        let other_root = self.nodes.len() as u32;

        for node in &other.nodes {
            let front = if node.front == EMPTY_INDEX || node.front == SOLID_INDEX {
                node.front
            } else {
                node.front + other_root
            };

            let back = if node.back == EMPTY_INDEX || node.back == SOLID_INDEX {
                node.back
            } else {
                node.back + other_root
            };

            self.nodes.push(Node::new(node.plane, front, back));
        }

        for x in &mut self.nodes[..other_root as usize] {
            if x.back == EMPTY_INDEX {
                x.back = other_root;
            }

            if x.front == EMPTY_INDEX {
                x.front = other_root;
            }
        }
    }

    pub fn simplify(&mut self) {}
}

impl CollisionObject {
    #[inline]
    fn distance_from_plane(plane: Vector, point: Vector) -> f32 {
        plane_normal(plane).dot(point) - plane.w
    }

    #[inline]
    fn ray_plane_intersection(plane: Vector, origin: Vector, dir: Vector) -> Option<f32> {
        let denom = plane_normal(plane).dot(dir);

        if denom.abs() < EPSILON {
            return None;
        }

        Some((plane.w - plane_normal(plane).dot(origin)) / denom)
    }

    pub fn raycast(&self, origin: Vector, dir: Vector) -> Option<CollisionData> {
        if self.nodes.is_empty() {
            return None;
        }

        let length = dir.length();
        let dir = dir.normalize();

        let mut stack = Vec::with_capacity(32);
        stack.push((0u32, 0.0, length, Vector::zero()));

        while let Some((index, t_min, t_max, normal)) = stack.pop() {
            if index == SOLID_INDEX {
                return Some(CollisionData {
                    position: origin + dir * t_min,
                    normal,
                });
            }

            if index == EMPTY_INDEX {
                continue;
            }

            let node = &self.nodes[index as usize];

            let start_dist = Self::distance_from_plane(node.plane, origin + dir * t_min);
            let end_dist = Self::distance_from_plane(node.plane, origin + dir * t_max);

            let normal = plane_normal(node.plane);

            if start_dist >= 0.0 && end_dist >= 0.0 {
                stack.push((node.front, t_min, t_max, normal));
            } else if start_dist < 0.0 && end_dist < 0.0 {
                stack.push((node.back, t_min, t_max, normal));
            } else if let Some(t) = Self::ray_plane_intersection(node.plane, origin, dir) {
                let t_split = t.clamp(t_min, t_max);

                if start_dist >= 0.0 {
                    stack.push((node.back, t_split, t_max, -normal));
                    stack.push((node.front, t_min, t_split, normal));
                } else {
                    stack.push((node.front, t_split, t_max, normal));
                    stack.push((node.back, t_min, t_split, -normal));
                }
            }
        }

        None
    }
    pub fn contains(&self, point: Vector) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let mut idx = 0;

        loop {
            if idx == SOLID_INDEX {
                return true;
            }

            if idx == EMPTY_INDEX {
                return false;
            }

            let node = &self.nodes[idx as usize];

            idx = if Self::distance_from_plane(node.plane, point) >= 0.0 {
                node.front
            } else {
                node.back
            };
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CollisionData {
    pub position: Vector,
    pub normal: Vector,
}

impl Default for CollisionData {
    fn default() -> Self {
        Self {
            position: Vector::zero(),
            normal: Vector::new(0.0, 1.0, 0.0, 0.0),
        }
    }
}

#[cfg(test)]
mod complex_contains_tests {
    use super::*;
    use crate::math::Vector;
}
