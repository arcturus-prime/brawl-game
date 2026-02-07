use crate::math::{BoundingBox3, Plane3, PlaneSide, Transform3, Vector3};
use obj::{Group, IndexTuple, Mtl, Obj, ObjError, ObjMaterial, Object, SimplePolygon};

use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaycastData {
    pub contents: HalfspaceContents,
    pub position: Vector3,
    pub normal: Vector3,
    pub t: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfspaceContents(pub u32);

impl HalfspaceContents {
    pub fn solid() -> Self {
        Self(0xFFFFFFFE)
    }

    pub fn empty() -> Self {
        Self(0xFFFFFFFF)
    }

    pub fn index(index: u32) -> Self {
        if index >= 0xFFFFFFFE {
            panic!("Index must be less than 0xFFFFFFFE");
        }

        Self(index)
    }

    pub fn invert(&mut self) {
        if self.is_solid() {
            *self = Self::empty()
        } else if self.is_empty() {
            *self = Self::solid()
        }
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0xFFFFFFFF
    }

    pub fn is_solid(self) -> bool {
        self.0 == 0xFFFFFFFE
    }

    pub fn is_index(self) -> bool {
        !self.is_empty() && !self.is_solid()
    }

    pub fn get_index(self) -> Option<u32> {
        if self.is_empty() || self.is_solid() {
            return None;
        }

        Some(self.0)
    }
}

#[derive(Clone, Debug, Default, Copy)]
pub struct HalfspaceMetadata(pub u32, pub u32);

impl HalfspaceMetadata {
    pub fn new() -> Self {
        Self(0, 0)
    }
}

#[derive(Clone, Debug)]
pub struct Halfspace {
    pub plane: Plane3,
    pub negative: HalfspaceContents,
    pub positive: HalfspaceContents,
    pub metadata: HalfspaceMetadata,
}

impl Halfspace {
    pub fn new(
        plane: Plane3,
        positive: HalfspaceContents,
        negative: HalfspaceContents,
        metadata: HalfspaceMetadata,
    ) -> Self {
        Self {
            plane,
            positive,
            negative,
            metadata,
        }
    }

    pub fn invert(&mut self) {
        self.plane = self.plane.flip();
        self.positive.invert();
        self.negative.invert();

        std::mem::swap(&mut self.positive, &mut self.negative);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("An error occured while loading the obj")]
    ObjError(#[from] ObjError),
}

#[derive(Clone, Debug)]
pub struct GeometryTree {
    nodes: Vec<Halfspace>,
    bounds: BoundingBox3,
}

impl GeometryTree {
    pub fn nodes(&self) -> &Vec<Halfspace> {
        &self.nodes
    }

    pub fn from_cube(size_x: f32, size_y: f32, size_z: f32, metadata: HalfspaceMetadata) -> Self {
        let half_x = size_x / 2.0;
        let half_y = size_y / 2.0;
        let half_z = size_z / 2.0;

        let nodes = vec![
            Halfspace::new(
                Plane3::new(Vector3::X, half_x),
                HalfspaceContents::empty(),
                HalfspaceContents::index(1),
                metadata,
            ),
            Halfspace::new(
                Plane3::new(-Vector3::X, half_x),
                HalfspaceContents::empty(),
                HalfspaceContents::index(2),
                metadata,
            ),
            Halfspace::new(
                Plane3::new(Vector3::Y, half_y),
                HalfspaceContents::empty(),
                HalfspaceContents::index(3),
                metadata,
            ),
            Halfspace::new(
                Plane3::new(-Vector3::Y, half_y),
                HalfspaceContents::empty(),
                HalfspaceContents::index(4),
                metadata,
            ),
            Halfspace::new(
                Plane3::new(Vector3::Z, half_z),
                HalfspaceContents::empty(),
                HalfspaceContents::index(5),
                metadata,
            ),
            Halfspace::new(
                Plane3::new(-Vector3::Z, half_z),
                HalfspaceContents::empty(),
                HalfspaceContents::solid(),
                metadata,
            ),
        ];

        let bounds = BoundingBox3::new(
            Vector3::new(-half_x, -half_y, -half_z),
            Vector3::new(half_x, half_y, half_z),
        );

        GeometryTree { nodes, bounds }
    }

    pub fn get_bounds(&self) -> &BoundingBox3 {
        &self.bounds
    }

    pub fn transform(&mut self, transform: Transform3) {
        for x in &mut self.nodes {
            x.plane.normal = transform.rotation.rotate_vector(x.plane.normal);
            x.plane.distance += transform.position.dot(x.plane.normal);
        }

        let corners = [
            self.bounds.min,
            Vector3::new(self.bounds.max.x, self.bounds.min.y, self.bounds.min.z),
            Vector3::new(self.bounds.min.x, self.bounds.max.y, self.bounds.min.z),
            Vector3::new(self.bounds.min.x, self.bounds.min.y, self.bounds.max.z),
            Vector3::new(self.bounds.min.x, self.bounds.max.y, self.bounds.max.z),
            Vector3::new(self.bounds.max.x, self.bounds.min.y, self.bounds.max.z),
            Vector3::new(self.bounds.max.x, self.bounds.max.y, self.bounds.min.z),
            self.bounds.max,
        ];

        let mut new_min = Vector3::one() * f32::INFINITY;
        let mut new_max = Vector3::one() * f32::NEG_INFINITY;

        for corner in corners {
            let point = transform.transform_vector(corner);

            new_min = Vector3::new(
                new_min.x.min(point.x),
                new_min.y.min(point.y),
                new_min.z.min(point.z),
            );
            new_max = Vector3::new(
                new_max.x.max(point.x),
                new_max.y.max(point.y),
                new_max.z.max(point.z),
            );
        }

        self.bounds = BoundingBox3::new(new_min, new_max);
    }

    pub fn invert(&mut self) {
        for x in &mut self.nodes {
            x.invert();
        }

        self.bounds = BoundingBox3::new(
            Vector3::one() * f32::NEG_INFINITY,
            Vector3::one() * f32::INFINITY,
        );
    }

    /// Perform a CSG union with another Tree
    pub fn union(&mut self, mut tree: GeometryTree) {
        let root = self.nodes.len() as u32;

        for node in &mut tree.nodes {
            if node.negative.is_index() {
                node.negative.0 += root;
            }

            if node.positive.is_index() {
                node.positive.0 += root;
            }
        }

        for x in &mut self.nodes {
            if x.positive.is_empty() {
                x.positive = HalfspaceContents::index(root);
            }

            if x.negative.is_empty() {
                x.negative = HalfspaceContents::index(root);
            }
        }

        self.nodes.append(&mut tree.nodes);
        self.bounds = self.bounds.union(&tree.bounds);
    }

    /// Perform a CSG intersection with another Tree
    pub fn intersection(&mut self, mut tree: GeometryTree) {
        let root = self.nodes.len() as u32;

        for node in &mut tree.nodes {
            if node.negative.is_index() {
                node.negative.0 += root;
            }

            if node.positive.is_index() {
                node.positive.0 += root;
            }
        }

        for x in &mut self.nodes {
            if x.positive.is_solid() {
                x.positive = HalfspaceContents::index(root);
            }

            if x.negative.is_solid() {
                x.negative = HalfspaceContents::index(root);
            }
        }

        self.nodes.append(&mut tree.nodes);
        self.bounds = self.bounds.intersection(&tree.bounds);
    }

    /// Cast a ray into a CSG shape
    pub fn raycast(&self, origin: Vector3, dir: Vector3) -> Option<RaycastData> {
        if self.nodes.is_empty() {
            return None;
        }

        let length = dir.length();
        let dir = dir.normalize();

        let mut stack = Vec::with_capacity(32);
        stack.push((HalfspaceContents::index(0), 0.0, length, None));

        while let Some((contents, t_min, t_max, normal)) = stack.pop() {
            if contents.is_solid()
                && let Some(normal) = normal
            {
                return Some(RaycastData {
                    contents: contents,
                    position: origin + dir * t_min,
                    normal,
                    t: t_min,
                });
            } else if contents.is_solid() {
                continue;
            }

            if contents.is_empty() {
                continue;
            }

            let node = self.nodes[contents.get_index().unwrap() as usize].clone();

            let start_dist = node.plane.distance_to_point(origin + dir * t_min);
            let end_dist = node.plane.distance_to_point(origin + dir * t_max);

            if start_dist >= 0.0 && end_dist >= 0.0 {
                stack.push((node.positive, t_min, t_max, normal));
            } else if start_dist < 0.0 && end_dist < 0.0 {
                stack.push((node.negative, t_min, t_max, normal));
            } else if let Some(result) = node.plane.raycast(origin, dir) {
                if start_dist >= 0.0 {
                    stack.push((node.negative, result.1, t_max, Some(node.plane.normal)));
                    stack.push((node.positive, t_min, result.1, normal));
                } else {
                    stack.push((node.positive, result.1, t_max, Some(node.plane.normal)));
                    stack.push((node.negative, t_min, result.1, normal));
                }
            }
        }

        None
    }

    pub fn treecast(
        &self,
        other: &GeometryTree,
        transform: Transform3,
        displacement: Vector3,
    ) -> Option<RaycastData> {
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return None;
        }

        let mut new_min = Vector3::one() * f32::INFINITY;
        let mut new_max = Vector3::one() * f32::NEG_INFINITY;

        let corners = [
            other.bounds.min,
            Vector3::new(other.bounds.max.x, other.bounds.min.y, other.bounds.min.z),
            Vector3::new(other.bounds.min.x, other.bounds.max.y, other.bounds.min.z),
            Vector3::new(other.bounds.min.x, other.bounds.min.y, other.bounds.max.z),
            Vector3::new(other.bounds.min.x, other.bounds.max.y, other.bounds.max.z),
            Vector3::new(other.bounds.max.x, other.bounds.min.y, other.bounds.max.z),
            Vector3::new(other.bounds.max.x, other.bounds.max.y, other.bounds.min.z),
            other.bounds.max,
            other.bounds.min + displacement,
            Vector3::new(other.bounds.max.x, other.bounds.min.y, other.bounds.min.z) + displacement,
            Vector3::new(other.bounds.min.x, other.bounds.max.y, other.bounds.min.z) + displacement,
            Vector3::new(other.bounds.min.x, other.bounds.min.y, other.bounds.max.z) + displacement,
            Vector3::new(other.bounds.min.x, other.bounds.max.y, other.bounds.max.z) + displacement,
            Vector3::new(other.bounds.max.x, other.bounds.min.y, other.bounds.max.z) + displacement,
            Vector3::new(other.bounds.max.x, other.bounds.max.y, other.bounds.min.z) + displacement,
            other.bounds.max + displacement,
        ];

        for corner in corners {
            let point = transform.transform_vector(corner);

            new_min = Vector3::new(
                new_min.x.min(point.x),
                new_min.y.min(point.y),
                new_min.z.min(point.z),
            );
            new_max = Vector3::new(
                new_max.x.max(point.x),
                new_max.y.max(point.y),
                new_max.z.max(point.z),
            );
        }

        let bounds = BoundingBox3::new(new_min, new_max);

        if !self.bounds.intersects(&bounds) {
            return None;
        }

        let intersection = self.bounds.intersection(&bounds);
        let starting_point = (intersection.max + intersection.min) / 2.0;

        enum Command {
            PopContraint,
            PushContraint(Plane3, bool, bool),
            ProcessNode(HalfspaceContents, bool),
        }

        let mut stack = vec![Command::ProcessNode(HalfspaceContents::index(0), false)];
        let mut path = vec![];

        let mut best_hit: Option<RaycastData> = None;

        while let Some(command) = stack.pop() {
            match command {
                Command::PopContraint => {
                    path.pop();
                }
                Command::PushContraint(plane, is_dynamic, is_inverted) => {
                    path.push((plane, is_dynamic, is_inverted));
                }
                Command::ProcessNode(contents, is_second_tree) => {
                    if contents.is_solid() && is_second_tree {
                        if let Some(hit) =
                            Self::solve_spacetime_constraints(&path, starting_point, displacement)
                            && best_hit.map_or(true, |h| hit.t < h.t)
                        {
                            best_hit = Some(hit)
                        }

                        continue;
                    } else if contents.is_solid() {
                        stack.push(Command::ProcessNode(HalfspaceContents::index(0), true));
                        continue;
                    }

                    if contents.is_empty() {
                        continue;
                    }

                    let node = if is_second_tree {
                        let mut node = other.nodes[contents.get_index().unwrap() as usize].clone();

                        node.plane.normal = transform.rotation.rotate_vector(node.plane.normal);
                        node.plane.distance += node.plane.normal.dot(transform.position);

                        node
                    } else {
                        self.nodes[contents.get_index().unwrap() as usize].clone()
                    };

                    let class = intersection.classify_with_plane(&node.plane);

                    if class == PlaneSide::Positive {
                        stack.push(Command::PopContraint);
                        stack.push(Command::ProcessNode(node.positive, is_second_tree));
                        stack.push(Command::PushContraint(
                            node.plane.flip(),
                            is_second_tree,
                            true,
                        ));
                    } else if class == PlaneSide::Negative {
                        stack.push(Command::PopContraint);
                        stack.push(Command::ProcessNode(node.negative, is_second_tree));
                        stack.push(Command::PushContraint(node.plane, is_second_tree, false));
                    } else if class == PlaneSide::Both {
                        stack.push(Command::PopContraint);
                        stack.push(Command::ProcessNode(node.positive, is_second_tree));
                        stack.push(Command::PushContraint(
                            node.plane.flip(),
                            is_second_tree,
                            true,
                        ));

                        stack.push(Command::PopContraint);
                        stack.push(Command::ProcessNode(node.negative, is_second_tree));
                        stack.push(Command::PushContraint(node.plane, is_second_tree, false));
                    }
                }
            }
        }

        best_hit
    }
    fn solve_spacetime_constraints(
        constraints: &[(Plane3, bool, bool)],
        starting_point: Vector3,
        displacement: Vector3,
    ) -> Option<RaycastData> {
        //TODO(arcprime): seidel's algorithm

        None
    }

    /// Perform a point containment test against the BSP
    pub fn point_query(&self, point: Vector3) -> HalfspaceContents {
        if self.nodes.is_empty() {
            return HalfspaceContents::empty();
        }

        let mut contents = HalfspaceContents::index(0);

        loop {
            if !contents.is_index() {
                return contents;
            }

            let node = &self.nodes[contents.get_index().unwrap() as usize];

            contents = if node.plane.distance_to_point(point) >= 0.0 {
                node.positive
            } else {
                node.negative
            };
        }
    }

    pub fn optimize(&mut self) {
        todo!()
    }

    /// Load a GeometryTree CSG object from an OBJ mesh (this WILL ignore holes in the mesh, so be warned)
    pub fn load_from_obj(obj: &Obj) -> Self {
        let mut polygons = Vec::new();

        for object in &obj.data.objects {
            for group in &object.groups {
                let mut metadata = HalfspaceMetadata::new();
                if let Some(material) = &group.material {}

                for polygon in &group.polys {
                    let pts: Vec<Vector3> = polygon
                        .0
                        .iter()
                        .map(|tuple| convert_to_vector3(obj.data.position[tuple.0]))
                        .collect();

                    polygons.push((pts, metadata));
                }
            }
        }

        let mut nodes = Vec::new();

        for (polygon, metadata) in polygons {}

        GeometryTree {
            nodes,
            bounds: get_bounding_box_of_obj(obj),
        }
    }
}

fn get_bounding_box_of_obj(obj: &Obj) -> BoundingBox3 {
    let mut min = Vector3::one() * 1e30;
    let mut max = Vector3::one() * -1e30;

    for object in &obj.data.objects {
        for group in &object.groups {
            for polygon in &group.polys {
                for IndexTuple(position, texture, normal) in &polygon.0 {
                    let position = obj.data.position[*position];

                    min = Vector3::new(
                        min.x.min(position[0]),
                        min.y.min(position[1]),
                        min.z.min(position[2]),
                    );
                    max = Vector3::new(
                        max.x.max(position[0]),
                        max.y.max(position[1]),
                        max.z.max(position[2]),
                    );
                }
            }
        }
    }

    BoundingBox3 { min, max }
}

fn classify_polygon_with_plane(polygon: &[Vector3], plane: &Plane3) -> PlaneSide {
    let mut negative = false;
    let mut positive = false;

    for x in polygon {
        let distance = plane.distance_to_point(*x);

        if distance >= 0.0 {
            positive = true;
        } else {
            negative = true;
        }
    }

    match (positive, negative) {
        (true, false) => PlaneSide::Positive,
        (false, true) => PlaneSide::Negative,
        _ => PlaneSide::Both,
    }
}

fn split_polygon_with_plane(polygon: &[Vector3], plane: &Plane3) -> (Vec<Vector3>, Vec<Vector3>) {
    let mut positive = vec![];
    let mut negative = vec![];

    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];

        let dist_a = plane.distance_to_point(a);
        let dist_b = plane.distance_to_point(b);

        if dist_a >= 0.0 {
            positive.push(a);
        }
        if dist_a <= 0.0 {
            negative.push(a);
        }

        if (dist_a > 0.0 && dist_b < 0.0) || (dist_a < 0.0 && dist_b > 0.0) {
            let t = dist_a / (dist_a - dist_b);
            let intersection = a + (b - a) * t;
            positive.push(intersection);
            negative.push(intersection);
        }
    }
    (positive, negative)
}

fn convert_to_vector3(point: [f32; 3]) -> Vector3 {
    Vector3::new(point[0], point[1], point[2])
}

fn get_polygon_plane(polygon: &[Vector3]) -> Plane3 {
    let cross = (polygon[1] - polygon[0])
        .cross(polygon[2] - polygon[0])
        .normalize();

    Plane3::new(cross, polygon[0].dot(cross))
}
