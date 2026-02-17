use std::collections::HashMap;

use obj::Obj;

use crate::math::Vector3;

pub struct Mesh {
    pub vertices: Vec<Vector3>,
    pub normals: Vec<Vector3>,
    pub uv: Vec<(f32, f32)>,

    pub indices: Vec<usize>,
}

impl Mesh {
    pub fn load_from_obj(obj: &Obj) -> Self {
        let data = &obj.data;

        let mut out_vertices: Vec<Vector3> = Vec::new();
        let mut normals: Vec<Vector3> = Vec::new();
        let mut uv: Vec<(f32, f32)> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();

        let mut index_map: HashMap<(usize, Option<usize>, Option<usize>), usize> = HashMap::new();

        for object in &data.objects {
            for group in &object.groups {
                for poly in &group.polys {
                    let in_vertices = &poly.0;
                    for i in 1..in_vertices.len().saturating_sub(1) {
                        for &corner_index in &[0, i, i + 1] {
                            let index_tuple = in_vertices[corner_index];
                            let key = (index_tuple.0, index_tuple.1, index_tuple.2);

                            let unified = *index_map.entry(key).or_insert_with(|| {
                                let new_index = out_vertices.len();

                                let p = data.position[index_tuple.0];
                                out_vertices.push(Vector3::new(p[0], p[1], p[2]));

                                let n = index_tuple
                                    .2
                                    .map(|ni| {
                                        let n = data.normal[ni];
                                        Vector3::new(n[0], n[1], n[2])
                                    })
                                    .unwrap_or(Vector3::ZERO);
                                normals.push(n);

                                let texture_coordinates = index_tuple
                                    .1
                                    .map(|ti| {
                                        let t = data.texture[ti];
                                        (t[0], 1.0 - t[1])
                                    })
                                    .unwrap_or((0.0, 0.0));
                                uv.push(texture_coordinates);

                                new_index
                            });

                            indices.push(unified);
                        }
                    }
                }
            }
        }

        Self {
            vertices: out_vertices,
            normals,
            uv,
            indices,
        }
    }
}
