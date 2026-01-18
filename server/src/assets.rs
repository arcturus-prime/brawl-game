use std::path::Path;

use obj::{Obj, ObjError};
use shared::math::Mesh;

#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("An error occured while loading the obj")]
    ObjError(#[from] ObjError),
}

pub fn load_mesh_from_obj(path: impl AsRef<Path>) -> Result<Mesh, AssetError> {
    let obj = Obj::load(path)?;
    let mut mesh = Mesh::default();

    Ok(mesh)
}
