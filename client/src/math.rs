use brawl_game_shared::math::Vector;
use raylib::math::Vector3;

pub fn to_raylib(vector: Vector) -> Vector3 {
    Vector3::new(vector.x, vector.y, vector.z)
}
