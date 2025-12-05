use raylib::{color::Color, prelude::RaylibDraw};

use crate::world::ClientWorld;

mod math;
mod world;

fn main() {
    let mut world = ClientWorld::new();

    let (mut rl, thread) = raylib::init().size(640, 480).title("Brawl Game").build();

    while !rl.window_should_close() {
        world.update(&rl);

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);

        world.render(d);
    }
}
