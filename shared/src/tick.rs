#[derive(Default)]
pub struct Ticker {
    tick: u32,

    warp: f32,
    step_size: f32,

    accumulator: f32,
}

impl Ticker {
    pub fn update<T: FnMut(u32, f32) -> ()>(&mut self, dt: f32, mut function: T) {
        self.accumulator += dt;

        while self.accumulator > self.step_size + self.warp {
            self.accumulator -= self.step_size + self.warp;
            self.tick += 1;

            function(self.tick, self.step_size)
        }
    }

    pub fn set_tick(&mut self, tick: u32) {
        self.tick = tick;
    }

    pub fn set_warp(&mut self, warp: f32) {
        self.warp = -warp;
    }
}
