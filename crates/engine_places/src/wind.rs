use bevy_ecs::prelude::Resource;
use engine_math::sin;
use engine_place::Slot;
use glam::Vec2;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GustWaves {
    pub slow_share: f32,
    pub slow_rate: f32,
    pub slow_travel: f32,
    pub quick_share: f32,
    pub quick_rate: f32,
    pub quick_travel: f32,
    pub quick_across: f32,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Wind {
    pub velocity: [f32; 2],
    pub gusts: f32,
    pub clock: f32,
    pub waves: GustWaves,
}

impl Wind {
    pub fn at(&self, x: f32, z: f32, seconds: f32) -> Vec2 {
        let velocity = Vec2::from_array(self.velocity);
        let speed = velocity.length();
        if speed < 1.0e-4 {
            return Vec2::ZERO;
        }
        let toward = velocity / speed;
        let travel = x * toward.x + z * toward.y;
        let waves = self.waves;
        let gust = 1.0
            + self.gusts
                * (waves.slow_share * sin(seconds * waves.slow_rate - travel * waves.slow_travel)
                    + waves.quick_share
                        * sin(seconds * waves.quick_rate - travel * waves.quick_travel
                            + x * waves.quick_across));
        velocity * gust
    }
}

impl Slot for Wind {
    const NAME: &'static str = "wind";
    const WGSL: Option<&'static str> = Some(include_str!("places/wind.wgsl"));
}
