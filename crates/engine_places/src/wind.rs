use bevy_ecs::prelude::Resource;
use engine_math::sin;
use engine_place::Slot;
use glam::Vec2;

pub const GUST_SLOW_SHARE: f32 = 0.6;
pub const GUST_SLOW_RATE: f32 = 0.9;
pub const GUST_SLOW_TRAVEL: f32 = 0.05;
pub const GUST_QUICK_SHARE: f32 = 0.4;
pub const GUST_QUICK_RATE: f32 = 2.3;
pub const GUST_QUICK_TRAVEL: f32 = 0.17;
pub const GUST_QUICK_ACROSS: f32 = 0.31;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Wind {
    pub velocity: [f32; 2],
    pub gusts: f32,
    pub clock: f32,
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
        let gust = 1.0
            + self.gusts
                * (GUST_SLOW_SHARE * sin(seconds * GUST_SLOW_RATE - travel * GUST_SLOW_TRAVEL)
                    + GUST_QUICK_SHARE
                        * sin(seconds * GUST_QUICK_RATE - travel * GUST_QUICK_TRAVEL
                            + x * GUST_QUICK_ACROSS));
        velocity * gust
    }
}

impl Slot for Wind {
    const NAME: &'static str = "wind";
    const WGSL: Option<&'static str> = Some(include_str!("places/wind.wgsl"));
}
