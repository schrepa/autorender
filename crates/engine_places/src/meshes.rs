use bevy_ecs::prelude::{Component, Resource};
use engine_math::clamp;
use engine_place::Slot;
use serde::{Deserialize, Serialize};

#[derive(Resource, Default)]
pub struct Instances;

impl Slot for Instances {
    const NAME: &'static str = "instances";
    const WGSL: Option<&'static str> = Some(include_str!("places/instances.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::INSTANCES];
}

#[derive(Component, Debug, Clone, Copy)]
pub struct Cube {
    pub color: [f32; 4],
}

#[derive(Component, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tint(pub [f32; 4]);

impl Tint {
    pub const WHITE: Self = Self([1.0; 4]);

    pub fn apply(&self, color: [f32; 4]) -> [f32; 4] {
        let mut tinted = color;
        for (channel, factor) in tinted.iter_mut().zip(self.0) {
            let factor = if factor.is_finite() {
                clamp(factor, 0.0, 1.0)
            } else {
                1.0
            };
            *channel *= factor;
        }
        tinted
    }
}

impl Default for Tint {
    fn default() -> Self {
        Self::WHITE
    }
}
