use bevy_ecs::prelude::{Component, Entity, Resource};
use engine_math::clamp;
use engine_place::Slot;
use serde::{Deserialize, Serialize};

pub const MOVED_SLOT: u32 = 0x8000_0000;
pub const FADE_SLOT: u32 = 0x00ff_ffff;
pub const FADE_WORD_ON: u32 = 0x100;
pub const FADE_WORD_OUT: u32 = 0x80;
pub const STILL_FRAMES: u32 = 16;

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Instances {
    pub slots: Vec<(Entity, u32)>,
    pub capacity: u32,
    pub candidates: u32,
    pub casters: bool,
}

impl Instances {
    pub fn drawn(&self, entity: Entity) -> Option<u32> {
        self.slots
            .binary_search_by_key(&entity, |&(drawn, _)| drawn)
            .ok()
            .map(|index| self.slots[index].1)
    }
}

impl Slot for Instances {
    const NAME: &'static str = "instances";
    const WGSL: Option<&'static str> = Some(contract!(
        "places/instances.wgsl",
        MOVED_SLOT,
        FADE_SLOT,
        FADE_WORD_ON,
        FADE_WORD_OUT,
        STILL_FRAMES,
    ));
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
