mod terrain;

pub use terrain::*;

use bevy_ecs::prelude::{Component, Resource, World};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundSample {
    pub height: f32,
    pub level: u32,
}

#[derive(Resource, Clone, Copy)]
pub struct Ground {
    pub sample: fn(&World, f32, f32) -> Option<GroundSample>,
    pub revision: fn(&World) -> u64,
}

impl Ground {
    pub fn sample_in(world: &World, x: f32, z: f32) -> Option<GroundSample> {
        let ground = world.get_resource::<Self>()?;
        (ground.sample)(world, x, z)
    }
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeightHeld(pub bool);

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFocus;
