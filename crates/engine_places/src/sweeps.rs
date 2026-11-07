use bevy_ecs::prelude::{Entity, Resource};
use engine_place::Slot;
use glam::Vec3;

pub trait Sweep: Send + Sync + 'static {
    fn sphere(&self, radius: f32, from: Vec3, along: Vec3, ignore: Entity) -> Option<f32>;
}

struct NoSweep;

impl Sweep for NoSweep {
    fn sphere(&self, _: f32, _: Vec3, _: Vec3, _: Entity) -> Option<f32> {
        None
    }
}

#[derive(Resource)]
pub struct Sweeps(pub Box<dyn Sweep>);

impl Default for Sweeps {
    fn default() -> Self {
        Self(Box::new(NoSweep))
    }
}

impl Slot for Sweeps {
    const NAME: &'static str = "sweeps";
}
