use bevy_ecs::prelude::{Entity, Resource};
use engine_place::Slot;
use glam::Vec3;

pub trait Obstruction: Send + Sync + 'static {
    fn sphere(&self, radius: f32, from: Vec3, along: Vec3, ignore: Entity) -> Option<f32>;
}

struct Open;

impl Obstruction for Open {
    fn sphere(&self, _: f32, _: Vec3, _: Vec3, _: Entity) -> Option<f32> {
        None
    }
}

#[derive(Resource)]
pub struct Obstructions(pub Box<dyn Obstruction>);

impl Default for Obstructions {
    fn default() -> Self {
        Self(Box::new(Open))
    }
}

impl Slot for Obstructions {
    const NAME: &'static str = "obstructions";
}
