use bevy_ecs::prelude::Resource;
use engine_place::{JobRef, List, Slot};

pub use engine_heightfield::layer::DrawRivers;

pub struct LayerKind {
    pub layer: &'static engine_heightfield::layer::LayerKind,
    pub job: JobRef,
}

pub struct TerrainLayers;

impl List for TerrainLayers {
    const NAME: &'static str = "terrain_layers";
    type Member = LayerKind;
}

#[derive(Resource, Clone, Copy, Default)]
pub struct DrawnRivers {
    pub make: Option<DrawRivers>,
}

impl Slot for DrawnRivers {
    const NAME: &'static str = "drawn_rivers";
}
