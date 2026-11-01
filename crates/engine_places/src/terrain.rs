use bevy_ecs::prelude::Resource;
use engine_place::{List, Slot};

pub use engine_heightfield::layer::{DrawRivers, LayerKind};

pub struct TerrainLayers;

impl List for TerrainLayers {
    const NAME: &'static str = "terrain_layers";
    type Member = &'static LayerKind;
}

#[derive(Resource, Clone, Copy, Default)]
pub struct DrawnRivers {
    pub make: Option<DrawRivers>,
}

impl Slot for DrawnRivers {
    const NAME: &'static str = "drawn_rivers";
}
