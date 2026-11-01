use std::sync::Arc;

use bevy_ecs::prelude::Resource;
use engine_heightfield::{GivenRiver, TerrainLayer};
use engine_place::{Error, JobRef, List, Slot};

pub struct LayerKind {
    pub name: &'static str,
    pub schema: fn() -> serde_json::Value,
    pub from_json: fn(&serde_json::Value) -> Result<Arc<dyn TerrainLayer>, Error>,
    pub job: JobRef,
}

pub struct TerrainLayers;

impl List for TerrainLayers {
    const NAME: &'static str = "terrain_layers";
    type Member = LayerKind;
}

pub type DrawRivers = fn(&[GivenRiver]) -> Result<Arc<dyn TerrainLayer>, Error>;

#[derive(Resource, Clone, Copy, Default)]
pub struct DrawnRivers {
    pub make: Option<DrawRivers>,
}

impl Slot for DrawnRivers {
    const NAME: &'static str = "drawn_rivers";
}
