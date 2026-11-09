use std::sync::Arc;

use bevy_ecs::prelude::{Entity, Resource};
use engine_heightfield::HydrologyBake;
use engine_place::{Slot, Stamp};
use engine_places_simulation::ground::TileKey;

#[derive(Clone, Debug, PartialEq)]
pub struct WaterTileErrors {
    pub terrain: Entity,
    pub key: TileKey,
    pub stamp: Stamp,
    pub errors: Arc<[u16]>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Waterline {
    pub subdivision_error: f32,
    pub subdivision_leaves: u32,
    pub watered: Vec<Entity>,
    pub errors: Vec<WaterTileErrors>,
}

impl Slot for Waterline {
    const NAME: &'static str = "water";
    const WGSL: Option<&'static str> = Some(include_str!("places/water.wgsl"));
}

#[derive(Resource, Clone, Default)]
pub struct Hydrology {
    pub bakes: Vec<(Entity, Arc<HydrologyBake>)>,
}

impl Slot for Hydrology {
    const NAME: &'static str = "hydrology";
}
