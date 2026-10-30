use std::sync::Arc;

use bevy_ecs::prelude::World;
use engine_core::{
    Error, GroundSource, TileKey, WaterFace, WaterSample, WaterSurfaces, WaterWorker,
};
use glam::Vec3;

use super::Terrains;
use super::ground_face::{keyed_tile, tile_surfaces, walked_tile};
use crate::jobs::{TerrainWorker, WalkedTile};

pub(crate) struct TerrainWater;

impl WaterFace for TerrainWater {
    fn water(&self, world: &World, point: Vec3) -> Option<WaterSample> {
        world.get_resource::<Terrains>()?.water_at(point)
    }

    fn reaches(&self, world: &World) -> Vec<(Vec3, Vec3)> {
        let terrains = world.get_resource::<Terrains>();
        let rivers = terrains
            .iter()
            .flat_map(|terrains| terrains.waters())
            .filter_map(|(_, origin, layers)| {
                let rivers = layers.hydrology()?.rivers();
                (!rivers.is_empty()).then(|| {
                    let (low, high) = rivers.bounds();
                    (
                        origin + Vec3::new(low.x, -1.0e5, low.y),
                        origin + Vec3::new(high.x, 1.0e5, high.y),
                    )
                })
            });
        let live = terrains
            .iter()
            .flat_map(|terrains| terrains.water_windows())
            .map(|(low, high)| {
                (
                    Vec3::new(low.x, -1.0e5, low.y),
                    Vec3::new(high.x, 1.0e5, high.y),
                )
            });
        rivers.chain(live).collect()
    }

    fn surfaces(&self, world: &World, ground: &GroundSource) -> Option<Arc<dyn WaterSurfaces>> {
        let terrains = world.get_resource::<Terrains>()?;
        let held = terrains
            .entries
            .iter()
            .any(|state| state.entity == ground.ground && state.around.is_none());
        held.then(|| Arc::new(TerrainSurfaces) as Arc<dyn WaterSurfaces>)
    }
}

struct TerrainSurfaces;

impl WaterSurfaces for TerrainSurfaces {
    fn decode(&self, key: TileKey, ground: &[u8]) -> Result<Option<Box<[f32]>>, Error> {
        Ok(tile_surfaces(&keyed_tile(key, ground)?))
    }
}

impl WaterWorker for TerrainWorker {
    fn surfaces(&self, identity: &[u8], key: TileKey) -> Result<Option<Box<[f32]>>, Error> {
        let mut walked = self.last_walk();
        if let Some(held) = walked.as_ref()
            && *held.identity == *identity
            && held.key == key
        {
            return Ok(tile_surfaces(&held.tile));
        }
        let (tile, _, _) = walked_tile(self, identity, key)?;
        let found = tile_surfaces(&tile);
        *walked = Some(WalkedTile {
            identity: identity.into(),
            key,
            tile,
        });
        Ok(found)
    }
}
