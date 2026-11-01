use std::sync::Arc;

use bevy_ecs::prelude::World;
use engine_core::{Error, TileKey};
use engine_places::{GroundSource, WaterFace, WaterSample, WaterSurfaces, WaterWorker};
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
}
