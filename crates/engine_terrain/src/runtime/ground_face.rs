use bevy_ecs::prelude::{Entity, World};
use engine_core::{
    GroundFace, GroundObstacle, GroundPatch, GroundPoint, GroundRead, GroundSolid, GroundSource,
    WaterSample,
};
use glam::{Vec2, Vec3};

use super::Terrains;

pub(crate) struct TerrainGround;

impl GroundFace for TerrainGround {
    fn point(&self, world: &World, x: f32, z: f32) -> Option<GroundPoint> {
        let sample = world.get_resource::<Terrains>()?.sample(x, z)?;
        Some(GroundPoint {
            height: sample.height,
            normal: sample.normal,
            level: sample.level,
            hole: sample.hole,
            material: None,
        })
    }

    fn map(&self, _world: &World, _name: &str, _x: f32, _z: f32) -> Option<f32> {
        None
    }

    fn water(&self, world: &World, point: Vec3) -> Option<WaterSample> {
        super::water_source::terrain_water(world, point)
    }

    fn revision(&self, world: &World) -> u64 {
        world
            .get_resource::<Terrains>()
            .map_or(0, Terrains::revision)
    }

    fn patches(&self, _world: &World) -> Vec<GroundPatch> {
        Vec::new()
    }

    fn solids(&self, _world: &World) -> Vec<GroundSolid> {
        Vec::new()
    }

    fn obstacles(&self, _world: &World, _ground: Entity) -> Vec<GroundObstacle> {
        Vec::new()
    }

    fn source(&self, _world: &World, _from: Vec2, _to: Vec2) -> GroundRead<GroundSource> {
        GroundRead::Off
    }

    fn heightfield(&self) -> bool {
        true
    }
}
