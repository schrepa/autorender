use engine_core::Listing;

pub use engine_collision::CollisionBrick;
pub use engine_ground_cover::GroundCoverBrick;
pub use engine_grounding::GroundingBrick;
pub use engine_layer_erosion::LayerErosionBrick;
pub use engine_layer_heightmap::LayerHeightmapBrick;
pub use engine_layer_hydrology::LayerHydrologyBrick;
pub use engine_layer_noise::LayerNoiseBrick;
pub use engine_layer_paths::LayerPathsBrick;
pub use engine_layer_relief::LayerReliefBrick;
pub use engine_layer_stamps::LayerStampsBrick;
pub use engine_navigation::NavigationBrick;
pub use engine_physics::PhysicsBrick;
pub use engine_scatter::ScatterBrick;
pub use engine_splines::SplinesBrick;
pub use engine_terrain::TerrainBrick;
pub use engine_terrain_shading::TerrainShadingBrick;
pub use engine_water::WaterBrick;
pub use engine_wind::WindBrick;
pub use engine_world_streaming::WorldStreamingBrick;

pub fn everything<L: Listing>(list: L) -> L {
    engine_standard_rendering::rendering(list)
        .with::<WindBrick>()
        .with::<CollisionBrick>()
        .with::<PhysicsBrick>()
        .with::<TerrainBrick>()
        .with::<TerrainShadingBrick>()
        .with::<GroundingBrick>()
        .with::<WorldStreamingBrick>()
        .with::<NavigationBrick>()
        .with::<ScatterBrick>()
        .with::<GroundCoverBrick>()
        .with::<WaterBrick>()
        .with::<SplinesBrick>()
        .with::<LayerNoiseBrick>()
        .with::<LayerStampsBrick>()
        .with::<LayerHeightmapBrick>()
        .with::<LayerPathsBrick>()
        .with::<LayerErosionBrick>()
        .with::<LayerHydrologyBrick>()
        .with::<LayerReliefBrick>()
        .places(engine_places_simulation::PLACES)
        .places(engine_places_world::PLACES)
}
