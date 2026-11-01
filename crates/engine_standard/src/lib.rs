use engine_ambient_occlusion::AmbientOcclusionBrick;
use engine_auto_exposure::AutoExposureBrick;
use engine_bloom::BloomBrick;
use engine_collision::CollisionBrick;
use engine_core::Listing;
use engine_debug_view::DebugViewBrick;
use engine_dither::DitherBrick;
use engine_ground_cover::GroundCoverBrick;
use engine_grounding::GroundingBrick;
use engine_layer_erosion::LayerErosionBrick;
use engine_layer_heightmap::LayerHeightmapBrick;
use engine_layer_hydrology::LayerHydrologyBrick;
use engine_layer_noise::LayerNoiseBrick;
use engine_layer_paths::LayerPathsBrick;
use engine_layer_relief::LayerReliefBrick;
use engine_layer_stamps::LayerStampsBrick;
use engine_lighting::LightingModelBrick;
use engine_lights::LightsBrick;
use engine_meshes::MeshesBrick;
use engine_navigation::NavigationBrick;
use engine_particles::ParticlesBrick;
use engine_physics::PhysicsBrick;
use engine_probes::ReflectionProbesBrick;
use engine_reflections::ReflectionsBrick;
use engine_scatter::ScatterBrick;
use engine_shadows::ShadowsBrick;
use engine_sky::SkyBrick;
use engine_splines::SplinesBrick;
use engine_sun::SunBrick;
use engine_taa::TaaBrick;
use engine_terrain::TerrainBrick;
use engine_terrain_shading::TerrainShadingBrick;
use engine_time_of_day::TimeOfDayBrick;
use engine_tonemap::TonemapBrick;
use engine_water::WaterBrick;
use engine_wind::WindBrick;
use engine_world_streaming::WorldStreamingBrick;

pub fn standard<L: Listing>(list: L) -> L {
    list.with::<TerrainBrick>()
        .with::<GroundingBrick>()
        .with::<WindBrick>()
        .with::<WorldStreamingBrick>()
        .with::<NavigationBrick>()
        .with::<TimeOfDayBrick>()
        .with::<CollisionBrick>()
        .with::<PhysicsBrick>()
        .with::<SunBrick>()
        .with::<LightingModelBrick>()
        .with::<MeshesBrick>()
        .with::<TerrainShadingBrick>()
        .with::<ScatterBrick>()
        .with::<ShadowsBrick>()
        .with::<LightsBrick>()
        .with::<SkyBrick>()
        .with::<ReflectionProbesBrick>()
        .with::<GroundCoverBrick>()
        .with::<WaterBrick>()
        .with::<AmbientOcclusionBrick>()
        .with::<ReflectionsBrick>()
        .with::<BloomBrick>()
        .with::<DitherBrick>()
        .with::<ParticlesBrick>()
        .with::<TaaBrick>()
        .with::<AutoExposureBrick>()
        .with::<TonemapBrick>()
        .with::<DebugViewBrick>()
        .with::<SplinesBrick>()
        .with::<LayerNoiseBrick>()
        .with::<LayerStampsBrick>()
        .with::<LayerHeightmapBrick>()
        .with::<LayerPathsBrick>()
        .with::<LayerErosionBrick>()
        .with::<LayerHydrologyBrick>()
        .with::<LayerReliefBrick>()
        .places(engine_places::PLACES)
}
