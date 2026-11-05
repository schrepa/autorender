use engine_core::Listing;

pub use engine_ambient_occlusion::AmbientOcclusionBrick;
pub use engine_animation::{AnimationBrick, ImportShared, Model, import_glb};
pub use engine_audio::AudioBrick;
pub use engine_auto_exposure::AutoExposureBrick;
pub use engine_bloom::BloomBrick;
pub use engine_collision::CollisionBrick;
pub use engine_debug_view::DebugViewBrick;
pub use engine_dither::DitherBrick;
pub use engine_ground_cover::GroundCoverBrick;
pub use engine_grounding::GroundingBrick;
pub use engine_layer_erosion::LayerErosionBrick;
pub use engine_layer_heightmap::LayerHeightmapBrick;
pub use engine_layer_hydrology::LayerHydrologyBrick;
pub use engine_layer_noise::LayerNoiseBrick;
pub use engine_layer_paths::LayerPathsBrick;
pub use engine_layer_relief::LayerReliefBrick;
pub use engine_layer_stamps::LayerStampsBrick;
pub use engine_lighting::LightingModelBrick;
pub use engine_lights::LightsBrick;
pub use engine_meshes::MeshesBrick;
pub use engine_navigation::NavigationBrick;
pub use engine_particles::ParticlesBrick;
pub use engine_physics::PhysicsBrick;
pub use engine_probes::ReflectionProbesBrick;
pub use engine_reflections::ReflectionsBrick;
pub use engine_scatter::ScatterBrick;
pub use engine_screens::ScreensBrick;
pub use engine_shadows::ShadowsBrick;
pub use engine_sky::SkyBrick;
pub use engine_splines::SplinesBrick;
pub use engine_sun::SunBrick;
pub use engine_taa::TaaBrick;
pub use engine_terrain::TerrainBrick;
pub use engine_terrain_shading::TerrainShadingBrick;
pub use engine_text::TextBrick;
pub use engine_time_of_day::TimeOfDayBrick;
pub use engine_tonemap::TonemapBrick;
pub use engine_ui::UiBrick;
pub use engine_water::WaterBrick;
pub use engine_wind::WindBrick;
pub use engine_world_streaming::WorldStreamingBrick;

pub fn basics<L: Listing>(list: L) -> L {
    list.with::<WindBrick>()
        .with::<TimeOfDayBrick>()
        .with::<CollisionBrick>()
        .with::<PhysicsBrick>()
        .with::<SunBrick>()
        .with::<LightingModelBrick>()
        .with::<MeshesBrick>()
        .with::<ShadowsBrick>()
        .with::<LightsBrick>()
        .with::<SkyBrick>()
        .with::<AmbientOcclusionBrick>()
        .with::<BloomBrick>()
        .with::<DitherBrick>()
        .with::<TaaBrick>()
        .with::<AutoExposureBrick>()
        .with::<TonemapBrick>()
        .with::<DebugViewBrick>()
        .with::<AudioBrick>()
        .with::<ScreensBrick>()
        .with::<TextBrick>()
        .with::<UiBrick>()
        .places(engine_places::PLACES)
}

pub fn everything<L: Listing>(list: L) -> L {
    basics(list)
        .with::<TerrainBrick>()
        .with::<TerrainShadingBrick>()
        .with::<GroundingBrick>()
        .with::<WorldStreamingBrick>()
        .with::<NavigationBrick>()
        .with::<ScatterBrick>()
        .with::<GroundCoverBrick>()
        .with::<WaterBrick>()
        .with::<ReflectionProbesBrick>()
        .with::<ReflectionsBrick>()
        .with::<ParticlesBrick>()
        .with::<SplinesBrick>()
        .with::<LayerNoiseBrick>()
        .with::<LayerStampsBrick>()
        .with::<LayerHeightmapBrick>()
        .with::<LayerPathsBrick>()
        .with::<LayerErosionBrick>()
        .with::<LayerHydrologyBrick>()
        .with::<LayerReliefBrick>()
}
