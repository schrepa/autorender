#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::Listing;

pub use engine_ambient_occlusion::AmbientOcclusionBrick;
pub use engine_animation::{AnimationBrick, ImportShared, Model, import_glb};
pub use engine_audio::AudioBrick;
pub use engine_auto_exposure::AutoExposureBrick;
pub use engine_bloom::BloomBrick;
pub use engine_debug_view::DebugViewBrick;
pub use engine_dither::DitherBrick;
pub use engine_lighting::LightingModelBrick;
pub use engine_lights::LightsBrick;
pub use engine_meshes::MeshesBrick;
pub use engine_particles::ParticlesBrick;
pub use engine_prepass::PrepassBrick;
pub use engine_probes::ReflectionProbesBrick;
pub use engine_reflections::ReflectionsBrick;
pub use engine_screens::ScreensBrick;
pub use engine_shadows::ShadowsBrick;
pub use engine_sky::SkyBrick;
pub use engine_sun::SunBrick;
pub use engine_taa::TaaBrick;
pub use engine_text::TextBrick;
pub use engine_time_of_day::TimeOfDayBrick;
pub use engine_tonemap::TonemapBrick;
pub use engine_ui::UiBrick;

pub fn rendering<L: Listing>(list: L) -> L {
    list.with::<TimeOfDayBrick>()
        .with::<SunBrick>()
        .with::<LightingModelBrick>()
        .with::<MeshesBrick>()
        .with::<ShadowsBrick>()
        .with::<LightsBrick>()
        .with::<SkyBrick>()
        .with::<PrepassBrick>()
        .with::<AmbientOcclusionBrick>()
        .with::<BloomBrick>()
        .with::<TaaBrick>()
        .with::<AutoExposureBrick>()
        .with::<TonemapBrick>()
        .with::<DitherBrick>()
        .with::<DebugViewBrick>()
        .with::<AudioBrick>()
        .with::<ScreensBrick>()
        .with::<TextBrick>()
        .with::<UiBrick>()
        .with::<ReflectionProbesBrick>()
        .with::<ReflectionsBrick>()
        .with::<ParticlesBrick>()
        .places(engine_places::PLACES)
        .places(engine_places_rendering::PLACES)
}
