#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{FoldOrder, Listing};

pub use engine_ambient_occlusion::AmbientOcclusionBrick;
pub use engine_animation::{
    AnimationBrick, ImportShared, ImportedModel, MODEL_PART, ModelLoaded, import_glb,
};
pub use engine_audio::AudioBrick;
pub use engine_auto_exposure::AutoExposureBrick;
pub use engine_bloom::BloomBrick;
pub use engine_debug_view::DebugViewBrick;
pub use engine_dither::DitherBrick;
pub use engine_dynamic_resolution::DynamicResolutionBrick;
pub use engine_fly_camera::FlyCameraBrick;
pub use engine_follow_camera::FollowCameraBrick;
pub use engine_impostors::ImpostorsBrick;
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
pub use engine_sprites::SpritesBrick;
pub use engine_sun::SunBrick;
pub use engine_taa::TaaBrick;
pub use engine_text::TextBrick;
pub use engine_texture_cook::TextureCook;
pub use engine_time_of_day::TimeOfDayBrick;
pub use engine_tonemap::TonemapBrick;
pub use engine_ui::UiBrick;

const POST_ORDER: [&str; 2] = ["bloom", "haze"];
const DISPLAY_ORDER: [&str; 2] = ["tonemap", "dither"];

pub fn minimal<L: Listing>(list: L) -> L {
    list.with::<TextureCook>()
        .with::<TimeOfDayBrick>()
        .with::<SunBrick>()
        .with::<LightingModelBrick>()
        .with::<MeshesBrick>()
        .with::<TonemapBrick>()
        .with::<FlyCameraBrick>()
        .places(engine_places::PLACES)
        .places(engine_places_rendering::PLACES)
}

pub fn two_d<L: Listing>(list: L) -> L {
    list.without::<ShadowsBrick>()
        .without::<ReflectionProbesBrick>()
        .with::<TextureCook>()
        .with::<LightingModelBrick>()
        .with::<SpritesBrick>()
        .with::<TextBrick>()
        .with::<UiBrick>()
        .with::<ScreensBrick>()
        .with::<TonemapBrick>()
        .with::<FollowCameraBrick>()
        .places(engine_places::PLACES)
        .places(engine_places_rendering::PLACES)
}

pub fn rendering<L: Listing>(list: L) -> L {
    list.with::<TextureCook>()
        .with::<TimeOfDayBrick>()
        .with::<SunBrick>()
        .with::<LightingModelBrick>()
        .with::<MeshesBrick>()
        .with::<ImpostorsBrick>()
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
        .with::<AnimationBrick>()
        .with::<DynamicResolutionBrick>()
        .places(engine_places::PLACES)
        .places(engine_places_rendering::PLACES)
        .settings(FoldOrder {
            post: POST_ORDER.map(str::to_owned).to_vec(),
            display: DISPLAY_ORDER.map(str::to_owned).to_vec(),
        })
}
