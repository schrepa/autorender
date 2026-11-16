#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

pub mod animation;
pub mod audio;
pub mod impostors;
pub mod lighting;
pub mod lights;
pub mod occlusion;
pub mod particles;
pub mod probes;
pub mod reflections;
pub mod screens;
pub mod shadows;
pub mod sky;
pub mod sun;
pub mod text;
pub mod tonemap;
pub mod ui;
pub mod wind;

use engine_place::PlaceDecl;

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

pub const PLACES: &[PlaceDecl] = &[
    PlaceDecl::slot::<sun::SunPath>(),
    PlaceDecl::list::<sun::SunClocks>(),
    PlaceDecl::slot::<sun::Sun>(),
    PlaceDecl::slot::<wind::ViewWind>(),
    PlaceDecl::slot::<sky::SkyLight>(),
    PlaceDecl::slot::<sky::EyeMedium>(),
    PlaceDecl::slot::<sky::SkyVisibility>(),
    PlaceDecl::slot::<tonemap::Display>(),
    PlaceDecl::slot::<lighting::Lighting>(),
    PlaceDecl::list::<lighting::SunBlockers>(),
    PlaceDecl::list::<lighting::Ambient>(),
    PlaceDecl::list::<lighting::Interiors>(),
    PlaceDecl::slot::<occlusion::AmbientVisibility>(),
    PlaceDecl::slot::<lights::Lights>(),
    PlaceDecl::component::<lights::LocalLight>("local_light"),
    PlaceDecl::slot::<shadows::ShadowMaps>(),
    PlaceDecl::slot::<shadows::CasterRoutes>(),
    PlaceDecl::slot::<probes::ProbeCaptures>(),
    PlaceDecl::component::<probes::ReflectionProbe>("reflection_probe"),
    PlaceDecl::slot::<reflections::TraceSource>(),
    PlaceDecl::list::<reflections::TraceReaders>(),
    PlaceDecl::list::<reflections::TraceDepths>(),
    PlaceDecl::slot::<particles::ParticleFormula>(),
    PlaceDecl::slot::<particles::ParticleBursts>(),
    PlaceDecl::component::<particles::ParticleEmitter>("particle_emitter"),
    PlaceDecl::component::<animation::Animator>("animator"),
    PlaceDecl::component::<animation::Attachment>("attachment"),
    PlaceDecl::component::<animation::Model>("model"),
    PlaceDecl::ask::<animation::Animate>(),
    PlaceDecl::slot::<text::Text>(),
    PlaceDecl::slot::<ui::UiValues>(),
    PlaceDecl::slot::<ui::UiEvents>(),
    PlaceDecl::component::<ui::UiRoot>("ui"),
    PlaceDecl::slot::<audio::Sounds>(),
    PlaceDecl::component::<audio::SoundSource>("sound_source"),
    PlaceDecl::slot::<screens::Screens>(),
    PlaceDecl::slot::<impostors::Impostors>(),
    PlaceDecl::component::<impostors::ImpostorAsks>("impostor_asks"),
    PlaceDecl::component::<impostors::ImpostorRecords>("impostor_records"),
    PlaceDecl::component::<screens::Screen>("screen"),
    PlaceDecl::ask::<screens::ScreenAsk>(),
];
