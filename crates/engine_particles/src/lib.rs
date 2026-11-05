#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod emitters;
mod part;
mod scene;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, ComponentKind, GpuStruct, SceneComponents, Schedule, Source};
use engine_places::particles::{ParticleBursts, ParticleEmitter, Transparency};
use engine_render::RenderBricks;

pub use emitters::{ParticleSettings, ParticleStats};

const GENERATED: &[&str] = &[
    <emitters::Emitter as GpuStruct>::WGSL,
    <part::Particles as GpuStruct>::WGSL,
];

pub struct ParticlesBrick;

impl Brick for ParticlesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(enabled)
            .settings::<ParticleSettings>()
            .stats::<ParticleStats>()
            .resource::<emitters::Gather>()
            .resource::<emitters::Packed>()
            .fill::<Transparency>(Some(shaders::particles::faces::transparency::MODULE))
            .fill::<ParticleBursts>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<ParticleEmitter>("particles")
                    .check(scene::check_emitter)
                    .reach(scene::emitter_reach)
                    .schema(scene::emitter_schema)
                    .example(scene::emitter_example),
                None,
            )
            .systems(
                Schedule::ViewSync,
                (emitters::gather_emitters, emitters::particle_stats),
            )
            .off(emitters::forget)
            .off_in(Schedule::ViewSync, emitters::drop_bursts)
            .generated(GENERATED)
            .part::<part::ParticlesGpu>();
    }
}

fn enabled(settings: Res<ParticleSettings>) -> bool {
    settings.enabled
}
