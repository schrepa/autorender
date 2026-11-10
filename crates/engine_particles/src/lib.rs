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
use engine_core::{
    Attributes, Brick, Bricks, ComponentKind, GpuStruct, KindDecl, SceneComponents,
    Schedule, Source,
};
use engine_places_rendering::particles::{ParticleBursts, ParticleEmitter, Transparency};
use engine_render::RenderBricks;

pub use emitters::{EMITTER_PART, ParticleSettings, ParticleStats};

pub use engine_places_rendering::particles::PARTICLES;

const GENERATED: &[&str] = &[
    emitters::WGSL,
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
            .moving::<ParticleEmitter>()
            .join::<SceneComponents>(
                ComponentKind::of::<ParticleEmitter>("particles")
                    .check(scene::check_emitter)
                    .schema(scene::emitter_schema),
                None,
            )
            .systems(Schedule::WorldSync, emitters::judge_emitters)
            .on_remove::<ParticleEmitter, _>(emitters::forget_emitter)
            .systems(
                Schedule::ViewSync,
                (emitters::gather_emitters, emitters::particle_stats),
            )
            .generated(GENERATED)
            .draw_kind(KindDecl {
                kind: PARTICLES,
                vertex: shaders::particles::draw::VS_PARTICLE.name(),
                needs: Attributes::NONE,
                casts_shadows: false,
                seen_by_probes: false,
            })
            .pass::<part::WeightedPass>()
            .pass::<part::CompositePass>()
            .draws::<part::ParticlesGpu>(PARTICLES, part::ParticlesGpu::draw);
    }
}

fn enabled(settings: Res<ParticleSettings>) -> bool {
    settings.enabled
}
