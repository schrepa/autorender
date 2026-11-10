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

use engine_core::{
    Attributes, Brick, Bricks, ColorPass, ComponentKind, GpuStruct, KindDecl, KindSkip,
    PassPurpose, PassRole, PassViews, SceneComponents, Schedule, Source,
};
use engine_places_rendering::particles::{ParticleBursts, ParticleEmitter};
use engine_render::RenderBricks;

pub use emitters::{EMITTER_PART, ParticleSettings, ParticleStats};

pub use engine_places_rendering::particles::PARTICLES;

const GENERATED: &[&str] = &[
    emitters::WGSL,
    <emitters::Emitter as GpuStruct>::WGSL,
    <part::Particles as GpuStruct>::WGSL,
];

const SKIPS: &[KindSkip] = &[
    KindSkip::role(
        PassRole::Depth,
        PassViews::Owned,
        "particles cast no shadows",
    ),
    KindSkip::role(
        PassRole::Lit,
        PassViews::Owned,
        "particles stay out of probe captures",
    ),
    KindSkip::pass::<ColorPass>("particles only blend"),
    KindSkip::role(PassRole::Depth, PassViews::Main, "particles write no depth"),
    KindSkip::purpose(PassPurpose::Debug, "particles carry no debug record"),
];

pub struct ParticlesBrick;

impl Brick for ParticlesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<ParticleSettings>()
            .stats::<ParticleStats>()
            .resource::<emitters::Gather>()
            .resource::<emitters::Packed>()
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
                needs: Attributes::NONE,
                streams: &[],
                skips: SKIPS,
            })
            .draws::<part::ParticlesGpu>(PARTICLES, part::ParticlesGpu::draw);
    }
}
