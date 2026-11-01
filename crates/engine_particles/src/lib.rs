#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod emitters;
mod nearest;
mod part;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, ComponentKind, SceneComponents, Schedule, Source};
use engine_places::particles::{ParticleEmitter, Transparency};
use engine_render::RenderBricks;

pub use emitters::{ParticleSettings, ParticleStats};

pub struct ParticlesBrick;

impl Brick for ParticlesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(emits)
            .settings::<ParticleSettings>()
            .stats::<ParticleStats>()
            .resource::<emitters::Gather>()
            .resource::<emitters::Packed>()
            .resource::<emitters::Owners>()
            .fill::<Transparency>(Some(shaders::particles::faces::transparency::MODULE))
            .join::<SceneComponents>(ComponentKind::of::<ParticleEmitter>("particles"), None)
            .systems(
                Schedule::ViewSync,
                (emitters::gather_emitters, emitters::particle_stats),
            )
            .off(emitters::forget)
            .part::<part::ParticlesGpu>();
    }
}

fn emits(settings: Res<ParticleSettings>, emitters: Query<(), With<ParticleEmitter>>) -> bool {
    settings.enabled && !emitters.is_empty()
}
