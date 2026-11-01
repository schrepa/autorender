#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod part;
mod settings;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source, gpu_struct};
use engine_places::bloom::Glow;
use engine_render::{DebugShown, PostLight, RenderBricks};

pub use settings::{Bloom, BloomStats};

gpu_struct! {
    #[derive(Resource)]
    pub struct BloomPiece {
        pub threshold: [f32; 4],
        pub applied: [f32; 4],
    }
}

pub struct BloomBrick;

impl Brick for BloomBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|bloom: Res<Bloom>| bloom.enabled())
            .settings::<Bloom>()
            .stats::<BloomStats>()
            .piece::<BloomPiece>()
            .fill::<Glow>(Some(shaders::bloom::faces::bloom::MODULE))
            .join::<PostLight>((), Some(shaders::bloom::apply::MODULE))
            .systems(Schedule::ViewSync, publish_bloom)
            .off(forget_bloom)
            .part::<part::BloomGpu>();
    }
}

fn publish_bloom(
    bloom: Res<Bloom>,
    shown: Res<DebugShown>,
    mut piece: ResMut<BloomPiece>,
    mut stats: ResMut<BloomStats>,
) {
    let active = shown.owner.is_none();
    piece.set_if_neq(BloomPiece {
        threshold: [bloom.threshold, 0.0, 0.0, 0.0],
        applied: [bloom.intensity, f32::from(active), 0.0, 0.0],
    });
    stats.set_if_neq(BloomStats { active });
}

fn forget_bloom(mut stats: ResMut<BloomStats>) {
    stats.set_if_neq(BloomStats::default());
}
