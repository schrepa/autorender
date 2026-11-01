#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod part;
mod settings;

use std::any::TypeId;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, PartFaults, Schedule, Source, View, gpu_struct};
use engine_places::bloom::Glow;
use engine_render::{PostLight, RenderBricks};

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
    view: Res<View>,
    faults: Res<PartFaults>,
    mut piece: ResMut<BloomPiece>,
    mut stats: ResMut<BloomStats>,
) {
    let active = !view.debug();
    piece.set_if_neq(BloomPiece {
        threshold: [bloom.threshold, 0.0, 0.0, 0.0],
        applied: [bloom.intensity, f32::from(active), 0.0, 0.0],
    });
    stats.set_if_neq(BloomStats {
        active: active && !faults.faulted(TypeId::of::<part::BloomGpu>()),
    });
}

fn forget_bloom(mut stats: ResMut<BloomStats>) {
    stats.set_if_neq(BloomStats::default());
}
