#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod part;
mod settings;

use bevy_ecs::prelude::*;
use engine_core::{
    AttachPoint, Brick, Bricks, Pass, PassName, PassViews, Schedule, SlotSet, Source, Writes,
    gpu_struct, never,
};
use engine_render::RenderBricks;

pub use settings::{Bloom, BloomStats};

pub(crate) struct BloomPass;

impl PassName for BloomPass {
    const NAME: &'static str = "bloom";
    const PASS: Pass = Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Main,
        at: AttachPoint::Post,
        takes: never,
    };
}

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
            .when(|bloom: Res<Bloom>| bloom.enabled)
            .settings::<Bloom>()
            .stats::<BloomStats>()
            .piece::<BloomPiece>()
            .applies::<BloomPass>(shaders::bloom::apply::MODULE)
            .systems(Schedule::ViewSync, publish_bloom)
            .part::<part::BloomGpu>();
    }
}

fn publish_bloom(bloom: Res<Bloom>, mut piece: ResMut<BloomPiece>, mut stats: ResMut<BloomStats>) {
    piece.set_if_neq(BloomPiece {
        threshold: [bloom.threshold, 0.0, 0.0, 0.0],
        applied: [bloom.intensity, 1.0, 0.0, 0.0],
    });
    stats.set_if_neq(BloomStats { active: true });
}
