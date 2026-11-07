#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod part;
mod settings;

use bevy_ecs::prelude::*;
use engine_core::{
    AttachPoint, Brick, Bricks, Pass, PassName, PassViews, PrimaryView, Schedule, SlotSet, Source,
    Writes, gpu_struct, never,
};
use engine_places_rendering::occlusion::OcclusionSource;
use engine_render::{PostScene, RenderBricks};

pub use settings::{AmbientOcclusion, AmbientOcclusionStats};

pub(crate) struct OcclusionPass;

impl PassName for OcclusionPass {
    const NAME: &'static str = "ambient_occlusion";
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
    pub struct AmbientPiece {
        pub occlusion: [f32; 4],
        pub canopy: [f32; 4],
    }
}

pub struct AmbientOcclusionBrick;

impl Brick for AmbientOcclusionBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|ambient: Res<AmbientOcclusion>| ambient.enabled)
            .settings::<AmbientOcclusion>()
            .stats::<AmbientOcclusionStats>()
            .piece::<AmbientPiece>()
            .pass::<OcclusionPass>()
            .join::<PostScene>((), Some(shaders::ambient_occlusion::apply::MODULE))
            .systems(Schedule::ViewSync, (publish_piece, publish_stats))
            .part::<part::AmbientGpu>();
    }
}

fn publish_piece(
    ambient: Res<AmbientOcclusion>,
    source: Res<OcclusionSource>,
    mut piece: ResMut<AmbientPiece>,
) {
    piece.set_if_neq(AmbientPiece {
        occlusion: [
            ambient.radius,
            ambient.intensity,
            0.0,
            1.0,
        ],
        canopy: [source.texel_share, 0.0, 0.0, 0.0],
    });
}

fn publish_stats(view: Res<PrimaryView>, mut stats: ResMut<AmbientOcclusionStats>) {
    stats.set_if_neq(AmbientOcclusionStats {
        active: !view.debug(),
    });
}
