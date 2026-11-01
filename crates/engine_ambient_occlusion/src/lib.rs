#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod part;
mod settings;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source, gpu_struct};
use engine_places::ground::Occupancy;
use engine_places::shadows::ShadowMaps;
use engine_render::{DebugShown, PostScene, RenderBricks};

pub use settings::{AmbientOcclusion, AmbientOcclusionStats};

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
            .when(|ambient: Res<AmbientOcclusion>, maps: Res<ShadowMaps>| {
                ambient.enabled() || maps.sky_darkens
            })
            .settings::<AmbientOcclusion>()
            .stats::<AmbientOcclusionStats>()
            .piece::<AmbientPiece>()
            .join::<PostScene>((), Some(shaders::ambient_occlusion::apply::MODULE))
            .systems(Schedule::ViewSync, (publish_piece, publish_stats))
            .off(forget_stats)
            .part::<part::AmbientGpu>();
    }
}

fn publish_piece(
    ambient: Res<AmbientOcclusion>,
    occupancy: Res<Occupancy>,
    mut piece: ResMut<AmbientPiece>,
) {
    piece.set_if_neq(AmbientPiece {
        occlusion: [
            ambient.radius,
            ambient.intensity,
            0.0,
            if ambient.enabled() { 1.0 } else { 0.0 },
        ],
        canopy: [occupancy.texel_share, 0.0, 0.0, 0.0],
    });
}

fn publish_stats(
    ambient: Res<AmbientOcclusion>,
    shown: Res<DebugShown>,
    mut stats: ResMut<AmbientOcclusionStats>,
) {
    stats.set_if_neq(AmbientOcclusionStats {
        active: ambient.enabled() && shown.owner.is_none(),
    });
}

fn forget_stats(mut stats: ResMut<AmbientOcclusionStats>) {
    stats.set_if_neq(AmbientOcclusionStats::default());
}
