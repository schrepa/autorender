#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod part;
mod settings;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source, gpu_struct};
use engine_places_rendering::occlusion::AmbientVisibility;
use engine_render::RenderBricks;

pub use settings::{AmbientOcclusion, AmbientOcclusionStats};

gpu_struct! {
    #[derive(Resource)]
    pub struct AmbientPiece {
        pub occlusion: [f32; 4],
    }
}

pub struct AmbientOcclusionBrick;

impl Brick for AmbientOcclusionBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<AmbientOcclusion>()
            .stats::<AmbientOcclusionStats>()
            .piece::<AmbientPiece>()
            .fill::<AmbientVisibility>(Some(
                shaders::ambient_occlusion::faces::ambient_occlusion::MODULE,
            ))
            .systems(Schedule::ViewSync, (publish_piece, publish_stats))
            .part::<part::AmbientGpu>();
    }
}

fn publish_piece(ambient: Res<AmbientOcclusion>, mut piece: ResMut<AmbientPiece>) {
    piece.set_if_neq(AmbientPiece {
        occlusion: [ambient.radius, ambient.intensity, 0.0, 1.0],
    });
}

fn publish_stats(mut stats: ResMut<AmbientOcclusionStats>) {
    stats.set_if_neq(AmbientOcclusionStats { active: true });
}
