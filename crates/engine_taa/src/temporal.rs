use bevy_ecs::prelude::*;
use engine_core::glam::Vec2;
use engine_core::{FrameTime, PrimaryView, ViewHistory, gpu_struct, profile};
use engine_render::{PassNames, Resolve, TargetDecl, TargetName, Written, wgpu};

use crate::antialiasing::{TemporalAntialiasing, TemporalAntialiasingStats};
use crate::previous::{JITTER_SAMPLES, jitter_offset, turn_of};

const CURRENT_WEIGHT: f32 = 0.1;

pub(crate) const HISTORY: TargetDecl = TargetDecl {
    name: TargetName("history"),
    format: wgpu::TextureFormat::Rgba16Float,
    written: Written::Composite,
    module: Some(crate::shaders::taa::history::MODULE),
};

pub(crate) const PASS: PassNames = PassNames {
    pass: "temporal_resolve",
    label: "engine/taa-resolve",
    span: "taa_resolve",
};

gpu_struct! {
    #[derive(Resource)]
    pub(crate) struct TaaPiece {
        pub(crate) options: [f32; 4],
        pub(crate) offset: [f32; 2],
        pub(crate) current_weight: f32,
        pub(crate) history_valid: u32,
    }
}

fn cycled(view: &PrimaryView) -> bool {
    view.facts().is_none_or(|facts| match facts.history {
        ViewHistory::Fresh => false,
        ViewHistory::Valid { since } => facts.frame.0.saturating_sub(since.0) + 1 >= JITTER_SAMPLES,
    })
}

pub(crate) fn publish(
    (view, time): (Res<PrimaryView>, Res<FrameTime>),
    settings: Res<TemporalAntialiasing>,
    mut resolve: ResMut<Resolve>,
    mut piece: ResMut<TaaPiece>,
    mut stats: ResMut<TemporalAntialiasingStats>,
) {
    let active = view.temporal();
    stats.set_if_neq(TemporalAntialiasingStats {
        active,
        settling: u32::from(active && !cycled(&view)),
    });
    if !active {
        resolve.set_if_neq(Resolve::default());
        return;
    }
    let history_valid = view.history().valid();
    let (offset, turn) = (jitter_offset(time.frame), turn_of(time.frame));
    profile::counter("render.taa_history", f64::from(u8::from(history_valid)));
    resolve.set_if_neq(Resolve {
        offset: Vec2::from(offset),
        turn,
    });
    piece.set_if_neq(TaaPiece {
        options: [
            turn,
            f32::from(u8::from(settings.nearest_motion)),
            f32::from(u8::from(settings.rounded_clamp)),
            settings.particle_response,
        ],
        offset,
        current_weight: CURRENT_WEIGHT,
        history_valid: u32::from(history_valid),
    });
}
