use bevy_ecs::prelude::*;
use engine_core::{FrameTime, Jitter, PrimaryView, ViewHistory, gpu_struct, profile};
use engine_render::{PassNames, TargetDecl, TargetName, Written, wgpu};

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

pub(crate) fn jitter(time: Res<FrameTime>, mut jitter: ResMut<Jitter>) {
    jitter.set_if_neq(Jitter {
        offset: Some(jitter_offset(time.frame)),
        turn: turn_of(time.frame),
    });
}

pub(crate) fn publish(
    (view, jitter): (Res<PrimaryView>, Res<Jitter>),
    settings: Res<TemporalAntialiasing>,
    mut piece: ResMut<TaaPiece>,
    mut stats: ResMut<TemporalAntialiasingStats>,
) {
    let offset = view.jitter();
    stats.set_if_neq(TemporalAntialiasingStats {
        active: offset.is_some(),
        settling: u32::from(offset.is_some() && !cycled(&view)),
    });
    let Some(offset) = offset else {
        return;
    };
    let history_valid = view.history().valid();
    profile::counter("render.taa_history", f64::from(u8::from(history_valid)));
    piece.set_if_neq(TaaPiece {
        options: [
            jitter.turn,
            f32::from(u8::from(settings.nearest_motion)),
            f32::from(u8::from(settings.rounded_clamp)),
            settings.particle_response,
        ],
        offset: offset.to_array(),
        current_weight: CURRENT_WEIGHT,
        history_valid: u32::from(history_valid),
    });
}
