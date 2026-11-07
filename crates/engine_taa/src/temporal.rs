use bevy_ecs::prelude::*;
use engine_core::{FrameTime, GpuStruct, Jitter, PrimaryView, ViewHistory, gpu_struct, profile};

use crate::antialiasing::{TemporalAntialiasing, TemporalAntialiasingStats};
use crate::previous::{JITTER_SAMPLES, jitter_offset, turn_of};

const CURRENT_WEIGHT: f32 = 0.1;

pub(crate) const GENERATED: &[&str] = &[<TaaView as GpuStruct>::WGSL];

gpu_struct! {
    #[derive(Resource)]
    pub(crate) struct TaaPiece {
        pub(crate) options: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct TaaView {
        pub(crate) offset: [f32; 2],
        pub(crate) history: u32,
        pub(crate) pad: u32,
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
    view: Res<PrimaryView>,
    settings: Res<TemporalAntialiasing>,
    mut piece: ResMut<TaaPiece>,
    mut stats: ResMut<TemporalAntialiasingStats>,
) {
    let active = view.jitter().is_some();
    stats.set_if_neq(TemporalAntialiasingStats {
        active,
        settling: u32::from(active && !cycled(&view)),
    });
    if !active {
        return;
    }
    profile::counter("render.taa_history", f64::from(u8::from(view.history().valid())));
    piece.set_if_neq(TaaPiece {
        options: [
            f32::from(u8::from(settings.nearest_motion)),
            f32::from(u8::from(settings.rounded_clamp)),
            settings.particle_response,
            CURRENT_WEIGHT,
        ],
    });
}
