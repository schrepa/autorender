use bevy_ecs::prelude::*;
use engine_core::glam::Vec2;
use engine_core::{View, gpu_struct, profile};
use engine_render::{Output, PassNames, Resolve, wgpu};

use crate::antialiasing::{TemporalAntialiasing, TemporalAntialiasingStats};
use crate::previous::History;

const CURRENT_WEIGHT: f32 = 0.1;

pub(crate) const HISTORY: Output = Output {
    name: "history",
    format: wgpu::TextureFormat::Rgba8UnormSrgb,
    layers: &[],
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

pub(crate) fn publish(
    view: Res<View>,
    settings: Res<TemporalAntialiasing>,
    mut history: ResMut<History>,
    mut resolve: ResMut<Resolve>,
    mut piece: ResMut<TaaPiece>,
    mut stats: ResMut<TemporalAntialiasingStats>,
) {
    let temporal = history.temporal(&view);
    stats.set_if_neq(TemporalAntialiasingStats {
        active: temporal.is_some(),
    });
    let Some(temporal) = temporal else {
        resolve.set_if_neq(Resolve::default());
        return;
    };
    profile::counter(
        "render.taa_history",
        f64::from(u8::from(temporal.history_valid)),
    );
    resolve.set_if_neq(Resolve {
        offset: Vec2::from(temporal.offset),
        turn: temporal.turn,
    });
    piece.set_if_neq(TaaPiece {
        options: [
            temporal.turn,
            f32::from(u8::from(settings.nearest_motion)),
            f32::from(u8::from(settings.rounded_clamp)),
            settings.particle_response,
        ],
        offset: temporal.offset,
        current_weight: CURRENT_WEIGHT,
        history_valid: u32::from(temporal.history_valid),
    });
}
