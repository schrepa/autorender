use bevy_ecs::prelude::*;
use engine_core::gpu_struct;
use engine_render::{Output, Resolve, wgpu};

use crate::antialiasing::{TemporalAntialiasing, TemporalAntialiasingStats};
use crate::previous::History;

const CURRENT_WEIGHT: f32 = 0.1;

pub(crate) const HISTORY: Output = Output {
    name: "history",
    format: wgpu::TextureFormat::Rgba8UnormSrgb,
    layers: &[],
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

pub(crate) fn publish_resolve(
    settings: Res<TemporalAntialiasing>,
    history: Res<History>,
    resolve: Res<Resolve>,
    mut piece: ResMut<TaaPiece>,
    mut stats: ResMut<TemporalAntialiasingStats>,
) {
    piece.set_if_neq(TaaPiece {
        options: [
            resolve.turn,
            f32::from(u8::from(settings.nearest_motion)),
            f32::from(u8::from(settings.rounded_clamp)),
            settings.particle_response,
        ],
        offset: resolve.offset.to_array(),
        current_weight: CURRENT_WEIGHT,
        history_valid: u32::from(history.valid),
    });
    stats.set_if_neq(TemporalAntialiasingStats {
        active: history.live,
    });
}
