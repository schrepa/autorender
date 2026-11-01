use bevy_ecs::prelude::*;
use engine_core::View;
use engine_core::glam::{UVec2, Vec2};
use engine_render::{DebugShown, Layer, Output, Resolve, wgpu};

use crate::antialiasing::TemporalAntialiasingStats;

const JITTER_SAMPLES: u32 = 8;

pub(crate) const MOTION: Output = Output {
    name: "motion",
    format: wgpu::TextureFormat::Rg16Float,
    layers: &[
        Layer::Occluders,
        Layer::Opaque,
        Layer::Late,
        Layer::Backdrop,
        Layer::Media,
    ],
};

#[derive(Resource, Default)]
pub(crate) struct History {
    frames: u32,
    last: Option<UVec2>,
    pub(crate) valid: bool,
    pub(crate) live: bool,
}

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut scale, mut value) = (1.0, 0.0);
    while index > 0 {
        scale /= base as f32;
        value += scale * (index % base) as f32;
        index /= base;
    }
    value
}

fn jitter_offset(frame: u32) -> [f32; 2] {
    let index = frame % JITTER_SAMPLES + 1;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
}

fn turn_of(frame: u32) -> f32 {
    let turn = (u64::from(frame) % 1024) as f32 * 0.618_034;
    turn - turn.floor()
}

pub(crate) fn publish_jitter(
    view: Res<View>,
    shown: Res<DebugShown>,
    mut history: ResMut<History>,
    mut resolve: ResMut<Resolve>,
) {
    if shown.owner.is_some() {
        history.last = None;
        history.valid = false;
        history.live = false;
        resolve.set_if_neq(Resolve::default());
        return;
    }
    let frame = history.frames;
    let size = view.render_size();
    history.valid = history.last == Some(size);
    history.last = Some(size);
    history.live = true;
    history.frames = frame.wrapping_add(1);
    resolve.set_if_neq(Resolve {
        offset: Vec2::from(jitter_offset(frame)),
        turn: turn_of(frame),
    });
}

pub(crate) fn forget(mut history: ResMut<History>, mut stats: ResMut<TemporalAntialiasingStats>) {
    *history = History::default();
    stats.set_if_neq(TemporalAntialiasingStats { active: false });
}
