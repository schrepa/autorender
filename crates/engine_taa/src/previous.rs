use bevy_ecs::prelude::*;
use engine_core::View;
use engine_core::glam::UVec2;
use engine_render::{Layer, Output, wgpu};

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
    frame: u32,
    last: Option<(UVec2, u64)>,
    drawn: Option<(UVec2, u64)>,
}

pub(crate) struct Temporal {
    pub(crate) offset: [f32; 2],
    pub(crate) turn: f32,
    pub(crate) history_valid: bool,
}

impl History {
    pub(crate) fn temporal(&mut self, view: &View) -> Option<Temporal> {
        if let Some(drawn) = self.drawn.take()
            && view.submitted()
        {
            self.last = Some(drawn);
            self.frame = self.frame.wrapping_add(1);
        }
        if !view.temporal() {
            self.last = None;
            return None;
        }
        let shown = (view.display_size(), view.scene());
        let history_valid = self.last == Some(shown);
        self.drawn = Some(shown);
        Some(Temporal {
            offset: jitter_offset(self.frame),
            turn: turn_of(self.frame),
            history_valid,
        })
    }
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

pub(crate) fn forget(mut history: ResMut<History>, mut stats: ResMut<TemporalAntialiasingStats>) {
    *history = History::default();
    stats.set_if_neq(TemporalAntialiasingStats { active: false });
}
