use bevy_ecs::prelude::*;
use engine_core::{
    FrameTime, GpuStruct, Jitter, Settling, View, ViewFacts, ViewHistory, ViewJitter, Views,
    gpu_struct, profile,
};

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

#[derive(Resource)]
pub(crate) struct TaaSettling(Settling);

impl Default for TaaSettling {
    fn default() -> Self {
        Self(Settling::on_frames())
    }
}

fn cycled(facts: &ViewFacts) -> bool {
    match facts.history {
        ViewHistory::Fresh => false,
        ViewHistory::Valid { since } => facts.frame.0.saturating_sub(since.0) + 1 >= JITTER_SAMPLES,
    }
}

pub(crate) fn jitter(
    time: Res<FrameTime>,
    views: Query<(Entity, &View)>,
    mut jitter: ResMut<Jitter>,
) {
    let (offset, noise) = (jitter_offset(time.frame), turn_of(time.frame));
    let mut given: Vec<ViewJitter> = views
        .iter()
        .filter(|(_, view)| view.target.surface())
        .map(|(view, _)| ViewJitter {
            view,
            offset,
            noise,
        })
        .collect();
    given.sort_by_key(|given| given.view);
    jitter.set_if_neq(Jitter { views: given });
}

pub(crate) fn publish(
    views: Res<Views>,
    settings: Res<TemporalAntialiasing>,
    mut piece: ResMut<TaaPiece>,
    (mut stats, settling): (ResMut<TemporalAntialiasingStats>, Res<TaaSettling>),
) {
    let jittered = || views.listed().iter().filter(|facts| facts.jitter.is_some());
    let active = jittered().next().is_some();
    settling
        .0
        .left(u32::from(jittered().any(|facts| !cycled(facts))));
    stats.set_if_neq(TemporalAntialiasingStats { active });
    if !active {
        return;
    }
    let kept = jittered().all(|facts| facts.history.valid());
    profile::counter("render.taa_history", f64::from(u8::from(kept)));
    piece.set_if_neq(TaaPiece {
        options: [
            f32::from(u8::from(settings.nearest_motion)),
            f32::from(u8::from(settings.rounded_clamp)),
            settings.particle_response,
            CURRENT_WEIGHT,
        ],
    });
}
