#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

engine::brick!();

use engine::prelude::*;
pub use engine_touch::{
    ButtonVisual, LookPadSpec, StickVisual, TouchAnchor, TouchButtonSpec, TouchLayout, TouchState,
    VirtualStickSpec,
};

pub struct TouchBrick;

impl Brick for TouchBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<TouchLayout>()
            .resource::<TouchState>()
            .systems(Schedule::BeforeTicks, update_touch)
            .systems(Schedule::FixedSimulation, step_touch);
    }
}

fn update_touch(
    devices: Res<Devices>,
    profile: Res<InputProfile>,
    views: Res<Views>,
    layout: Res<TouchLayout>,
    mut state: ResMut<TouchState>,
) {
    state.update(&devices, *profile, views.frame(), &layout);
}

fn step_touch(mut state: ResMut<TouchState>) {
    state.step_tick();
}
