use bevy_ecs::prelude::Resource;
use engine_place::{Slot, gpu_struct};

gpu_struct! {
    #[derive(Resource)]
    pub struct Display {
        pub grade: [f32; 4],
        pub shadow_tint: [f32; 4],
        pub highlight_tint: [f32; 4],
        pub exposure: [f32; 4],
    }
}

impl Slot for Display {
    const NAME: &'static str = "tonemap";
    const WGSL: Option<&'static str> = Some(include_str!("places/tonemap.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::TONEMAP];
}
