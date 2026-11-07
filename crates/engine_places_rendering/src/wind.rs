use bevy_ecs::prelude::Resource;
use engine_place::{Slot, gpu_struct};

gpu_struct! {
    #[derive(Resource)]
    pub struct ViewWind {
        pub velocity: [f32; 2],
        pub gusts: f32,
        pub pad: f32,
    }
}

impl Slot for ViewWind {
    const NAME: &'static str = "view_wind";
    const WGSL: Option<&'static str> = Some(include_str!("places/view_wind.wgsl"));
}
