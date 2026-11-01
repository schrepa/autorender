use engine_core::{gpu_consts, gpu_struct};

gpu_struct! {
    pub(crate) struct ExposureUniform {
        pub frame: [f32; 4],
        pub reach: [f32; 4],
        pub meter: [f32; 4],
        pub day: [f32; 4],
        pub drawn: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct ExposureState {
        pub exposure: f32,
        pub stops: f32,
        pub metered: f32,
        pub held: f32,
        pub view: f32,
        pub day: f32,
    }
}

gpu_consts! {
    pub(crate) const EXPOSURE_BINS: u32 = 128;
    pub(crate) const EXPOSURE_GRID: u32 = 128;
    pub(crate) const EXPOSURE_GROUP: u32 = 16;
}
