use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ExposureUniform as ["ExposureUniform"] {
        pub frame: [f32; 4],
        pub reach: [f32; 4],
        pub meter: [f32; 4],
        pub day: [f32; 4],
        pub drawn: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ExposureState as ["ExposureState"] {
        pub exposure: f32,
        pub stops: f32,
        pub metered: f32,
        pub held: f32,
        pub view: f32,
        pub day: f32,
    }
}

engine_render::gpu_consts! {
    pub(crate) const EXPOSURE_BINS: u32 = 128 => ["EXPOSURE_BINS" in "post::exposure"];
    pub(crate) const EXPOSURE_GRID: u32 = 128 => ["EXPOSURE_GRID" in "post::exposure"];
    pub(crate) const EXPOSURE_GROUP: u32 = 16 => ["@meter_exposure" in "post::exposure"];
}
