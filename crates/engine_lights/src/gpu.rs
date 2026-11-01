engine_core::gpu_struct! {
    pub(crate) struct Clusters {
        pub(crate) view_projection: [[f32; 4]; 4],
        pub(crate) depth: [f32; 4],
        pub(crate) projection: [f32; 4],
        pub(crate) grid: [f32; 4],
        pub(crate) counts: [u32; 4],
        pub(crate) lights: [u32; 4],
        pub(crate) offsets: [u32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct LightInput {
        pub(crate) position: [f32; 3],
        pub(crate) range: f32,
        pub(crate) color: [f32; 3],
        pub(crate) slot: u32,
        pub(crate) direction: [f32; 3],
        pub(crate) cos_inner: f32,
        pub(crate) cos_outer: f32,
        pub(crate) radius: f32,
        pub(crate) length: f32,
        pub(crate) region: u32,
    }
}

engine_core::gpu_consts! {
    pub(crate) const NO_SLOT: u32 = 0xffffffff;
    pub(crate) const GROUP_LIGHTS: u32 = 8;
    pub(crate) const SHAPED_LIGHTS: u32 = 1;
    pub(crate) const KEPT_REGIONS: u32 = 2;
    pub(crate) const LIT_VIEWS: u32 = 7;
}
