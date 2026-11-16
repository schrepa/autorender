use engine_core::{GpuStruct, gpu_consts, gpu_struct};

pub(crate) const CLOUD_LIGHT_BYTES: usize =
    (LIGHT_TEXELS * LIGHT_TEXELS * (LIGHT_HEIGHTS - 1) * 4) as usize;

gpu_struct! {
    pub(crate) struct CloudUniform {
        pub eye: [f32; 4],
        pub mirror: [f32; 4],
        pub layer: [f32; 4],
        pub drift: [f32; 4],
        pub dome: [u32; 4],
        pub sun: [f32; 4],
        pub zenith: [f32; 4],
        pub horizon: [f32; 4],
        pub light: [f32; 4],
        pub light_rows: [u32; 4],
        pub layers: [u32; 4],
        pub counts: [u32; 4],
        pub march: [f32; 4],
        pub sampling: [f32; 4],
        pub weather: [f32; 4],
        pub bank: [f32; 4],
        pub gather: [f32; 4],
        pub profile: [f32; 4],
        pub top: [f32; 4],
        pub phase: [f32; 4],
        pub lighting: [f32; 4],
        pub sky_light: [f32; 4],
        pub high: [f32; 4],
        pub high_fibre: [f32; 4],
        pub high_turn: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SkyPiece {
        pub zenith: [f32; 4],
        pub horizon: [f32; 4],
        pub ground: [f32; 4],
        pub night: [f32; 4],
        pub star_turn: [f32; 4],
        pub moon: [f32; 4],
        pub night_shape: [f32; 4],
        pub sun_light: [f32; 4],
        pub domes_older: [f32; 4],
        pub domes_newer: [f32; 4],
        pub domes_older_mirror: [f32; 4],
        pub domes_newer_mirror: [f32; 4],
        pub domes_eye: [f32; 4],
        pub domes_view: [f32; 4],
        pub domes_layers: [u32; 4],
        pub domes_size: [f32; 4],
        pub sky_flat: [f32; 4],
        pub sky_x: [f32; 4],
        pub sky_y: [f32; 4],
        pub sky_z: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SkyDomes {
        pub layers: [u32; 4],
        pub size: [f32; 4],
        pub taps: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SkyAir {
        pub rayleigh: [f32; 4],
        pub mie: [f32; 4],
        pub ozone: [f32; 4],
        pub ground: [f32; 4],
        pub radii: [f32; 4],
        pub sun: [f32; 4],
        pub moon: [f32; 4],
    }
}

gpu_consts! {
    pub(crate) const LIGHT_TEXELS: u32 = 384;
    pub(crate) const LIGHT_HEIGHTS: u32 = 16;
    pub(crate) const LIGHT_STEPS: u32 = 5;
    pub(crate) const LIGHT_NEAR_STEPS: u32 = 2;
    pub(crate) const EYE_LAYER: u32 = 0;
    pub(crate) const MIRROR_LAYER: u32 = 1;
    pub(crate) const DOME_LAYERS: u32 = MIRROR_LAYER + 1;
    pub(crate) const SKY_VIEW_LAYER: u32 = 0;
    pub(crate) const DOME_FIRST: u32 = SKY_VIEW_LAYER + 1;
    pub(crate) const CLOUD_GROUP: u32 = 8;
    pub(crate) const SKY_TABLE_GROUP: u32 = 8;
    pub(crate) const SKY_VIEW_WIDTH: u32 = 192;
    pub(crate) const SKY_VIEW_HEIGHT: u32 = 108;
    pub(crate) const CLOUD_HORIZON: f32 = 0.05;
    pub(crate) const CLOUD_REACH: f32 = 60000.0;
    pub(crate) const SKY_MULTIPLE_SIZE: f32 = 32.0;
}

pub(crate) const GENERATED: &[&str] = &[
    WGSL,
    <CloudUniform as GpuStruct>::WGSL,
    <SkyPiece as GpuStruct>::WGSL,
    <SkyDomes as GpuStruct>::WGSL,
    <SkyAir as GpuStruct>::WGSL,
];
