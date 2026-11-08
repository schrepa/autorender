use engine_core::{GpuStruct, gpu_consts, gpu_struct};

pub(crate) const CLOUD_SHADOW_BYTES: usize = (CLOUD_SHADOW_CELLS * CLOUD_SHADOW_CELLS * 2) as usize;
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
        pub shadow: [f32; 4],
        pub light: [f32; 4],
        pub light_rows: [u32; 4],
        pub layers: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SkyPiece {
        pub zenith: [f32; 4],
        pub horizon: [f32; 4],
        pub ground: [f32; 4],
        pub aerial: [f32; 4],
        pub aerial_color: [f32; 4],
        pub cloud_shadow: [f32; 4],
        pub cloud_layer: [f32; 4],
        pub night: [f32; 4],
        pub star_turn: [f32; 4],
        pub moon: [f32; 4],
        pub night_shape: [f32; 4],
        pub haze_march: [f32; 4],
        pub sun_light: [f32; 4],
        pub domes_older: [f32; 4],
        pub domes_newer: [f32; 4],
        pub domes_older_mirror: [f32; 4],
        pub domes_newer_mirror: [f32; 4],
        pub domes_eye: [f32; 4],
        pub domes_view: [f32; 4],
        pub domes_layers: [u32; 4],
        pub haze: [f32; 4],
        pub wind: [f32; 4],
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
    pub(crate) const CLOUD_SHADOW_CELLS: u32 = 128;
    pub(crate) const LIGHT_TEXELS: u32 = 384;
    pub(crate) const LIGHT_HEIGHTS: u32 = 16;
    pub(crate) const EYE_LAYER: u32 = 0;
    pub(crate) const MIRROR_LAYER: u32 = 1;
    pub(crate) const DOME_LAYERS: u32 = MIRROR_LAYER + 1;
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
    <SkyAir as GpuStruct>::WGSL,
];
