use bytemuck::{Pod, Zeroable};

pub(crate) use super::constants::{LIGHT_HEIGHTS, LIGHT_TEXELS, MIRROR_LAYER, SHADOW_CELLS};
pub(crate) const CLOUD_SHADOW_BYTES: usize = (SHADOW_CELLS * SHADOW_CELLS * 2) as usize;
pub(crate) const CLOUD_LIGHT_BYTES: usize =
    (LIGHT_TEXELS * LIGHT_TEXELS * (LIGHT_HEIGHTS - 1) * 4) as usize;

gpu_structs! {
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct CloudUniform as ["CloudUniform"] {
        pub eye: [f32; 4],
        pub mirror: [f32; 4],
        pub layer: [f32; 4],
        pub drift: [f32; 4],
        pub dome: [u32; 4],
        pub sun: [f32; 4],
        pub sun_color: [f32; 4],
        pub zenith: [f32; 4],
        pub horizon: [f32; 4],
        pub shadow: [f32; 4],
        pub shadow_rows: [u32; 4],
        pub light: [f32; 4],
        pub light_rows: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct DomeUniform as ["CloudDomes"] {
        pub older: [f32; 4],
        pub newer: [f32; 4],
        pub older_mirror: [f32; 4],
        pub newer_mirror: [f32; 4],
        pub eye: [f32; 4],
        pub view: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct ViewUniform as ["CloudView"] {
        pub inverse_view_projection: [[f32; 4]; 4],
        pub pixels: [u32; 4],
    }
}
