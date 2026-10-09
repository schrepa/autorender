//! Upload structs of `crate::clouds`.

use bytemuck::{Pod, Zeroable};

pub(crate) use super::constants::SHADOW_CELLS;
/// Bytes of the shadows' grid: a half float a cell.
pub(crate) const CLOUD_SHADOW_BYTES: usize = (SHADOW_CELLS * SHADOW_CELLS * 2) as usize;

gpu_structs! {
    /// WGSL `CloudUniform`.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct CloudUniform as ["CloudUniform"] {
        pub eye: [f32; 4],
        pub layer: [f32; 4],
        pub drift: [f32; 4],
        pub dome: [u32; 4],
        pub sun: [f32; 4],
        pub sun_color: [f32; 4],
        pub zenith: [f32; 4],
        pub horizon: [f32; 4],
        pub shadow: [f32; 4],
        pub shadow_rows: [u32; 4],
    }

    /// WGSL `CloudDomes` (`scene.wgsl`): the shown domes' origins, whether they
    /// show and the newer's share, and the eye lookups correct for with the
    /// cloud layer's middle.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct DomeUniform as ["CloudDomes"] {
        pub older: [f32; 4],
        pub newer: [f32; 4],
        pub eye: [f32; 4],
        pub view: [f32; 4],
    }

    /// WGSL `CloudView` (`clouds.wgsl`): the view's unjittered
    /// inverse view-projection; its blocks along x and y, pixels along a
    /// block's side, and whether blocks seeing the sky march.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct ViewUniform as ["CloudView"] {
        pub inverse_view_projection: [[f32; 4]; 4],
        pub blocks: [u32; 4],
    }
}
