//! Upload structs of `crate::terrain`.

pub(crate) use super::constants::PALETTE_ENTRIES;
use bytemuck::{Pod, Zeroable};

/// Bytes of the palette.
pub(crate) const PALETTE_BYTES: usize = PALETTE_ENTRIES * size_of::<PaletteEntry>();

gpu_structs! {
    /// A surface material as the terrain palette holds it: its linear color
    /// (its base color factor with maps) and roughness; its tiling, height
    /// contrast, projection and maps' layer plus one (0 without maps); its
    /// normal scale, occlusion strength, metalness and mean occlusion; its
    /// maps' mean color and roughness, its factors applied; the meters its
    /// relief spans, tracks sink it at most and its normal maps' slope
    /// variance; and its cover's canopy: the linear color its blades show
    /// seen from afar and their leaves' area per square meter of ground.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PaletteEntry as ["TerrainPaletteEntry", "PaletteEntry" in COVER_GROW_SHADER] {
        pub color: [f32; 4],
        pub params: [f32; 4],
        pub maps: [f32; 4],
        pub far: [f32; 4],
        pub relief: [f32; 4],
        pub canopy: [f32; 4],
    }

    /// A surface material's cover, at its palette entry's index: blades
    /// per square meter, the shortest and tallest blades and their root
    /// width, then its tip color and sway.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct CoverEntry as ["CoverSpec"] {
        pub shape: [f32; 4],
        pub color: [f32; 4],
    }

    /// A patch as `PatchInput` reads it.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct PatchRecord {
        pub first: [i32; 2],
        pub shape: [f32; 4],
        pub atlas: [u32; 4],
        pub holes: u32,
        pub fade: f32,
    }

    /// WGSL `TerrainView`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ViewUniform as ["TerrainView"] {
        pub relative_view_projection: [[f32; 4]; 4],
        pub eye_offset: [f32; 4],
        pub inverse_relative_view_projection: [[f32; 4]; 4],
    }

    /// WGSL `TerrainUniform`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TerrainUniform as ["TerrainUniform"] {
        pub anchor: [f32; 4],
        pub eye: [f32; 4],
        pub origin: [f32; 4],
        pub directory: [u32; 4],
        pub relief: [f32; 4],
        pub deform_grid: [i32; 4],
        pub deform: [f32; 4],
        pub cover: [f32; 4],
        pub vt: [u32; 4],
        pub vt_scale: [f32; 4],
        pub vt_layout: [u32; 4],
    }
}
