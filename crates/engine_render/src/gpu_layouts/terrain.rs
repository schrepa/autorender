pub(crate) use super::constants::PALETTE_ENTRIES;
use bytemuck::{Pod, Zeroable};

pub(crate) const PALETTE_BYTES: usize = PALETTE_ENTRIES * size_of::<PaletteEntry>();

gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PaletteEntry as ["TerrainPaletteEntry", "PaletteEntry" in "cover::grow", "PaletteEntry" in "terrain::bounce_maps", "PaletteEntry" in "scatter::snap", "PaletteEntry" in "post::reflections"] {
        pub color: [f32; 4],
        pub params: [f32; 4],
        pub maps: [f32; 4],
        pub far: [f32; 4],
        pub relief: [f32; 4],
        pub canopy: [f32; 4],
        pub placed: [f32; 4],
        pub placed_size: [f32; 4],
        pub heights: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct CoverEntry as ["CoverSpec"] {
        pub shape: [f32; 4],
        pub color: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct PatchRecord {
        pub first: [i32; 2],
        pub shape: [f32; 2],
        pub atlas: [u32; 2],
        pub holes: u32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ViewUniform as ["TerrainView"] {
        pub relative_view_projection: [[f32; 4]; 4],
        pub eye_offset: [f32; 4],
        pub inverse_relative_view_projection: [[f32; 4]; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TerrainUniform as ["TerrainUniform"] {
        pub anchor: [f32; 4],
        pub eye: [f32; 4],
        pub surface: [u32; 4],
        pub origin: [f32; 4],
        pub directory: [u32; 4],
        pub relief: [f32; 4],
        pub cover: [f32; 4],
        pub macro_map: [u32; 4],
        pub steep: [f32; 4],
        pub pixel_error: [f32; 4],
        pub water: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SubdivisionUniform as ["TerrainSubdivision"] {
        pub square: [f32; 4],
        pub tree: [u32; 4],
        pub frame: [u32; 4],
        pub extent: [f32; 4],
        pub cutout: [f32; 4],
        pub planes: [[f32; 4]; 6],
    }
}
