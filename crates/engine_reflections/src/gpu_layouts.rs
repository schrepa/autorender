use bytemuck::{Pod, Zeroable};

use engine_render::gpu_layouts::impostors::ImpostorRecord;

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct ReflectionsPiece as ["ReflectionsPiece" in "reflections::piece"] {
        pub surface: [f32; 4],
        pub trace: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TerrainInfo as ["TerrainInfo" in "post::reflections"] {
        pub origin: [f32; 4],
        pub levels: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TraceTerrain as ["TraceTerrain"] {
        pub info: TerrainInfo,
        pub cutout: [f32; 4],
        pub maps: [u32; 4],
        pub reach: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PlacedGroup as ["PlacedGroup" in "post::reflections"] {
        pub impostor: ImpostorRecord,
        pub color: [u32; 2],
        pub pad: [u32; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PlacedRecord as ["Record" in "post::reflections"] {
        pub center: [f32; 3],
        pub radius: f32,
        pub group: u32,
        pub scale: f32,
        pub terrain: u32,
        pub sink: f32,
        pub state: u32,
        pub pad: [u32; 3],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PaletteEntry as ["PaletteEntry" in "post::reflections"] {
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
}

engine_render::gpu_consts! {
    pub(crate) const NO_PLACED: u32 = u32::MAX => ["NO_PLACED" in "post::reflections"];
    pub(crate) const IMPOSTOR_KINDS: u32 = engine_render::gpu_layouts::constants::IMPOSTOR_KINDS => ["IMPOSTOR_KINDS" in "post::reflections"];
    pub(crate) const PEAK_MIPS: u32 = engine_terrain::ERROR_MIPS => ["PEAK_MIPS" in "post::reflections"];
    pub(crate) const PEAK_CELLS: u32 = engine_terrain::ERROR_CELLS => ["PEAK_CELLS" in "post::reflections"];
    pub(crate) const TILE_QUADS: u32 = engine_terrain::TILE_QUADS => ["TILE_QUADS" in "post::reflections"];
    pub(crate) const PALETTE_ENTRIES: usize = 16_384 / size_of::<PaletteEntry>() => ["PALETTE_ENTRIES" in "post::reflections"];
}
