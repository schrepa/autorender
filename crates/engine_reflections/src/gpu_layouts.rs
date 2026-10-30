use bytemuck::{Pod, Zeroable};

use engine_faces::heightfield::TerrainInfo;
use engine_render::gpu_layouts::impostors::ImpostorRecord;

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct ReflectionsPiece as ["ReflectionsPiece" in "reflections::piece"] {
        pub surface: [f32; 4],
        pub trace: [f32; 4],
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
}

engine_render::gpu_consts! {
    pub(crate) const NO_PLACED: u32 = u32::MAX => ["NO_PLACED" in "post::reflections"];
    pub(crate) const IMPOSTOR_KINDS: u32 = engine_render::gpu_layouts::constants::IMPOSTOR_KINDS => ["IMPOSTOR_KINDS" in "post::reflections"];
}
