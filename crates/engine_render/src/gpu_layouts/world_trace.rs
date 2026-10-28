use bytemuck::{Pod, Zeroable};

use super::impostors::ImpostorRecord;
use super::scatter::TerrainInfo;

gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TraceTerrain as ["TraceTerrain"] {
        pub info: TerrainInfo,
        pub cutout: [f32; 4],
        pub maps: [u32; 4],
        pub reach: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PlacedGroup as ["PlacedGroup" in REFLECTIONS_SHADER] {
        pub impostor: ImpostorRecord,
        pub color: [u32; 2],
        pub pad: [u32; 2],
    }
}
