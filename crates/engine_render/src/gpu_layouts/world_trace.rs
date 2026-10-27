//! Upload structs of `crate::world_trace`.

use bytemuck::{Pod, Zeroable};

use super::impostors::ImpostorRecord;
use super::scatter::TerrainInfo;

gpu_structs! {
    /// WGSL `TraceTerrain` of `reflections.wgsl`: a terrain as the world
    /// trace walks it. Its place, as setting objects on it reads it; the
    /// ground its surroundings' cutout leaves to the terrain they surround
    /// (terrain-local least x and z, then greatest; zeros without one); its
    /// sky map's layer plus one (0 until it arrives) and the levels whose
    /// tiles hold placed objects, a bit each (zw unused); and the widest
    /// apron of those tiles' grids (meters), ground cover's density, 0
    /// where it does not grow, and the highest its resident ground or a
    /// placed object of those grids stands (world; w unused).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TraceTerrain as ["TraceTerrain"] {
        pub info: TerrainInfo,
        pub cutout: [f32; 4],
        pub maps: [u32; 4],
        pub reach: [f32; 4],
    }

    /// WGSL `PlacedGroup` of `reflections.wgsl`: a draw group of placed
    /// objects as the world trace meets its objects. Its impostor in the
    /// placed objects' atlas (zeroed without one), whose views it reads
    /// and whose box bounds them; and its objects' linear color and
    /// opacity, packed as slots' colors are (`gpu_data::pack_color`), which
    /// the impostor's albedo leaves out (`pad` unused).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PlacedGroup as ["PlacedGroup" in REFLECTIONS_SHADER] {
        pub impostor: ImpostorRecord,
        pub color: [u32; 2],
        pub pad: [u32; 2],
    }
}
