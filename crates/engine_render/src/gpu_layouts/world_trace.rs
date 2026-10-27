//! Upload structs of `crate::world_trace`.

use bytemuck::{Pod, Zeroable};

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
    /// canopy of those grids stands (world; w unused).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TraceTerrain as ["TraceTerrain"] {
        pub info: TerrainInfo,
        pub cutout: [f32; 4],
        pub maps: [u32; 4],
        pub reach: [f32; 4],
    }
}
