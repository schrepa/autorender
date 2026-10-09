//! Upload structs of `crate::terrain_vt`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `TerrainVtBake`: a page's slot, mip and page along x and z.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct BakeRecord as ["TerrainVtBake"] {
        pub slot: u32,
        pub mip: u32,
        pub page: [i32; 2],
    }
}
