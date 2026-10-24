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

    /// WGSL `FarMean` of `terrain_vt_far.wgsl`: a coarser far page made as
    /// the mean of the next finer far mip's pages under it: its page along
    /// x and z, the finer mip's last page along x and z, the slots of the
    /// finer mip's pages from one before twice its page on (four rows of
    /// four along x, rows along z, held within the finer mip's pages), and
    /// its own slot.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct FarMeanRecord as ["FarMean" in TERRAIN_VT_FAR_SHADER] {
        pub page: [i32; 2],
        pub last: [i32; 2],
        pub fine: [[u32; 4]; 4],
        pub slot: u32,
        pub pad: [u32; 3],
    }
}
