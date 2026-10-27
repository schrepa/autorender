//! Upload structs of `crate::occupancy`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `Splat` of `occupancy.wgsl`: one map's window and what a frame
    /// redraws of it. The window's first corner (world x and z), meters a
    /// side and the maps' reference height; the slots, draw groups, 0 and
    /// the map's texels a side; the texel its first corner falls on (x and
    /// y) and texels a cell side; the cells redrawn, a bit each; the finer
    /// map's window's first cell (x and z, in cells from the world's origin)
    /// and 1 where there is one; and the finer map's cells whose mean this
    /// map takes, a bit each from the finer window's first corner.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SplatUniform as ["Splat"] {
        pub area: [f32; 4],
        pub counts: [u32; 4],
        pub place: [u32; 4],
        pub dirty: [[u32; 4]; 2],
        pub finer: [i32; 4],
        pub means: [[u32; 4]; 2],
    }

    /// WGSL `Cover`: a draw group's footprint radius, the bottom and top of
    /// what it covers above its origin (mesh units), and its density.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct Cover as ["Cover" in OCCUPANCY_SHADER] {
        pub footprint: f32,
        pub bottom: f32,
        pub top: f32,
        pub density: f32,
    }

    /// WGSL `ListedDraw`: the arguments of the draw of the footprints a
    /// frame lists, whose instances the selection counts from none.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ListedDraw as ["ListedDraw" in OCCUPANCY_SHADER] {
        pub vertices: u32,
        pub instances: u32,
        pub first_vertex: u32,
        pub first_instance: u32,
    }
}
