use bytemuck::{Pod, Zeroable};

pub(crate) const MAX_OCCUPANCY_LEVELS: usize = engine_terrain::MAX_OCCUPANCY_LEVELS as usize;

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct FrameUniform as ["Frame" in "scatter::cull"] {
        pub eye: [f32; 4],
        pub counts: [u32; 4],
        pub strides: [u32; 4],
        pub fade: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ViewData as ["View" in "scatter::cull"] {
        pub planes: [[f32; 4]; 6],
        pub limits: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SnapUniform as ["Snap"] {
        pub counts: [u32; 4],
        pub eye: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct GroupData as ["Group" in "scatter::cull"] {
        pub ids: u32,
        pub count: u32,
        pub levels: u32,
        pub camera: u32,
        pub errors: [f32; 4],
        pub impostor: f32,
        pub record: u32,
        pub casters: u32,
        pub parts: u32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct DrawArgs as ["DrawArgs" in "scatter::cull"] {
        pub index_count: u32,
        pub instance_count: u32,
        pub first_index: u32,
        pub base_vertex: i32,
        pub first_instance: u32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SplatUniform as ["Splat"] {
        pub area: [f32; 4],
        pub counts: [u32; 4],
        pub place: [u32; 4],
        pub dirty: [[u32; 4]; 2],
        pub finer: [i32; 4],
        pub means: [[u32; 4]; 2],
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct Cover as ["Cover" in "terrain::occupancy_maps"] {
        pub footprint: f32,
        pub bottom: f32,
        pub top: f32,
        pub density: f32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ListedDraw as ["ListedDraw" in "terrain::occupancy_maps"] {
        pub vertices: u32,
        pub instances: u32,
        pub first_vertex: u32,
        pub first_instance: u32,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct CanopyPiece as ["CanopyPiece" in "scatter::piece"] {
        pub occupancy: [f32; 4],
        pub occupancy_maps: [f32; 4],
        pub occupancy_centers: [[f32; 4]; MAX_OCCUPANCY_LEVELS / 2],
        pub foliage: [f32; 4],
    }
}

engine_render::gpu_consts! {
    pub(crate) const SCATTER_LEVELS: u32 = 5 => ["SHOWN_CARD" in "scatter::cull"];
    pub(crate) const SHOWN_NONE: u32 = 6 => ["SHOWN_NONE" in "scatter::cull"];
    pub(crate) const FADED: u32 = 65535 => ["FADED" in "scatter::cull"];
    pub(crate) const STATE_ADOPT: u32 = u32::MAX => ["STATE_ADOPT" in "scatter::cull"];
    pub(crate) const DEAD: u32 = engine_faces::placed::DEAD => ["DEAD"];
    pub(crate) const TERRAIN_INDEX: u32 = 0x1ff => ["TERRAIN_INDEX"];
    pub(crate) const LAYER_SHIFT: u32 = 9 => ["LAYER_SHIFT"];
    pub(crate) const NO_LAYER: u32 = 0x7f => ["NO_LAYER"];
    pub(crate) const NEAR_LEVEL_SHIFT: u32 = 16 => ["NEAR_LEVEL_SHIFT"];
    pub(crate) const RANK_SHIFT: u32 = 21 => ["RANK_SHIFT" in "scatter::cull"];

    pub(crate) const OCCUPANCY_CELLS: u32 = 16 => ["CELLS" in "terrain::occupancy_maps"];
    pub(crate) const OCCUPANCY_QUAD: u32 = 6 => ["QUAD" in "terrain::occupancy_maps"];

    pub(crate) const SCATTER_GROUP: u32 = 64 => ["@snap_scatter" in "scatter::snap", "@cull_scatter" in "scatter::cull"];
    pub(crate) const OCCUPANCY_GROUP: u32 = 64 => ["@select_covers" in "terrain::occupancy_maps"];
}
