use bytemuck::{Pod, Zeroable};

gpu_structs! {
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
}
