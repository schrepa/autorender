use bytemuck::{Pod, Zeroable};

gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ClusterUniform as ["Clusters"] {
        pub view_projection: [[f32; 4]; 4],
        pub depth: [f32; 4],
        pub projection: [f32; 4],
        pub grid: [f32; 4],
        pub counts: [u32; 4],
        pub lights: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct LightInput as ["Input" in "lights::binning"] {
        pub position: [f32; 3],
        pub range: f32,
        pub color: [f32; 3],
        pub slot: u32,
        pub direction: [f32; 3],
        pub cos_inner: f32,
        pub cos_outer: f32,
        pub radius: f32,
        pub length: f32,
        pub region: u32,
    }
}
