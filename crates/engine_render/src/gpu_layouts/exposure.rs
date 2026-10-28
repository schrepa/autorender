use bytemuck::{Pod, Zeroable};

gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ExposureUniform as ["ExposureUniform"] {
        pub frame: [f32; 4],
        pub reach: [f32; 4],
        pub meter: [f32; 4],
        pub day: [f32; 4],
        pub drawn: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ExposureState as ["ExposureState"] {
        pub exposure: f32,
        pub stops: f32,
        pub metered: f32,
        pub held: f32,
        pub view: f32,
        pub day: f32,
    }
}
