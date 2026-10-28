use bytemuck::{Pod, Zeroable};

gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SkyUniform as ["SkyUniform"] {
        pub rayleigh: [f32; 4],
        pub mie: [f32; 4],
        pub ozone: [f32; 4],
        pub ground: [f32; 4],
        pub radii: [f32; 4],
        pub sun: [f32; 4],
        pub frame: [f32; 4],
        pub moon: [f32; 4],
    }
}
