//! Upload structs of `crate::particles`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// An emitter as `particles.wgsl` reads it.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct EmitterData as ["Emitter"] {
        pub origin: [f32; 3],
        pub rate: f32,
        pub velocity: [f32; 3],
        pub lifetime: f32,
        pub direction: [f32; 3],
        pub speed: f32,
        pub acceleration: [f32; 3],
        /// Cosine of the spread half-angle.
        pub spread: f32,
        pub start_color: [f32; 4],
        pub end_color: [f32; 4],
        pub size: [f32; 2],
        pub emission: f32,
        pub stretch: f32,
        pub radius: f32,
        pub seed: u32,
        pub first: u32,
        pub count: u32,
    }

    /// The particles' uniform of `particles.wgsl`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ParticleUniform as ["Particles"] {
        pub right: [f32; 4],
        pub up: [f32; 4],
        pub size: [f32; 4],
    }
}
