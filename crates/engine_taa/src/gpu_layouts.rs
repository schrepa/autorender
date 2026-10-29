use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ResolveUniform as ["Resolve"] {
        pub texel: [f32; 2],
        pub current_weight: f32,
        pub history_valid: u32,
        pub options: [f32; 4],
        pub render: [f32; 4],
    }
}
