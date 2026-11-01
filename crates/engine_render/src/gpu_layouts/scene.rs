use bytemuck::{Pod, Zeroable};

pub const MAX_LEVELS: usize = engine_core::MAX_SHADOW_LEVELS as usize;

pub type PoseData = engine_core::PoseRows;

pub type GroundData = [u32; 4];

gpu_structs! {
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct ColorData as ["SlotColor"] {
        pub color: [u32; 2],
        pub surface: u32,
        pub flags: u32,
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct Vertex {
        pub position: [f32; 3],
        pub normal: [i16; 2],
        pub uv: [f32; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct ExtraVertex {
        pub color: [u16; 4],
        pub uv1: [f32; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct MaterialUniform as ["MaterialMaps"] {
        pub transforms: [[f32; 4]; 10],
        pub emissive: [f32; 4],
        pub params: [f32; 4],
        pub variant: [f32; 4],
        pub top: [f32; 4],
        pub layer: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct PostUniform as ["Post"] {
        pub inverse_view_projection: [[f32; 4]; 4],
        pub view_projection: [[f32; 4]; 4],
        pub eye: [f32; 4],
        pub size: [f32; 4],
        pub extent: [f32; 4],
        pub occlusion: [f32; 4],
        pub previous_view_projection: [[f32; 4]; 4],
        pub previous_eye: [f32; 4],
        pub sky: [f32; 4],
        pub detail: [f32; 4],
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct PresentationFrame as ["Frame" in "presentation::cull"] {
        pub alpha: f32,
        pub candidates: u32,
        pub budget: u32,
        pub culling: u32,
        pub groups: u32,
        pub batches: u32,
        pub node_capacity: u32,
        pub blend: u32,
        pub planes: [[f32; 4]; 6],
        pub absolute_view_projection: [[f32; 4]; 4],
        pub motion: u32,
        pub mask: u32,
        pub occlusion: u32,
        pub subpixel: u32,
        pub view_projection: [[f32; 4]; 4],
        pub viewport: [f32; 4],
        pub plane: u32,
        pub lod_error: f32,
        pub pad: [u32; 2],
        pub eye: [f32; 4],
        pub fade: [f32; 4],
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct CandidateData as ["Candidate"] {
        pub slot: u32,
        pub node: u32,
        pub batch: u32,
        pub position: u32,
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct BatchData as ["Batch" in "presentation::cull"] {
        pub start: u32,
        pub count: u32,
        pub index_count: u32,
        pub group: u32,
        pub bounds_min: [f32; 4],
        pub bounds_max: [f32; 4],
        pub first_index: u32,
        pub base_vertex: i32,
        pub first_instance: u32,
        pub first_plane: u32,
        pub lod_ranges: [u32; 4],
        pub lod_errors: [f32; 4],
    }
}
