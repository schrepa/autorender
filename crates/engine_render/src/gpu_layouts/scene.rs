use bytemuck::{Pod, Zeroable};

use super::constants::{SHADOW_REGIONS, SKY_LAYERS};

pub(crate) const MAX_LEVELS: usize = engine_core::MAX_SHADOW_LEVELS as usize;
pub(crate) const MAX_MOVING_TILES: usize = engine_core::MAX_MOVING_TILES as usize;
pub(crate) const MAX_OCCUPANCY_LEVELS: usize = engine_terrain::MAX_OCCUPANCY_LEVELS as usize;

pub(crate) type PoseData = engine_core::PoseRows;

pub(crate) type GroundData = [u32; 4];

gpu_structs! {
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct CameraUniform as ["Camera"] {
        pub view_projection: [[f32; 4]; 4],
        pub previous_view_projection: [[f32; 4]; 4],
        pub inverse_view_projection: [[f32; 4]; 4],
        pub position: [f32; 4],
        pub jitter: [f32; 4],
        pub options: [f32; 4],
        pub wind: [f32; 4],
        pub region: [f32; 4],
        pub detail: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct LightingUniform as ["Lighting"] {
        pub sun: [f32; 4],
        pub sun_color: [f32; 4],
        pub zenith: [f32; 4],
        pub horizon: [f32; 4],
        pub ground: [f32; 4],
        pub fog: [f32; 4],
        pub effects: [f32; 4],
        pub grade: [f32; 4],
        pub shadow_tint: [f32; 4],
        pub highlight_tint: [f32; 4],
        pub aerial: [f32; 4],
        pub aerial_color: [f32; 4],
        pub occupancy: [f32; 4],
        pub occupancy_maps: [f32; 4],
        pub occupancy_centers: [[f32; 4]; MAX_OCCUPANCY_LEVELS / 2],
        pub cloud_shadow: [f32; 4],
        pub cloud_layer: [f32; 4],
        pub night: [f32; 4],
        pub star_turn: [f32; 4],
        pub moon: [f32; 4],
        pub foliage: [f32; 4],
        pub highlights: [f32; 4],
        pub night_shape: [f32; 4],
        pub haze_march: [f32; 4],
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ColorData as ["SlotColor"] {
        pub color: [u32; 2],
        pub surface: u32,
        pub flags: u32,
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct Vertex {
        pub position: [f32; 3],
        pub normal: [i16; 2],
        pub uv: [f32; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ExtraVertex {
        pub color: [u16; 4],
        pub uv1: [f32; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct MaterialUniform as ["MaterialMaps"] {
        pub transforms: [[f32; 4]; 10],
        pub emissive: [f32; 4],
        pub params: [f32; 4],
        pub variant: [f32; 4],
        pub top: [f32; 4],
        pub layer: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ShadowUniform as ["Shadow"] {
        pub light: [[f32; 4]; 3],
        pub state: [f32; 4],
        pub level_state: [f32; 4],
        pub atlas: [f32; 4],
        pub windows: [[f32; 4]; MAX_LEVELS],
        pub bands: [[f32; 4]; MAX_LEVELS],
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ShadowCullUniform as ["Cull" in SHADOW_SHADER] {
        pub light: [[f32; 4]; 3],
        pub regions: [[f32; 4]; SHADOW_REGIONS],
        pub region_depths: [[f32; 4]; SHADOW_REGIONS],
        pub windows: [[f32; 4]; MAX_LEVELS],
        pub candidates: u32,
        pub groups: u32,
        pub capacity: u32,
        pub mode: u32,
        pub counts: [u32; 4],
        pub words: [u32; 4],
        pub moving: [u32; 4],
        pub sky: [[f32; 4]; 3],
        pub sky_rect: [f32; 4],
        pub sky_depths: [f32; 4],
    }

    #[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
    pub(crate) struct RouteData as ["Routes" in SHADOW_SHADER] {
        pub clips: [[[f32; 4]; 4]; MAX_MOVING_TILES],
        pub slots: [[f32; 4]; MAX_MOVING_TILES],
        pub atlas: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ComposeData {
        pub place: [f32; 4],
        pub source: [i32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PostUniform as ["Post"] {
        pub inverse_view_projection: [[f32; 4]; 4],
        pub view_projection: [[f32; 4]; 4],
        pub eye: [f32; 4],
        pub size: [f32; 4],
        pub extent: [f32; 4],
        pub occlusion: [f32; 4],
        pub bloom: [f32; 4],
        pub reflections: [f32; 4],
        pub haze: [f32; 4],
        pub previous_view_projection: [[f32; 4]; 4],
        pub previous_eye: [f32; 4],
        pub sun_direction: [f32; 4],
        pub sky: [f32; 4],
        pub detail: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SkyUniform as ["Sky" in POST_SHADER] {
        pub clips: [[[f32; 4]; 4]; SKY_LAYERS],
        pub directions: [[f32; 4]; SKY_LAYERS],
        pub counts: [f32; 4],
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ResolveUniform as ["Resolve"] {
        pub texel: [f32; 2],
        pub current_weight: f32,
        pub history_valid: u32,
        pub options: [f32; 4],
        pub render: [f32; 4],
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct PresentationFrame as ["Frame" in PRESENTATION_SHADER] {
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
    pub(crate) struct CandidateData as ["Candidate"] {
        pub slot: u32,
        pub node: u32,
        pub batch: u32,
        pub position: u32,
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct BatchData as ["Batch" in PRESENTATION_SHADER, "Batch" in SHADOW_SHADER] {
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
