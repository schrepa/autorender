use bytemuck::{Pod, Zeroable};

pub(crate) const MAX_LEVELS: usize = engine_core::MAX_SHADOW_LEVELS as usize;
pub(crate) const MAX_MOVING_TILES: usize = engine_core::MAX_MOVING_TILES as usize;

const _: () = assert!(
    size_of::<RouteData>() == engine_faces::shadows::ROUTES_BYTES,
    "routes_bytes_match"
);

engine_render::gpu_consts! {
    pub(crate) const SHADOW_REGIONS: usize = 7 => ["REGIONS" in "shadows::maps"];
    pub(crate) const MOVING_MAP: usize = SHADOW_REGIONS => ["MOVING_MAP" in "shadows::maps"];
    pub(crate) const SKY_MAP: usize = SHADOW_REGIONS + 1 => ["SKY_MAP" in "shadows::maps"];
    pub(crate) const MOVING_PAIRS: u32 = 8 => ["MOVING_PAIRS" in "shadows::maps"];
    pub(crate) const MOVING_TILES: usize = MAX_MOVING_TILES => ["MOVING_TILES" in "shadows::maps"];
    pub(crate) const SHADOW_LEVELS: usize = MAX_LEVELS => ["SHADOW_LEVELS"];
    pub(crate) const SHADOW_TOTALS: usize = 4 => ["TOTALS" in "shadows::maps"];
    pub(crate) const SHADOW_GROUP: u32 = 256 => ["@select_casters" in "shadows::maps"];
    pub(crate) const SHADOW_MARKS_GROUP: u32 = 8 => ["@mark_shadow_tiles" in "shadows::marks"];
    pub(crate) const STILLNESS: u32 = 1 => ["STILLNESS" in "shadows::maps"];
    pub(crate) const SKY_REFRESH: u32 = 2 => ["SKY_REFRESH" in "shadows::maps"];
    pub(crate) const SKY_DIRECTIONS: usize = 13 => ["SKY_DIRECTIONS"];
    pub(crate) const SKY_LAYERS: usize = 2 * SKY_DIRECTIONS => ["SKY_LAYERS"];
    pub(crate) const SHADOW_APRON: u32 = 1 => ["SHADOW_APRON"];
}

engine_render::gpu_mirrors! {
    engine_render::gpu_data::DRAW_ARGS_BYTES => ["size_of DrawArgs" in "shadows::maps"];
}

engine_render::gpu_structs! {
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
    pub(crate) struct ShadowCullUniform as ["Cull" in "shadows::maps"] {
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
    pub(crate) struct RouteData as ["Routes" in "shadows::maps"] {
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
    pub(crate) struct SkyUniform as ["Sky"] {
        pub clips: [[[f32; 4]; 4]; SKY_LAYERS],
        pub directions: [[f32; 4]; SKY_LAYERS],
        pub counts: [f32; 4],
    }
}
