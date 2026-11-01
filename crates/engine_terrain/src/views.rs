use serde::Serialize;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainView {
    #[cfg_attr(not(target_arch = "wasm32"), schemars(with = "serde_json::Value"))]
    pub terrain: crate::scene::SceneTerrain,
    pub levels: u32,
    pub extent: [f32; 2],
    pub resident_tiles_by_level: Vec<u32>,
    pub level_errors: Vec<f32>,
    pub level_ranges: Vec<f32>,
    pub hydrology: Option<HydrologyView>,
    pub erosion: Option<ErosionView>,
    pub paths: Vec<PathView>,
    pub edits: Vec<TerrainEditView>,
    pub rivers: Vec<TerrainRiverView>,
    pub surroundings: Option<SurroundingsView>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SurroundingsView {
    pub reach: [f32; 2],
    pub tiles: u32,
    pub spacing: f32,
    pub resident_tiles: u32,
    pub drawn: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainEditView {
    pub id: String,
    pub stamp: crate::Stamp,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainRiverView {
    pub id: String,
    pub river: crate::GivenRiver,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct PathView {
    pub layer: usize,
    pub baked: bool,
    pub length: Option<f32>,
    pub samples: Option<usize>,
    pub steepest: Option<f32>,
    pub deepest_cut: Option<f32>,
    pub highest_fill: Option<f32>,
    pub bridges: Option<usize>,
    pub bridged: Option<f32>,
    pub tunnels: Option<usize>,
    pub tunneled: Option<f32>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ErosionView {
    pub baking: bool,
    pub samples: Option<usize>,
    pub rounds: Option<u32>,
    pub deepest: Option<f32>,
    pub highest: Option<f32>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct HydrologyView {
    pub baking: bool,
    pub samples: Option<usize>,
    pub channel_samples: Option<usize>,
    pub lake_samples: Option<usize>,
    pub max_area: Option<f32>,
    pub falls: Option<usize>,
    pub given: Option<usize>,
    pub uncut: Option<usize>,
    pub flow: Option<crate::RiverFlowView>,
    pub fall_list: Vec<crate::FallView>,
    pub rivers: Vec<RiverView>,
}

pub const RIVERS_SHOWN: usize = 16;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RiverView {
    pub name: Option<String>,
    pub given: Option<[u32; 2]>,
    pub courses: usize,
    pub length: f32,
    pub first: [f32; 2],
    pub last: [f32; 2],
    pub width: [f32; 2],
    pub depth: [f32; 2],
    pub flood: [f32; 2],
    pub given_width: bool,
    pub given_depth: bool,
    pub given_flood: bool,
    pub uncut: f32,
}
