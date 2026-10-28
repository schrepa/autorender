use serde::Serialize;

use super::queries::TerrainKey;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainHeightResult {
    pub samples: Vec<Option<HeightSample>>,
    pub stats: engine_core::TerrainStats,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct HeightSample {
    pub terrain: TerrainKey,
    pub height: f32,
    pub normal: [f32; 3],
    pub hole: bool,
    pub level: u32,
    pub water: Option<engine_core::terrain_data::HydrologySample>,
    pub standing_water: Option<StandingWater>,
    pub path: Option<f32>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StandingWater {
    pub surface: f32,
    pub depth: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainSweepResult {
    pub points: u32,
    pub resident: usize,
    pub level_zero: usize,
    pub holes: usize,
    pub collision: Option<SweepRays>,
    pub layers: SweepLayers,
    pub examples: Vec<SweepExample>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SweepRays {
    pub compared: usize,
    pub missed: usize,
    pub placed_first: usize,
    pub worst: Worst,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SweepLayers {
    pub compared: usize,
    pub worst: Worst,
}

#[derive(Default, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Worst {
    #[serde(rename = "max_difference")]
    pub difference: f32,
    pub at: Option<[f32; 2]>,
}

impl Worst {
    pub(super) fn add(&mut self, difference: f32, point: engine_core::glam::Vec2) {
        if difference > self.difference || self.at.is_none() {
            self.difference = engine_core::math::max(self.difference, difference);
            self.at = Some(point.to_array());
        }
    }
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SweepExample {
    pub at: [f32; 2],
    pub height: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainRouteResult {
    pub terrain: TerrainKey,
    pub options: engine_core::RouteOptions,
    pub origin: [f32; 2],
    pub route: Option<RouteFound>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RouteFound {
    pub points: Vec<[f64; 2]>,
    pub length: f32,
    pub steepest: f32,
    pub hairpin_steps: usize,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum TerrainEdited {
    River {
        temporary: bool,
        applied: bool,
        id: String,
        rivers: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        dirty: Option<Reach>,
    },
    Edit {
        temporary: bool,
        applied: bool,
        id: String,
        edits: usize,
        dirty: Reach,
        largest_change: f32,
    },
    Undo {
        temporary: bool,
        undone: Option<String>,
        edits: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        dirty: Option<Reach>,
    },
    Redo {
        temporary: bool,
        redone: Option<String>,
        edits: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        dirty: Option<Reach>,
    },
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Reach {
    pub min: [f32; 2],
    pub max: [f32; 2],
}
