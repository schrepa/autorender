use std::collections::BTreeMap;

use serde::Serialize;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FrameStats {
    pub frame: u64,
    pub tick: u64,
    pub simulation_ms: f64,
    pub extraction_ms: f64,
    pub render_ms: f64,
    pub fixed_steps: u32,
    pub dropped_seconds: f64,
    pub cut_seconds: f64,
    pub pipelines_listed: u32,
    pub pipelines_unlisted: u32,
    pub readiness: engine_core::Readiness,
    pub budget: Vec<engine_core::BudgetUse>,
    pub resolution: ResolutionStats,
    pub dynamic_resolution: DynamicResolutionStats,
    pub jobs: engine_core::JobStats,
    pub refused_views: u32,
    pub disabled_features: engine_core::Names,
    #[serde(flatten)]
    pub render: engine_core::RenderStats,
    pub bricks_cost: Option<engine_registry::BrickCosts>,
    #[serde(flatten)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub bricks: BTreeMap<&'static str, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResolutionStats {
    pub display_width: u32,
    pub display_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub dropped_pixels: u64,
    pub gpu_ms: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicResolutionStats {
    pub active: bool,
    pub scale: f32,
    pub budget_ms: f32,
}
