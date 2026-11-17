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
    pub preparation_ms: f64,
    pub dropped_seconds: f64,
    pub cut_seconds: f64,
    pub pipelines_listed: u32,
    pub pipelines_unlisted: u32,
    pub readiness: engine_core::Readiness,
    pub resolution: ResolutionStats,
    pub jobs: engine_core::JobStats,
    pub refused_views: u32,
    #[serde(flatten)]
    pub render: engine_core::RenderStats,
    pub bricks: BTreeMap<&'static str, serde_json::Value>,
    pub pipelines_per_pass: BTreeMap<&'static str, PassPipelines>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PassPipelines {
    pub render: u32,
    pub compute: u32,
}

#[cfg(feature = "schema")]
pub fn frame_stats_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(FrameStats)).expect("schema_serializes")
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResolutionStats {
    pub display_width: u32,
    pub display_height: u32,
    pub dropped_pixels: u64,
    pub gpu_ms: Option<f32>,
}
