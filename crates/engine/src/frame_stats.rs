use bevy_ecs::resource::Resource;
use serde::Serialize;

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize)]
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
    pub readiness: engine_core::Readiness,
    pub resolution: ResolutionStats,
    pub refused_views: u32,
    #[serde(flatten)]
    pub render: engine_core::RenderStats,
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
    pub gpu_ms: GpuTime,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum GpuTime {
    Ms(f32),
    Unavailable { unavailable: &'static str },
}

impl Default for GpuTime {
    fn default() -> Self {
        Self::Unavailable {
            unavailable: "not_yet_read",
        }
    }
}
