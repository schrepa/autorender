use serde::{Deserialize, Serialize};

pub const MAX_GIVEN_RIVERS: usize = 4096;
pub const MAX_RIVER_POINTS: usize = 16_384;
pub const MAX_CREDIT: usize = 1024;
pub const MAX_WIDTH: f32 = 50_000.0;
pub const MAX_DEPTH: f32 = 500.0;
pub const MAX_DISCHARGE: f32 = 1.0e6;
pub const MAX_PRECISION: f32 = 1000.0;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GivenRiver {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = MAX_CREDIT))]
    pub name: Option<String>,
    #[schemars(length(min = 2, max = MAX_RIVER_POINTS))]
    pub points: Vec<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub width: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depth: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub discharge: Vec<f32>,
}

engine_codec::record!(GivenRiver {
    name,
    points,
    width,
    depth,
    discharge,
});
