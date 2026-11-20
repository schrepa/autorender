use engine_drawing::DetailLimit;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::light::Filtering;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SkyNumbers {
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    pub below: [f32; 3],
    pub haze_height: f32,
    pub disc_degrees: f32,
    pub disc_soft_degrees: f32,
    pub disc_strength: f32,
    pub glow_power: f32,
    pub glow_strength: f32,
    pub mirror_roughness: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FogNumbers {
    pub start_m: f32,
    pub density_per_m: f32,
    pub most: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GradeNumbers {
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BlobNumbers {
    pub radius_scale: f32,
    pub opacity: f32,
    pub fade_height_m: f32,
    pub depth_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AmbientNumbers {
    pub sky_scale: f32,
    pub ground_scale: f32,
    pub base_occlusion: f32,
    pub base_height_m: f32,
    pub floor_m: f32,
    pub black_floor: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LightingNumbers {
    pub sun_wrap: f32,
    pub sky: SkyNumbers,
    pub fog: FogNumbers,
    pub grade: GradeNumbers,
    pub blob: BlobNumbers,
    #[serde(default)]
    pub ambient: Option<AmbientNumbers>,
    pub filtering: Filtering,
    pub detail: DetailLimit,
    pub visible_sets: bool,
}

#[derive(Deserialize)]
struct FilePart {
    lighting: LightingNumbers,
}

pub(super) fn parse(whole: &serde_json::Value) -> Result<LightingNumbers, String> {
    let lighting = FilePart::deserialize(whole)
        .map(|part| part.lighting)
        .map_err(|error| error.to_string())?;
    lighting.filtering.checked()?;
    lighting.detail.checked("$.lighting.detail")?;
    Ok(lighting)
}
