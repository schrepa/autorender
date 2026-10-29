use engine_settings::{ExtraRules, InvalidSetting, settings};
use serde::{Deserialize, Serialize};

pub const MAX_GIVEN_RIVERS: usize = 4096;
pub const MAX_RIVER_POINTS: usize = 16_384;
pub const MAX_CREDIT: usize = 1024;
pub const MAX_WIDTH: f32 = 50_000.0;
pub const MAX_DEPTH: f32 = 500.0;
pub const MAX_DISCHARGE: f32 = 1.0e6;
pub const MAX_PRECISION: f32 = 1000.0;

settings! {
    extra_rules;
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct GivenRiver {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "schema", schemars(length(max = MAX_CREDIT)))]
        pub name: Option<String> = None => any,
        #[cfg_attr(feature = "schema", schemars(length(min = 2, max = MAX_RIVER_POINTS)))]
        pub points: Vec<[f32; 2]> = Vec::new()
            => range(-crate::Layers::MAX_COORDINATE, crate::Layers::MAX_COORDINATE),
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub width: Vec<f32> = Vec::new() => above_to(0.0, MAX_WIDTH),
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub depth: Vec<f32> = Vec::new() => above_to(0.0, MAX_DEPTH),
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub discharge: Vec<f32> = Vec::new() => above_to(0.0, MAX_DISCHARGE),
    }
}

impl ExtraRules for GivenRiver {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if !(2..=MAX_RIVER_POINTS).contains(&self.points.len()) {
            return Err(
                InvalidSetting::new("points", engine_settings::Expected::Array).allowed(
                    engine_settings::Allowed::Range {
                        min: 2.0,
                        max: MAX_RIVER_POINTS as f64,
                    },
                ),
            );
        }
        for (field, values) in [
            ("width", &self.width),
            ("depth", &self.depth),
            ("discharge", &self.discharge),
        ] {
            if !(values.len() <= 1 || values.len() == self.points.len()) {
                return Err(InvalidSetting::new(
                    field,
                    engine_settings::Expected::Matching,
                ));
            }
        }
        if self
            .name
            .as_ref()
            .is_some_and(|name| name.chars().count() > MAX_CREDIT)
        {
            return Err(
                InvalidSetting::new("name", engine_settings::Expected::String).allowed(
                    engine_settings::Allowed::AtMost {
                        max: MAX_CREDIT as u64,
                    },
                ),
            );
        }
        Ok(())
    }
}

crate::codec::record!(GivenRiver {
    name,
    points,
    width,
    depth,
    discharge,
});
