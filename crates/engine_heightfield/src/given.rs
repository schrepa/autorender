use engine_codec::{Allowed, Error, ErrorCode, Expected};
use serde::{Deserialize, Serialize};

use crate::Layers;

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

impl GivenRiver {
    pub fn validate(&self) -> Result<(), Error> {
        let reach = Layers::MAX_COORDINATE;
        for (index, point) in self.points.iter().enumerate() {
            for (axis, &value) in point.iter().enumerate() {
                if !(-reach..=reach).contains(&value) {
                    return Err(broken(format!("$.points[{index}][{axis}]"))
                        .range(f64::from(-reach), f64::from(reach))
                        .got(f64::from(value)));
                }
            }
        }
        for (field, values, most) in [
            ("width", &self.width, MAX_WIDTH),
            ("depth", &self.depth, MAX_DEPTH),
            ("discharge", &self.discharge, MAX_DISCHARGE),
        ] {
            if let Some((index, &value)) = values
                .iter()
                .enumerate()
                .find(|&(_, &value)| !(value > 0.0 && value <= most))
            {
                return Err(broken(format!("$.{field}[{index}]"))
                    .expected(Expected::Above)
                    .range(0.0, f64::from(most))
                    .got(f64::from(value)));
            }
        }
        if !(2..=MAX_RIVER_POINTS).contains(&self.points.len()) {
            return Err(broken("$.points")
                .expected(Expected::Array)
                .allowed(Allowed::Range {
                    min: 2.0,
                    max: MAX_RIVER_POINTS as f64,
                }));
        }
        for (field, values) in [
            ("width", &self.width),
            ("depth", &self.depth),
            ("discharge", &self.discharge),
        ] {
            if !(values.len() <= 1 || values.len() == self.points.len()) {
                return Err(broken(format!("$.{field}")).expected(Expected::Matching));
            }
        }
        if self
            .name
            .as_ref()
            .is_some_and(|name| name.chars().count() > MAX_CREDIT)
        {
            return Err(broken("$.name")
                .expected(Expected::String)
                .allowed(Allowed::AtMost {
                    max: MAX_CREDIT as u64,
                }));
        }
        Ok(())
    }
}

fn broken(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::OutOfRange, path)
}

engine_codec::record!(GivenRiver {
    name,
    points,
    width,
    depth,
    discharge,
});
