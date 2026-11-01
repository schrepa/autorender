use engine_core::{Error, ErrorCode, ExtraRules, InvalidSetting, Reach, Setting, fields};
use engine_places::lights::{LightRegion, LocalLight, MAX_SPOT_DEGREES, MIN_SPOT_DEGREES, Spot};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::lights::{light_rule, spot_order, spot_rule};

fields! {
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneLight {
        pub color: [f32; 3] = LocalLight::default().color.to_array()
            => same(light_rule("color")),
        pub intensity: f32 = LocalLight::default().intensity
            => same(light_rule("intensity")),
        pub range: f32 = LocalLight::default().range => same(light_rule("range")),
        pub offset: [f32; 3] = LocalLight::default().offset.to_array()
            => same(light_rule("offset")),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub spot: Option<SceneSpot> = None => any,
        #[serde(skip_serializing_if = "is_zero")]
        pub radius: f32 = LocalLight::default().radius => same(light_rule("radius")),
        #[serde(skip_serializing_if = "is_zero")]
        pub length: f32 = LocalLight::default().length => same(light_rule("length")),
        #[serde(skip_serializing_if = "is_both")]
        pub region: SceneLightRegion = SceneLightRegion::Both => any,
    }
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

fn is_both(region: &SceneLightRegion) -> bool {
    *region == SceneLightRegion::Both
}

impl Setting for SceneLightRegion {}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SceneLightRegion {
    #[default]
    Both,
    Interior,
    Exterior,
}

impl From<LightRegion> for SceneLightRegion {
    fn from(region: LightRegion) -> Self {
        match region {
            LightRegion::Both => Self::Both,
            LightRegion::Interior => Self::Interior,
            LightRegion::Exterior => Self::Exterior,
        }
    }
}

fields! {
    extra_rules;
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneSpot {
        pub direction: [f32; 3] = Spot::default().direction.to_array()
            => same(spot_rule("direction")),
        pub inner_angle_degrees: f32 = Spot::default().inner_angle.to_degrees()
            => range(0.0, MAX_SPOT_DEGREES),
        pub outer_angle_degrees: f32 = Spot::default().outer_angle.to_degrees()
            => range(MIN_SPOT_DEGREES, MAX_SPOT_DEGREES),
    }
}

impl ExtraRules for SceneSpot {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        spot_order(
            Vec3::from_array(self.direction),
            self.inner_angle_degrees.to_radians(),
            self.outer_angle_degrees.to_radians(),
        )
    }
}

impl From<&LocalLight> for SceneLight {
    fn from(light: &LocalLight) -> Self {
        Self {
            color: light.color.to_array(),
            intensity: light.intensity,
            range: light.range,
            offset: light.offset.to_array(),
            spot: light.spot.map(|spot| {
                let [inner, outer] = spot.authored_degrees.unwrap_or([
                    spot.inner_angle.to_degrees(),
                    spot.outer_angle.to_degrees(),
                ]);
                SceneSpot {
                    direction: spot.direction.to_array(),
                    inner_angle_degrees: inner,
                    outer_angle_degrees: outer,
                }
            }),
            radius: light.radius,
            length: light.length,
            region: light.region.into(),
        }
    }
}

pub(crate) fn check_light(light: &LocalLight, path: &str) -> Result<(), Error> {
    SceneLight::from(light)
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

pub(crate) fn light_reach(light: &LocalLight, reach: &Reach<'_>, extend: &mut dyn FnMut(f32)) {
    extend(reach.from + light.range);
}

pub(crate) fn light_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(SceneLight)).unwrap_or(Value::Null)
}

pub(crate) fn light_example() -> Value {
    json!({})
}
