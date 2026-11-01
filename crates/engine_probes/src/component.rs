use engine_core::{Error, ErrorCode, Reach, fields};
use engine_places::probes::ReflectionProbe;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fields! {
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "Interior")]
    pub struct SceneInterior {
        pub sky: f32 = 1.0 => range(0.0, 1.0),
        pub sun: f32 = 1.0 => range(0.0, 1.0),
    }
}

fields! {
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct SceneReflectionProbe {
        pub extent: [f32; 3] = [1.0; 3] => above(0.0),
        #[serde(default = "default_blend")]
        pub blend: f32 = default_blend() => at_least(0.0),
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub interior: Option<SceneInterior> = None => any,
    }
}

fn default_blend() -> f32 {
    1.0
}

impl From<&ReflectionProbe> for SceneReflectionProbe {
    fn from(probe: &ReflectionProbe) -> Self {
        Self {
            extent: probe.extent.to_array(),
            blend: probe.blend,
            interior: probe.interior.map(|interior| SceneInterior {
                sky: interior.sky,
                sun: interior.sun,
            }),
        }
    }
}

pub(crate) fn valid(probe: &ReflectionProbe) -> bool {
    SceneReflectionProbe::from(probe).validate().is_ok()
}

pub(crate) fn check(probe: &ReflectionProbe, path: &str) -> Result<(), Error> {
    SceneReflectionProbe::from(probe)
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

pub(crate) fn reach(probe: &ReflectionProbe, reach: &Reach<'_>, extend: &mut dyn FnMut(f32)) {
    extend(reach.from + probe.extent.length());
}

pub(crate) fn schema() -> Value {
    serde_json::to_value(schemars::schema_for!(SceneReflectionProbe))
        .unwrap_or(Value::Null)
}

pub(crate) fn example() -> Value {
    json!({ "extent": [1.0, 1.0, 1.0] })
}
