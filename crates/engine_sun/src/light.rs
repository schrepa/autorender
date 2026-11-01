use bevy_ecs::prelude::Resource;
use engine_codec::Expected;
use engine_core::{Error, ErrorCode, Section};
use engine_places::lighting::LightPreset;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SunValues {
    pub(crate) sun_direction: Vec3,
    pub(crate) sun_color: Vec3,
}

fn sun_for(preset: LightPreset) -> SunValues {
    let (sun_direction, sun_color) = match preset {
        LightPreset::Day => (Vec3::new(0.55, 0.75, 0.35), Vec3::new(2.3, 2.1, 1.75)),
        LightPreset::GoldenHour => (Vec3::new(0.82, 0.45, 0.1), Vec3::new(2.6, 1.6, 0.85)),
        LightPreset::Dusk => (Vec3::new(0.9, 0.07, 0.4), Vec3::new(2.2, 0.85, 0.5)),
        LightPreset::Night => (Vec3::new(-0.35, 0.6, -0.45), Vec3::new(0.35, 0.45, 0.85)),
    };
    SunValues {
        sun_direction,
        sun_color,
    }
}

fn vec3_or(authored: Option<[f32; 3]>, preset: Vec3) -> Vec3 {
    authored.map_or(preset, Vec3::from_array)
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Resource,
    engine_core::__private::serde::Serialize,
    engine_core::__private::serde::Deserialize,
    engine_core::__private::schemars::JsonSchema,
)]
#[component(immutable)]
#[serde(crate = "engine_core::__private::serde", default, deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars", rename = "SceneSun")]
pub struct Sunlight {
    pub preset: LightPreset,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sun_direction: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sun_color: Option<[f32; 3]>,
}

impl Default for Sunlight {
    fn default() -> Self {
        Self {
            preset: LightPreset::GoldenHour,
            sun_direction: None,
            sun_color: None,
        }
    }
}

impl Section for Sunlight {
    const KEY: &'static str = "sun";

    fn check(&self) -> Result<(), Error> {
        match self.sun_direction {
            Some(direction) if !direction.iter().all(|value| value.is_finite()) => {
                return Err(Error::new(ErrorCode::InvalidValue, "sun.sun_direction")
                    .expected(Expected::Finite)
                    .got(direction));
            }
            Some(_) | None => {}
        }
        match self.sun_color {
            Some(color) if !color.iter().all(|value| value.is_finite() && *value >= 0.0) => {
                return Err(Error::new(ErrorCode::InvalidValue, "sun.sun_color")
                    .expected(Expected::Finite)
                    .at_least(0)
                    .got(color));
            }
            Some(_) | None => {}
        }
        match self.sun_direction {
            Some(direction) if Vec3::from_array(direction).length_squared() == 0.0 => {
                Err(Error::new(ErrorCode::InvalidValue, "sun.sun_direction")
                    .expected(Expected::Nonzero)
                    .got(direction))
            }
            Some(_) | None => Ok(()),
        }
    }
}

impl Sunlight {
    pub(crate) fn values(&self) -> SunValues {
        let preset = sun_for(self.preset);
        SunValues {
            sun_direction: vec3_or(self.sun_direction, preset.sun_direction),
            sun_color: vec3_or(self.sun_color, preset.sun_color),
        }
    }
}
