use engine_core::{Expected, InvalidSetting, settings};
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

settings! {
    #[section("sun")]
    #[check(check_direction)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSun")]
    pub struct Sunlight {
        pub preset: LightPreset = LightPreset::GoldenHour => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sun_direction: Option<[f32; 3]> = None => finite,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sun_color: Option<[f32; 3]> = None => at_least(0.0),
    }
}

fn check_direction(sun: &Sunlight) -> Result<(), InvalidSetting> {
    match sun.sun_direction {
        Some(direction) if Vec3::from_array(direction).length_squared() == 0.0 => {
            Err(InvalidSetting::new("sun_direction", Expected::Nonzero).got(direction))
        }
        Some(_) | None => Ok(()),
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
