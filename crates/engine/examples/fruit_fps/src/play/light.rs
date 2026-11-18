use engine::prelude::{Component, Resource};
use engine_core::{Error, ErrorCode, Expected, TextureRef};
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Light {
    pub direction: [f32; 3],
    pub sun: [f32; 3],
    pub sky: [f32; 3],
    pub ground: [f32; 3],
}

impl Light {
    pub fn checked(&self, path: &str) -> Result<(), Error> {
        if let Some(at) = self.direction.iter().position(|value| !value.is_finite()) {
            return Err(
                Error::new(ErrorCode::InvalidValue, format!("{path}.direction[{at}]"))
                    .expected(Expected::Finite)
                    .got(self.direction[at]),
            );
        }
        if self.direction == [0.0; 3] {
            return Err(
                Error::new(ErrorCode::InvalidValue, format!("{path}.direction"))
                    .expected(Expected::Nonzero)
                    .got(self.direction),
            );
        }
        let colors = [
            ("sun", self.sun),
            ("sky", self.sky),
            ("ground", self.ground),
        ];
        for (name, color) in colors {
            if let Some(at) = color
                .iter()
                .position(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(
                    Error::new(ErrorCode::OutOfRange, format!("{path}.{name}[{at}]"))
                        .expected(Expected::Finite)
                        .got(color[at]),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    pub start_m: f32,
    pub density_per_m: f32,
    pub most: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyLook {
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    pub below: [f32; 3],
    pub haze_height: f32,
    pub disc_cos_inner: f32,
    pub disc_cos_outer: f32,
    pub disc_strength: f32,
    pub glow_power: f32,
    pub glow_strength: f32,
    pub mirror_roughness: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
}

impl Default for Grade {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            contrast: 1.0,
            saturation: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ambient {
    pub sky_scale: f32,
    pub ground_scale: f32,
    pub base_occlusion: f32,
    pub base_height_m: f32,
    pub floor_m: f32,
    pub black_floor: f32,
}

impl Default for Ambient {
    fn default() -> Self {
        Self {
            sky_scale: 1.0,
            ground_scale: 1.0,
            base_occlusion: 1.0,
            base_height_m: 1.0,
            floor_m: 0.0,
            black_floor: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightField {
    pub map: TextureRef,
    pub rect: [f32; 4],
    pub height_m: f32,
    pub scale: f32,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Lighting {
    pub lit: bool,
    pub to_sun: Vec3,
    pub sun: [f32; 3],
    pub sky: [f32; 3],
    pub ground: [f32; 3],
    pub wrap: f32,
    pub sky_look: Option<SkyLook>,
    pub fog: Option<Fog>,
    pub grade: Grade,
    pub ambient: Ambient,
    pub antialias: bool,
    pub field: Option<LightField>,
}
