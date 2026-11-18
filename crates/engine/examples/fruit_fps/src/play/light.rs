use engine::prelude::{Component, Resource};
use engine_core::{Error, ErrorCode, Expected, Filter, Sampler, TextureRef};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MipFilter {
    Linear,
    Nearest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MapFilter {
    pub anisotropy: u16,
    pub mip: MipFilter,
}

impl MapFilter {
    pub const ENGINE: Self = Self {
        anisotropy: Sampler::ANISOTROPY,
        mip: MipFilter::Linear,
    };

    pub fn checked(&self, path: &str) -> Result<(), String> {
        if !(1..=16).contains(&self.anisotropy) {
            return Err(format!(
                "{path}.anisotropy: {} is not within 1..=16",
                self.anisotropy
            ));
        }
        if self.anisotropy > 1 && self.mip != MipFilter::Linear {
            return Err(format!(
                "{path}: anisotropy {} needs mip \"linear\" (WebGPU allows anisotropy only with linear filters)",
                self.anisotropy
            ));
        }
        Ok(())
    }

    pub fn applied(self, sampler: Sampler) -> Sampler {
        let mip = match self.mip {
            MipFilter::Linear => Filter::Linear,
            MipFilter::Nearest => Filter::Nearest,
        };
        let linear = sampler.mag_filter == Filter::Linear
            && sampler.min_filter == Filter::Linear
            && self.mip == MipFilter::Linear;
        Sampler {
            mipmap_filter: sampler.mipmap_filter.map(|_| mip),
            anisotropy: if linear { self.anisotropy } else { 1 },
            ..sampler
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Filtering {
    pub base_color: MapFilter,
    pub normal: MapFilter,
    pub orm: MapFilter,
    pub lightmap: MapFilter,
    #[serde(rename = "macro")]
    pub macro_map: MapFilter,
    pub sky_picture: MapFilter,
    pub light_field: MapFilter,
}

impl Default for Filtering {
    fn default() -> Self {
        Self {
            base_color: MapFilter::ENGINE,
            normal: MapFilter::ENGINE,
            orm: MapFilter::ENGINE,
            lightmap: MapFilter::ENGINE,
            macro_map: MapFilter::ENGINE,
            sky_picture: MapFilter::ENGINE,
            light_field: MapFilter::ENGINE,
        }
    }
}

impl Filtering {
    pub fn checked(&self) -> Result<(), String> {
        [
            (self.base_color, "base_color"),
            (self.normal, "normal"),
            (self.orm, "orm"),
            (self.lightmap, "lightmap"),
            (self.macro_map, "macro"),
            (self.sky_picture, "sky_picture"),
            (self.light_field, "light_field"),
        ]
        .into_iter()
        .try_for_each(|(filter, name)| filter.checked(&format!("$.lighting.filtering.{name}")))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DetailLimit {
    pub least_px: f32,
    pub overlap: f32,
    pub never_m: f32,
}

impl DetailLimit {
    pub fn checked(&self) -> Result<(), String> {
        if !(self.least_px.is_finite() && self.least_px >= 0.0) {
            return Err(format!(
                "$.lighting.detail.least_px: {} is not 0 or more",
                self.least_px
            ));
        }
        if !(0.0..1.0).contains(&self.overlap) {
            return Err(format!(
                "$.lighting.detail.overlap: {} is not within 0..1",
                self.overlap
            ));
        }
        if !(self.never_m.is_finite() && self.never_m > 0.0) {
            return Err(format!(
                "$.lighting.detail.never_m: {} is not above 0",
                self.never_m
            ));
        }
        Ok(())
    }
}

impl Default for DetailLimit {
    fn default() -> Self {
        Self {
            least_px: 0.0,
            overlap: 0.0,
            never_m: 1.0,
        }
    }
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
    pub filtering: Filtering,
    pub detail: DetailLimit,
    pub visible_sets: bool,
}
