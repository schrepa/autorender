//! Clouds (decision 084): a layer of cumulus over the world, drifting with
//! the wind and lit by the sun and the sky. Renderers march it from the
//! camera into a dome of the sky above it, a slice of the dome's rows each
//! frame (all of them on exact frames), which the sky, its reflections and
//! the environment maps show; and from above the world toward the sun into
//! a map of the light it lets through, which shades the ground under it.

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_range};
use crate::math::clamp;

/// The cloud layer: whether it draws; how much of the sky it covers (0,
/// clear, to 1, overcast); the world height of its base and how thick it
/// is (meters, base -10000 to 30000, thickness 50 to 10000); how dense its
/// clouds are (0.01 to 4); the wind it drifts in (meters per second along x
/// and z, each -100 to 100); the meters across its largest shapes (200 to
/// 50000); the seed of its shapes; how much of the sun's light its shadows
/// take from the ground (0 to 1); the dome's texels around the horizon
/// (256 to 4096, a multiple of 64; a quarter as many from the horizon up);
/// and the frames over which ordinary frames march the whole dome (1 to
/// 64). Opt-in: a scene without a `clouds` section has none. Requires
/// version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneClouds")
)]
#[serde(default, deny_unknown_fields)]
pub struct Clouds {
    pub enabled: bool,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub coverage: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = -10000.0, max = 30000.0)))]
    pub base: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 50.0, max = 10000.0)))]
    pub thickness: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 4.0)))]
    pub density: f32,
    pub wind: [f32; 2],
    #[cfg_attr(feature = "schema", schemars(range(min = 200.0, max = 50000.0)))]
    pub scale: f32,
    pub seed: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub shadows: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 256, max = 4096)))]
    pub resolution: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 64)))]
    pub slices: u32,
}

impl Default for Clouds {
    fn default() -> Self {
        Self {
            enabled: true,
            coverage: 0.45,
            base: 1800.0,
            thickness: 1400.0,
            density: 1.0,
            wind: [8.0, 3.0],
            scale: 3000.0,
            seed: 1,
            shadows: 0.75,
            resolution: 2048,
            slices: 32,
        }
    }
}

impl Clouds {
    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("coverage", self.coverage, 0.0..=1.0)?;
        check_range("base", self.base, -10_000.0..=30_000.0)?;
        check_range("thickness", self.thickness, 50.0..=10_000.0)?;
        check_range("density", self.density, 0.01..=4.0)?;
        for value in self.wind {
            check_range("wind", value, -100.0..=100.0)?;
        }
        check_range("scale", self.scale, 200.0..=50_000.0)?;
        check_range("shadows", self.shadows, 0.0..=1.0)?;
        if !(256..=4096).contains(&self.resolution) || !self.resolution.is_multiple_of(64) {
            return Err(InvalidSetting::new(
                "resolution",
                "must be in [256, 4096] and a multiple of 64",
            ));
        }
        if !(1..=64).contains(&self.slices) {
            return Err(InvalidSetting::new("slices", "must be in [1, 64]"));
        }
        Ok(())
    }

    /// Every value in range: out-of-range values clamp, unusable ones fall
    /// back to the defaults.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let within = |value: f32, low: f32, high: f32, fallback: f32| {
            if value.is_finite() {
                clamp(value, low, high)
            } else {
                fallback
            }
        };
        Self {
            enabled: self.enabled,
            coverage: within(self.coverage, 0.0, 1.0, defaults.coverage),
            base: within(self.base, -10_000.0, 30_000.0, defaults.base),
            thickness: within(self.thickness, 50.0, 10_000.0, defaults.thickness),
            density: within(self.density, 0.01, 4.0, defaults.density),
            wind: [0, 1].map(|axis| within(self.wind[axis], -100.0, 100.0, defaults.wind[axis])),
            scale: within(self.scale, 200.0, 50_000.0, defaults.scale),
            seed: self.seed,
            shadows: within(self.shadows, 0.0, 1.0, defaults.shadows),
            resolution: self.resolution.clamp(256, 4096) / 64 * 64,
            slices: self.slices.clamp(1, 64),
        }
    }
}
