//! Rendering effects of the scene: sun shadows, ambient
//! occlusion and bloom, sun shafts,
//! screen-space reflections and refraction through glass.
//! Every extracted frame carries sanitized copies with the
//! atmosphere; renderers skip the work of disabled effects.
//!
//! Each effect is its own resource: disable one with its settings, or remove
//! the resource to remove the effect. Shadows, ambient occlusion and bloom
//! are inserted by default; sun shafts, reflections and refraction are
//! opt-in: insert them (their defaults are on). Without an `Atmosphere`
//! frames use the default one. Each is also a scene environment section in
//! the same JSON shape and checks itself with `validate`;
//! their docs are the sections' schema descriptions, written for authors.
//!
//! Shadows: every rendered instance casts and receives them. One map covers
//! `bounds`, or without bounds a sphere of `distance` around the camera that
//! moves when the camera travels a quarter of it. Instances that stay still
//! are drawn into that map once and kept; moving ones are drawn every frame.
//! Sharper maps follow the first `near_distance` of the view, split into
//! `cascades`: each has three times the texel density of the
//! next and is a depth pass per frame over the casters it covers, and each
//! has half the far maps' `resolution` per side. Kept `levels`
//! hold still casters sharper than that map near the camera: squares
//! `level_extent` wide and four times wider each, following the camera a
//! tile at a time and redrawing only the tiles it uncovers.
//!
//! Ambient occlusion darkens the sky's ambient light in creases, corners and
//! under objects. Bloom makes light glow beyond bright pixels such as
//! emissive surfaces, sun glints and the sun itself: emission and highlights
//! usually exceed its default threshold, which applies to brightness before
//! tone mapping; sunlit walls do not. Sun shafts are sunlight the fog
//! scatters toward the eye, with a glow around the sun; they read the fog's
//! density and height falloff as the medium. Screen-space reflections fall
//! back to the sky where their rays leave the screen. Refraction bends the
//! opaque scene seen through transmissive materials along their index of
//! refraction and thickness and blurs it by their roughness; frames without
//! glass never pay for it.

use bevy_ecs::prelude::{Resource, World};
use glam::Vec3;
use serde::{Deserialize, Deserializer, Serialize};

use crate::environment::{InvalidSetting, check_range};
use crate::{Camera, Environment, FrameTime, RenderSnapshot};

/// Copies the camera eye, presentation time and environment of `world`
/// into `snapshot`, sanitized ([`Environment::current`]): a missing
/// atmosphere reads as the default one, a missing effect resource turns that
/// effect off.
pub(crate) fn extract_lighting(world: &World, snapshot: &mut RenderSnapshot) {
    snapshot.camera_position = world.resource::<Camera>().eye();
    snapshot.elapsed_seconds = world
        .get_resource::<FrameTime>()
        .map_or(0.0, |time| time.elapsed_seconds as f32);
    snapshot.presented_seconds = world
        .get_resource::<crate::PresentedTime>()
        .map_or(0.0, |time| time.seconds);
    snapshot.environment = Environment::current(world).sanitized();
}

/// Most near shadow cascades ([`Shadows::cascades`]).
pub const MAX_SHADOW_CASCADES: u32 = 4;
/// Most kept shadow levels ([`Shadows::levels`]).
pub const MAX_KEPT_LEVELS: u32 = 3;

/// Sun shadows; omitted fields take the engine defaults.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneShadows")
)]
#[serde(default, deny_unknown_fields)]
pub struct Shadows {
    pub enabled: bool,
    /// Region to keep shadows for: the whole scene gives the sharpest, stable
    /// shadows. Without it the maps follow the camera.
    #[serde(with = "bounds", skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(with = "Option<bounds::Bounds>"))]
    pub bounds: Option<(Vec3, Vec3)>,
    /// Shadows fade out this far from the camera (world units).
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub distance: f32,
    /// View distance of the sharper maps that follow the camera.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub near_distance: f32,
    /// Sharper maps (cascades) splitting `near_distance`, each reaching a
    /// third as far as the next: 1 to 4, 3 when omitted. Version 6.
    #[serde(
        deserialize_with = "cascades_or_default",
        skip_serializing_if = "is_default_cascades"
    )]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Option<u32>", range(min = 1, max = 4))
    )]
    pub cascades: u32,
    /// Texels per side of the far maps: 1024, 2048 or 4096.
    #[cfg_attr(feature = "schema", schemars(range(min = 1024, max = 4096)))]
    pub resolution: u32,
    /// Kept levels: maps of still casters following the
    /// camera, sharper than the far maps where they reach, each with half
    /// the far maps' `resolution` per side; 0 to 3, none when omitted.
    /// Worlds larger than their shadows' sharpness give them two or three.
    #[serde(skip_serializing_if = "is_zero")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0, max = 3)))]
    pub levels: u32,
    /// Side of the finest kept level (world units), which each next level
    /// is four times; 256 when omitted.
    #[serde(skip_serializing_if = "is_default_level_extent")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub level_extent: f32,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

fn is_default_level_extent(extent: &f32) -> bool {
    *extent == Shadows::default().level_extent
}

/// Shadow bounds as a scene writes them: `{"min": [x, y, z], "max": [x, y, z]}`.
mod bounds {
    use glam::Vec3;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// A world box (minimum and maximum corners).
    #[derive(Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneBounds")
    )]
    #[serde(deny_unknown_fields)]
    pub(super) struct Bounds {
        min: [f32; 3],
        max: [f32; 3],
    }

    pub(super) fn serialize<S: Serializer>(
        bounds: &Option<(Vec3, Vec3)>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        bounds
            .map(|(min, max)| Bounds {
                min: min.to_array(),
                max: max.to_array(),
            })
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<(Vec3, Vec3)>, D::Error> {
        Ok(Option::<Bounds>::deserialize(deserializer)?
            .map(|bounds| (Vec3::from_array(bounds.min), Vec3::from_array(bounds.max))))
    }
}

/// Reads a null cascade count as the default, as an omitted one.
fn cascades_or_default<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    Ok(Option::<u32>::deserialize(deserializer)?.unwrap_or(Shadows::default().cascades))
}

/// Scenes omit the default cascade count, which documents before version 6
/// cannot change.
fn is_default_cascades(cascades: &u32) -> bool {
    *cascades == Shadows::default().cascades
}

impl Default for Shadows {
    fn default() -> Self {
        Self {
            enabled: true,
            bounds: None,
            distance: 500.0,
            near_distance: 100.0,
            cascades: 3,
            resolution: 2048,
            levels: 0,
            level_extent: 256.0,
        }
    }
}

impl Shadows {
    /// Usable settings: unusable bounds are dropped, a resolution rounds up
    /// to the next accepted power of two, cascades clamp to their range, and
    /// unusable distances (or a near distance beyond the shadow distance)
    /// fall back to the defaults.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let bounds = self
            .bounds
            .filter(|(min, max)| min.is_finite() && max.is_finite() && min.cmple(*max).all());
        let distance = if self.distance.is_finite() && self.distance > 0.0 {
            self.distance
        } else {
            defaults.distance
        };
        let near_distance = if self.near_distance.is_finite()
            && self.near_distance > 0.0
            && self.near_distance < distance
        {
            self.near_distance
        } else {
            crate::math::min(defaults.near_distance, distance * 0.5)
        };
        Self {
            enabled: self.enabled,
            bounds,
            distance,
            near_distance,
            cascades: self.cascades.clamp(1, MAX_SHADOW_CASCADES),
            resolution: self.resolution.clamp(1024, 4096).next_power_of_two(),
            levels: self.levels.min(MAX_KEPT_LEVELS),
            level_extent: if self.level_extent.is_finite() && self.level_extent > 0.0 {
                self.level_extent
            } else {
                defaults.level_extent
            },
        }
    }

    /// Ok when a scene may hold these settings: positive distances with the
    /// near one inside the other, a resolution of 1024, 2048 or 4096, 1 to
    /// [`MAX_SHADOW_CASCADES`] cascades, at most [`MAX_KEPT_LEVELS`] kept
    /// levels of a finite positive extent, and finite bounds with `min` at
    /// most `max` on every axis.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        if !self.distance.is_finite() || self.distance <= 0.0 {
            return Err(InvalidSetting::new(
                "distance",
                "must be finite and positive",
            ));
        }
        if !self.near_distance.is_finite()
            || self.near_distance <= 0.0
            || self.near_distance >= self.distance
        {
            return Err(InvalidSetting::new(
                "near_distance",
                "must be finite, positive and less than distance",
            ));
        }
        if ![1024, 2048, 4096].contains(&self.resolution) {
            return Err(InvalidSetting::new(
                "resolution",
                "must be 1024, 2048 or 4096",
            ));
        }
        if !(1..=MAX_SHADOW_CASCADES).contains(&self.cascades) {
            return Err(InvalidSetting::new(
                "cascades",
                format!("must be 1 to {MAX_SHADOW_CASCADES}"),
            ));
        }
        if self.levels > MAX_KEPT_LEVELS {
            return Err(InvalidSetting::new(
                "levels",
                format!("must be 0 to {MAX_KEPT_LEVELS}"),
            ));
        }
        if !self.level_extent.is_finite() || self.level_extent <= 0.0 {
            return Err(InvalidSetting::new(
                "level_extent",
                "must be finite and positive",
            ));
        }
        if let Some((min, max)) = self.bounds
            && (!min.is_finite() || !max.is_finite() || !min.cmple(max).all())
        {
            return Err(InvalidSetting::new(
                "bounds",
                "corners must be finite with min at most max on every axis",
            ));
        }
        Ok(())
    }

    /// View distance each near cascade reaches, nearest first: the last
    /// reaches `near_distance`, and each one before it a third as far.
    pub fn cascade_reaches(&self) -> impl Iterator<Item = f32> {
        let settings = self.sanitized();
        (0..settings.cascades).map(move |cascade| {
            let steps = (settings.cascades - 1 - cascade) as i32;
            settings.near_distance / 3.0_f32.powi(steps)
        })
    }
}

/// Ambient occlusion; an intensity of 0 turns it off.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneAmbientOcclusion")
)]
#[serde(default, deny_unknown_fields)]
pub struct AmbientOcclusion {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub intensity: f32,
    /// World distance within which geometry occludes.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub radius: f32,
}

impl Default for AmbientOcclusion {
    fn default() -> Self {
        Self {
            intensity: 0.8,
            radius: 2.5,
        }
    }
}

impl AmbientOcclusion {
    /// Ok for an intensity in `[0, 1]` and a finite, positive radius.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("intensity", self.intensity, 0.0..=1.0)?;
        if !self.radius.is_finite() || self.radius <= 0.0 {
            return Err(InvalidSetting::new("radius", "must be finite and positive"));
        }
        Ok(())
    }

    /// Intensity clamped into `[0, 1]` (NaN disables it); an unusable radius
    /// falls back to the default.
    pub fn sanitized(self) -> Self {
        Self {
            intensity: if self.intensity.is_nan() {
                0.0
            } else {
                crate::math::clamp(self.intensity, 0.0, 1.0)
            },
            radius: if self.radius.is_finite() && self.radius > 0.0 {
                self.radius
            } else {
                Self::default().radius
            },
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

/// Bloom; an intensity of 0 turns it off.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneBloom")
)]
#[serde(default, deny_unknown_fields)]
pub struct Bloom {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4.0)))]
    pub intensity: f32,
    /// Exposed brightness above which light glows.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub threshold: f32,
}

impl Default for Bloom {
    fn default() -> Self {
        Self {
            intensity: 0.6,
            threshold: 2.0,
        }
    }
}

impl Bloom {
    /// Ok for an intensity in `[0, 4]` and a finite, nonnegative threshold.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("intensity", self.intensity, 0.0..=4.0)?;
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err(InvalidSetting::new(
                "threshold",
                "must be finite and nonnegative",
            ));
        }
        Ok(())
    }

    /// Intensity clamped into `[0, 4]` (NaN disables it); an unusable
    /// threshold falls back to the default.
    pub fn sanitized(self) -> Self {
        Self {
            intensity: if self.intensity.is_nan() {
                0.0
            } else {
                crate::math::clamp(self.intensity, 0.0, 4.0)
            },
            threshold: if self.threshold.is_finite() && self.threshold >= 0.0 {
                self.threshold
            } else {
                Self::default().threshold
            },
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

/// Sun shafts: sunlight scattered by the fog between shadow
/// casters. Requires version 6 and shadows; an intensity of 0 turns them off.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneSunShafts")
)]
#[serde(default, deny_unknown_fields)]
pub struct SunShafts {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4.0)))]
    pub intensity: f32,
    /// Forward scattering: higher concentrates light toward the sun.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 0.95)))]
    pub anisotropy: f32,
    /// World distance along each view ray light is gathered over.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001, max = 100000.0)))]
    pub distance: f32,
}

/// Ok for a world distance that is positive and at most 100 000.
fn check_distance(distance: f32) -> Result<(), InvalidSetting> {
    if distance.is_finite() && distance > 0.0 && distance <= 100_000.0 {
        Ok(())
    } else {
        Err(InvalidSetting::new(
            "distance",
            "must be positive and at most 100000",
        ))
    }
}

impl Default for SunShafts {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            anisotropy: 0.6,
            distance: 300.0,
        }
    }
}

impl SunShafts {
    /// Ok for an intensity in `[0, 4]`, an anisotropy in `[0, 0.95]` and a
    /// distance that is positive and at most 100 000.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("intensity", self.intensity, 0.0..=4.0)?;
        check_range("anisotropy", self.anisotropy, 0.0..=0.95)?;
        check_distance(self.distance)
    }

    /// Intensity and anisotropy clamped into their ranges (NaN disables);
    /// an unusable distance falls back to the default.
    pub fn sanitized(self) -> Self {
        Self {
            intensity: if self.intensity.is_nan() {
                0.0
            } else {
                crate::math::clamp(self.intensity, 0.0, 4.0)
            },
            anisotropy: if self.anisotropy.is_nan() {
                Self::default().anisotropy
            } else {
                crate::math::clamp(self.anisotropy, 0.0, 0.95)
            },
            distance: if self.distance.is_finite() && self.distance > 0.0 {
                self.distance
            } else {
                Self::default().distance
            },
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

/// Screen-space reflections. Requires version 6; an intensity
/// of 0 turns them off.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneReflections")
)]
#[serde(default, deny_unknown_fields)]
pub struct Reflections {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub intensity: f32,
    /// Surfaces rougher than this reflect only the sky.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.05, max = 1.0)))]
    pub max_roughness: f32,
    /// World distance reflection rays travel.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001, max = 100000.0)))]
    pub distance: f32,
}

impl Default for Reflections {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            max_roughness: 0.5,
            distance: 120.0,
        }
    }
}

impl Reflections {
    /// Ok for an intensity in `[0, 1]`, a maximum roughness in `[0.05, 1]`
    /// and a distance that is positive and at most 100 000.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("intensity", self.intensity, 0.0..=1.0)?;
        check_range("max_roughness", self.max_roughness, 0.05..=1.0)?;
        check_distance(self.distance)
    }

    /// Values clamped into their ranges (NaN intensity disables); an
    /// unusable distance falls back to the default.
    pub fn sanitized(self) -> Self {
        Self {
            intensity: if self.intensity.is_nan() {
                0.0
            } else {
                crate::math::clamp(self.intensity, 0.0, 1.0)
            },
            max_roughness: if self.max_roughness.is_nan() {
                Self::default().max_roughness
            } else {
                crate::math::clamp(self.max_roughness, 0.05, 1.0)
            },
            distance: if self.distance.is_finite() && self.distance > 0.0 {
                self.distance
            } else {
                Self::default().distance
            },
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

/// Refraction through glass. Requires version 8; a strength
/// of 0 turns it off.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneRefraction")
)]
#[serde(default, deny_unknown_fields)]
pub struct Refraction {
    /// Scales how far rays bend through transmissive materials.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4.0)))]
    pub strength: f32,
    /// Blur radius in pixels that fully rough glass reaches.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 64.0)))]
    pub blur: f32,
}

impl Default for Refraction {
    fn default() -> Self {
        Self {
            strength: 1.0,
            blur: 12.0,
        }
    }
}

impl Refraction {
    /// Ok for a strength in `[0, 4]` and a blur in `[0, 64]`.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("strength", self.strength, 0.0..=4.0)?;
        check_range("blur", self.blur, 0.0..=64.0)
    }

    /// Values clamped into their ranges; NaN strength disables and a NaN
    /// blur reads as the default.
    pub fn sanitized(self) -> Self {
        Self {
            strength: if self.strength.is_nan() {
                0.0
            } else {
                crate::math::clamp(self.strength, 0.0, 4.0)
            },
            blur: if self.blur.is_nan() {
                Self::default().blur
            } else {
                crate::math::clamp(self.blur, 0.0, 64.0)
            },
        }
    }

    pub fn enabled(&self) -> bool {
        self.strength > 0.0
    }
}
