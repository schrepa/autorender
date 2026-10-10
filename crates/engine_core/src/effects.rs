//! Rendering effects of the scene: sun shadows, ambient
//! occlusion and bloom, sun shafts,
//! screen-space reflections and refraction through glass.
//! Every extracted frame carries sanitized copies with the
//! atmosphere; renderers skip the work of disabled effects.
//!
//! Each effect is its own resource: disable one with its settings, or remove
//! the resource to remove the effect. The environment section list
//! (`environment_sections!`) says which are inserted by default and which
//! are opt-in, inserted to turn them on. Without an `Atmosphere`
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

use crate::environment::InvalidSetting;
use crate::{Camera, Environment, FrameTime, RenderSnapshot};
use engine_settings::{ExtraRules, settings};

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

settings! {
    extra_rules;
    /// Sun shadows; omitted fields take the engine defaults.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneShadows")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Shadows {
        /// Whether the sun casts shadows.
        pub enabled: bool = true => any,
        /// Region to keep shadows for: the whole scene gives the sharpest,
        /// stable shadows. Without it the maps follow the camera. Corners
        /// finite, `min` at most `max` on every axis.
        #[serde(with = "bounds", skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "schema", schemars(with = "Option<bounds::Bounds>"))]
        pub bounds: Option<(Vec3, Vec3)> = None => any,
        /// Shadows fade out this far from the camera (world units).
        pub distance: f32 = 500.0 => above(0.0),
        /// View distance of the sharper maps that follow the camera, less
        /// than `distance`.
        pub near_distance: f32 = 100.0 => above(0.0),
        /// Sharper maps (cascades) splitting `near_distance`, each reaching a
        /// third as far as the next; 3 when omitted. Version 6.
        #[serde(
            deserialize_with = "cascades_or_default",
            skip_serializing_if = "is_default_cascades"
        )]
        #[cfg_attr(feature = "schema", schemars(with = "Option<u32>"))]
        pub cascades: u32 = 3 => range(1, MAX_SHADOW_CASCADES),
        /// Texels per side of the far maps.
        pub resolution: u32 = 2048 => power_of_two(1024, 4096),
        /// Kept levels: maps of still casters following the camera, sharper
        /// than the far maps where they reach, each with half the far maps'
        /// `resolution` per side; none when omitted. Worlds larger than their
        /// shadows' sharpness give them two or three.
        #[serde(skip_serializing_if = "is_zero")]
        pub levels: u32 = 0 => range(0, MAX_KEPT_LEVELS),
        /// Side of the finest kept level (world units), which each next level
        /// is four times; 256 when omitted.
        #[serde(skip_serializing_if = "is_default_level_extent")]
        pub level_extent: f32 = 256.0 => above(0.0),
    }
}

impl ExtraRules for Shadows {
    /// The near maps reach less far than the shadows, and bounds are a box.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.near_distance >= self.distance {
            return Err(InvalidSetting::new(
                "near_distance",
                "must be less than distance",
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

    /// Unusable bounds are dropped; a near distance beyond the shadows'
    /// reaches half of it at most.
    fn sanitize_extra(self) -> Self {
        let bounds = self
            .bounds
            .filter(|(min, max)| min.is_finite() && max.is_finite() && min.cmple(*max).all());
        let near_distance = if self.near_distance < self.distance {
            self.near_distance
        } else {
            crate::math::min(Self::default().near_distance, self.distance * 0.5)
        };
        Self {
            bounds,
            near_distance,
            ..self
        }
    }
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

impl Shadows {
    /// What a world without these settings reads as: no sun shadows.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
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

settings! {
    /// Ambient occlusion; an intensity of 0 turns it off.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneAmbientOcclusion")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct AmbientOcclusion {
        /// How much creases and corners darken the ambient light; 0 turns it
        /// off.
        pub intensity: f32 = 0.8 => range(0.0, 1.0),
        /// World distance within which geometry occludes.
        pub radius: f32 = 2.5 => above(0.0),
    }
}

impl AmbientOcclusion {
    /// What a world without these settings reads as: no ambient occlusion.
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

settings! {
    /// Bloom; an intensity of 0 turns it off.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneBloom")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Bloom {
        /// How strongly light glows beyond bright pixels; 0 turns it off.
        pub intensity: f32 = 0.6 => range(0.0, 4.0),
        /// Exposed brightness above which light glows.
        pub threshold: f32 = 2.0 => at_least(0.0),
    }
}

impl Bloom {
    /// What a world without these settings reads as: no bloom.
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

settings! {
    /// Sun shafts: sunlight scattered by the fog between shadow
    /// casters. Requires version 6 and shadows; an intensity of 0 turns them
    /// off.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSunShafts")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct SunShafts {
        /// How bright the shafts are; 0 turns them off.
        pub intensity: f32 = 1.0 => range(0.0, 4.0),
        /// Forward scattering: higher concentrates light toward the sun.
        pub anisotropy: f32 = 0.6 => range(0.0, 0.95),
        /// World distance along each view ray light is gathered over.
        pub distance: f32 = 300.0 => range(0.001, 100_000.0),
    }
}

impl SunShafts {
    /// What a world without these settings reads as: no sun shafts.
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

settings! {
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
        /// How strongly glossy surfaces reflect the screen; 0 turns them off.
        pub intensity: f32 = 1.0 => range(0.0, 1.0),
        /// Surfaces rougher than this reflect only the sky.
        pub max_roughness: f32 = 0.5 => range(0.05, 1.0),
        /// World distance reflection rays travel.
        pub distance: f32 = 120.0 => range(0.001, 100_000.0),
    }
}

impl Reflections {
    /// What a world without these settings reads as: no screen-space reflections.
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

settings! {
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
        /// Scales how far rays bend through transmissive materials; 0 turns
        /// it off.
        pub strength: f32 = 1.0 => range(0.0, 4.0),
        /// Blur radius in pixels that fully rough glass reaches.
        pub blur: f32 = 12.0 => range(0.0, 64.0),
    }
}

impl Refraction {
    /// What a world without these settings reads as: no refraction.
    pub fn off() -> Self {
        Self {
            strength: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.strength > 0.0
    }
}
