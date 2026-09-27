//! Rendering effects of the scene: sun shadows (decision 020), ambient
//! occlusion and bloom (decision 021), sun shafts (decision 024),
//! screen-space reflections (decision 025) and refraction through glass
//! (decision 038). Every extracted frame carries sanitized copies with the
//! atmosphere; renderers skip the work of disabled effects.
//!
//! Each effect is its own resource: disable one with its settings, or remove
//! the resource to remove the effect. Shadows, ambient occlusion and bloom
//! are inserted by default; sun shafts, reflections and refraction are
//! opt-in: insert them (their defaults are on). Without an `Atmosphere`
//! frames use the default one.

use bevy_ecs::prelude::{Resource, World};
use glam::Vec3;

use crate::{Atmosphere, Camera, FrameTime, RenderSnapshot};

/// Copies the camera eye, presentation time, atmosphere and effects of
/// `world` into `snapshot`, sanitized. A missing atmosphere reads as the
/// default one; a missing effect resource turns that effect off.
pub(crate) fn extract_lighting(world: &World, snapshot: &mut RenderSnapshot) {
    snapshot.camera_position = world.resource::<Camera>().eye();
    snapshot.elapsed_seconds = world
        .get_resource::<FrameTime>()
        .map_or(0.0, |time| time.elapsed_seconds as f32);
    snapshot.atmosphere = world
        .get_resource::<Atmosphere>()
        .map_or_else(Atmosphere::default, |value| value.sanitized());
    snapshot.shadows = world.get_resource::<Shadows>().map_or(
        Shadows {
            enabled: false,
            ..Shadows::default()
        },
        |value| value.sanitized(),
    );
    snapshot.ambient_occlusion = world.get_resource::<AmbientOcclusion>().map_or(
        AmbientOcclusion {
            intensity: 0.0,
            ..AmbientOcclusion::default()
        },
        |value| value.sanitized(),
    );
    snapshot.bloom = world.get_resource::<Bloom>().map_or(
        Bloom {
            intensity: 0.0,
            ..Bloom::default()
        },
        |value| value.sanitized(),
    );
    snapshot.sun_shafts = world.get_resource::<SunShafts>().map_or(
        SunShafts {
            intensity: 0.0,
            ..SunShafts::default()
        },
        |value| value.sanitized(),
    );
    snapshot.reflections = world.get_resource::<Reflections>().map_or(
        Reflections {
            intensity: 0.0,
            ..Reflections::default()
        },
        |value| value.sanitized(),
    );
    snapshot.refraction = world.get_resource::<Refraction>().map_or(
        Refraction {
            strength: 0.0,
            ..Refraction::default()
        },
        |value| value.sanitized(),
    );
}

/// Most near shadow cascades ([`Shadows::cascades`]).
pub const MAX_SHADOW_CASCADES: u32 = 4;

/// Shadows cast by the atmosphere's sun (decision 020).
///
/// Every rendered instance casts and receives them. One map covers `bounds`,
/// or without bounds a sphere of `distance` around the camera that moves
/// when the camera travels a quarter of it. Instances that stay still are
/// drawn into that map once and kept; moving ones are drawn every frame.
/// Sharper maps follow the first `near_distance` of the view, split into
/// `cascades` (decision 028).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Shadows {
    pub enabled: bool,
    /// World region (minimum and maximum corners) whose shadows are kept.
    /// A region that holds the whole scene fits the map to it and never
    /// moves it; `None` follows the camera instead.
    pub bounds: Option<(Vec3, Vec3)>,
    /// Shadows fade out this far from the camera.
    pub distance: f32,
    /// View distance covered by the sharper maps that follow the camera.
    pub near_distance: f32,
    /// Near maps (cascades) splitting `near_distance`, 1 to
    /// [`MAX_SHADOW_CASCADES`]: each reaches a third as far as the next and
    /// so has three times its texel density. Each is a depth pass per frame
    /// over the casters it covers.
    pub cascades: u32,
    /// Texels per side of the far maps: a power of two in `1024..=4096`.
    /// Each near map has half as many.
    pub resolution: u32,
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

/// Screen-space ambient occlusion (decision 021): creases, corners and the
/// ground under objects receive less of the sky's ambient light.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct AmbientOcclusion {
    /// How much occluded ambient light darkens, in `[0, 1]`; 0 disables it.
    pub intensity: f32,
    /// World distance within which geometry occludes.
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

/// Bloom (decision 021): light glows beyond bright pixels, such as emissive
/// surfaces, sun glints and the sun itself.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Bloom {
    /// Strength of the glow in `[0, 4]`; 0 disables it.
    pub intensity: f32,
    /// Exposed brightness (before tone mapping) above which light glows.
    /// Emission and highlights usually exceed the default; sunlit walls do not.
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

/// Sunlight scattered toward the eye by the atmosphere's fog (decision 024):
/// shafts of light between shadow casters and a glow around the sun. Opt-in:
/// insert this resource (its default is on). It needs sun shadows, and reads
/// the fog's density and height falloff as the medium.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SunShafts {
    /// Strength in `[0, 4]`; 0 turns the shafts off.
    pub intensity: f32,
    /// Forward scattering in `[0, 0.95]`: higher concentrates light toward
    /// the sun.
    pub anisotropy: f32,
    /// World distance along each view ray light is gathered over.
    pub distance: f32,
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

/// Screen-space reflections (decision 025): glossy surfaces reflect the
/// scene on screen, falling back to the sky where their rays leave it.
/// Opt-in: insert this resource (its default is on).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Reflections {
    /// Strength in `[0, 1]`; 0 turns reflections off.
    pub intensity: f32,
    /// Surfaces rougher than this, in `[0.05, 1]`, reflect only the sky.
    pub max_roughness: f32,
    /// World distance reflection rays travel.
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

/// Refraction through glass (decision 038): transmissive materials show the
/// opaque scene behind them, along rays their index of refraction bends
/// through their thickness, blurred by their roughness. Opt-in: insert this
/// resource (its default is on); without it transmissive materials blend as
/// plain transparency, and frames without glass never pay for it.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Refraction {
    /// Scales how far rays bend, in `[0, 4]`; 0 turns refraction off.
    pub strength: f32,
    /// The blur radius in pixels, in `[0, 64]`, that fully rough glass
    /// reaches; smoother glass blurs less.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refraction_clamps_into_its_ranges() {
        let wild = Refraction {
            strength: 9.0,
            blur: -3.0,
        }
        .sanitized();
        assert_eq!((wild.strength, wild.blur), (4.0, 0.0));
        let broken = Refraction {
            strength: f32::NAN,
            blur: f32::NAN,
        }
        .sanitized();
        assert!(!broken.enabled());
        assert_eq!(broken.blur, Refraction::default().blur);
        assert!(Refraction::default().enabled());
    }

    #[test]
    fn defaults_are_usable_and_unusable_settings_fall_back() {
        assert_eq!(Shadows::default().sanitized(), Shadows::default());
        assert_eq!(
            AmbientOcclusion::default().sanitized(),
            AmbientOcclusion::default()
        );
        assert_eq!(Bloom::default().sanitized(), Bloom::default());
        let shadows = Shadows {
            bounds: Some((Vec3::ONE, Vec3::ZERO)),
            distance: f32::NAN,
            near_distance: 5000.0,
            resolution: 1500,
            ..Shadows::default()
        }
        .sanitized();
        assert_eq!(shadows.bounds, None);
        assert_eq!(shadows.distance, Shadows::default().distance);
        assert_eq!(shadows.near_distance, Shadows::default().near_distance);
        assert_eq!(shadows.resolution, 2048);
        assert_eq!(
            Shadows {
                resolution: 0,
                ..Shadows::default()
            }
            .sanitized()
            .resolution,
            1024
        );
        assert_eq!(
            Shadows {
                resolution: u32::MAX,
                ..Shadows::default()
            }
            .sanitized()
            .resolution,
            4096
        );
        // A short shadow distance keeps the near map inside it.
        let short = Shadows {
            distance: 60.0,
            near_distance: 100.0,
            ..Shadows::default()
        }
        .sanitized();
        assert_eq!(short.near_distance, 30.0);
        // Cascades clamp to their range and split the near distance by
        // thirds.
        for (cascades, expected) in [(0, 1), (3, 3), (9, MAX_SHADOW_CASCADES)] {
            let shadows = Shadows {
                cascades,
                ..Shadows::default()
            };
            assert_eq!(shadows.sanitized().cascades, expected);
        }
        let reaches: Vec<f32> = Shadows {
            near_distance: 90.0,
            ..Shadows::default()
        }
        .cascade_reaches()
        .collect();
        assert_eq!(reaches, [10.0, 30.0, 90.0]);
        let occlusion = AmbientOcclusion {
            intensity: f32::NAN,
            radius: -1.0,
        }
        .sanitized();
        assert!(!occlusion.enabled());
        assert_eq!(occlusion.radius, AmbientOcclusion::default().radius);
        let bloom = Bloom {
            intensity: 9.0,
            threshold: f32::INFINITY,
        }
        .sanitized();
        assert_eq!(bloom.intensity, 4.0);
        assert_eq!(bloom.threshold, Bloom::default().threshold);
    }

    #[test]
    fn opt_in_effects_default_on_and_sanitize_their_ranges() {
        assert!(SunShafts::default().enabled());
        assert!(Reflections::default().enabled());
        assert_eq!(SunShafts::default().sanitized(), SunShafts::default());
        assert_eq!(Reflections::default().sanitized(), Reflections::default());
        let shafts = SunShafts {
            intensity: f32::NAN,
            anisotropy: 2.0,
            distance: -1.0,
        }
        .sanitized();
        assert!(!shafts.enabled());
        assert_eq!(shafts.anisotropy, 0.95);
        assert_eq!(shafts.distance, SunShafts::default().distance);
        let reflections = Reflections {
            intensity: 3.0,
            max_roughness: 0.0,
            distance: f32::INFINITY,
        }
        .sanitized();
        assert_eq!(reflections.intensity, 1.0);
        assert_eq!(reflections.max_roughness, 0.05);
        assert_eq!(reflections.distance, Reflections::default().distance);
    }
}
