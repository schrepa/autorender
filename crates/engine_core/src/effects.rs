//! Rendering effects of the scene: sun shadows (decision 020), ambient
//! occlusion and bloom (decision 021). Every extracted frame carries sanitized
//! copies with the atmosphere; renderers skip the work of disabled effects.
//!
//! Each effect is its own resource, inserted by default: disable one with its
//! settings, or remove the resource to remove the effect. Without an
//! `Atmosphere` frames use the default one.

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
}

/// Shadows cast by the atmosphere's sun (decision 020).
///
/// Every rendered instance casts and receives them. One map covers `bounds`,
/// or without bounds a sphere of `distance` around the camera that moves
/// when the camera travels a quarter of it. Instances that stay still are
/// drawn into that map once and kept; moving ones are drawn every frame. A
/// sharper map follows the first `near_distance` of the view.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Shadows {
    pub enabled: bool,
    /// World region (minimum and maximum corners) whose shadows are kept.
    /// A region that holds the whole scene fits the map to it and never
    /// moves it; `None` follows the camera instead.
    pub bounds: Option<(Vec3, Vec3)>,
    /// Shadows fade out this far from the camera.
    pub distance: f32,
    /// View distance covered by the sharper map that follows the camera.
    pub near_distance: f32,
    /// Texels per side of the far maps: a power of two in `1024..=4096`.
    /// The near map has half as many.
    pub resolution: u32,
}

impl Default for Shadows {
    fn default() -> Self {
        Self {
            enabled: true,
            bounds: None,
            distance: 500.0,
            near_distance: 100.0,
            resolution: 2048,
        }
    }
}

impl Shadows {
    /// Usable settings: unusable bounds are dropped, a resolution rounds up
    /// to the next accepted power of two, and unusable distances (or a near
    /// distance beyond the shadow distance) fall back to the defaults.
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
            resolution: self.resolution.clamp(1024, 4096).next_power_of_two(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
