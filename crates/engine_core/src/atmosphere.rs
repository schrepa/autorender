//! Scene lighting and atmosphere (decision 018): a directional sun, a sky
//! gradient that also colors ambient light, reflections and fog, and the
//! exposure and saturation of the final image. Every extracted frame carries
//! a copy.

use bevy_ecs::prelude::Resource;
use glam::Vec3;

/// Lighting of the rendered scene, read when frames are extracted.
///
/// Colors are linear RGB. The sun and sky may exceed 1: the renderer tone
/// maps the result. Vertical surfaces take their ambient light from the
/// horizon color, upward ones from the zenith and downward ones from the
/// ground, and fog takes the sky's color behind what it covers, so the
/// three sky colors set the scene's palette.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Atmosphere {
    /// Direction toward the sun; normalized when rendered.
    pub sun_direction: Vec3,
    /// Sun color times intensity.
    pub sun_color: Vec3,
    /// Sky straight up, at the horizon, and below it.
    pub zenith: Vec3,
    pub horizon: Vec3,
    pub ground: Vec3,
    /// Scale of the sky's ambient light on surfaces.
    pub ambient: f32,
    /// Fog begins this far from the camera.
    pub fog_start: f32,
    /// Fog density per unit of distance at height zero.
    pub fog_density: f32,
    /// How fast fog thins with height, per unit of height.
    pub fog_falloff: f32,
    /// The most fog can cover, in `[0, 1]`.
    pub fog_max: f32,
    /// Silhouette light in the horizon color, facing away from the camera.
    pub rim: f32,
    /// Scale of sun highlights and sky reflections; 1 is physically based.
    pub specular: f32,
    /// Scale of material emission: how strongly lamps, lit windows and other
    /// artificial lights show in this light (decision 019).
    pub emission: f32,
    /// Exposure before tone mapping.
    pub exposure: f32,
    /// Saturation after tone mapping; 1 keeps it.
    pub saturation: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self::golden_hour()
    }
}

impl Atmosphere {
    /// A high sun under a blue sky.
    pub fn day() -> Self {
        Self {
            sun_direction: Vec3::new(0.55, 0.75, 0.35),
            sun_color: Vec3::new(2.2, 2.05, 1.8),
            zenith: Vec3::new(0.16, 0.34, 0.85),
            horizon: Vec3::new(0.62, 0.74, 0.92),
            ground: Vec3::new(0.24, 0.21, 0.18),
            ambient: 0.6,
            fog_start: 40.0,
            fog_density: 0.0016,
            fog_falloff: 0.02,
            fog_max: 0.75,
            rim: 0.15,
            specular: 1.0,
            emission: 0.1,
            exposure: 0.9,
            saturation: 1.1,
        }
    }

    /// A low warm sun, a peach horizon and long blue shadows: the default.
    pub fn golden_hour() -> Self {
        Self {
            sun_direction: Vec3::new(0.82, 0.45, 0.1),
            sun_color: Vec3::new(2.6, 1.6, 0.85),
            zenith: Vec3::new(0.12, 0.22, 0.6),
            horizon: Vec3::new(1.05, 0.62, 0.45),
            ground: Vec3::new(0.2, 0.13, 0.12),
            ambient: 0.62,
            fog_start: 40.0,
            fog_density: 0.0022,
            fog_falloff: 0.022,
            fog_max: 0.75,
            rim: 0.45,
            specular: 1.0,
            emission: 0.5,
            exposure: 0.85,
            saturation: 1.15,
        }
    }

    /// The sun at the horizon under an indigo sky, with pink haze.
    pub fn dusk() -> Self {
        Self {
            sun_direction: Vec3::new(0.9, 0.07, 0.4),
            sun_color: Vec3::new(2.2, 0.85, 0.5),
            zenith: Vec3::new(0.05, 0.05, 0.24),
            horizon: Vec3::new(0.95, 0.36, 0.48),
            ground: Vec3::new(0.08, 0.05, 0.09),
            ambient: 0.65,
            fog_start: 40.0,
            fog_density: 0.0026,
            fog_falloff: 0.02,
            fog_max: 0.78,
            rim: 0.6,
            specular: 1.0,
            emission: 1.0,
            exposure: 0.95,
            saturation: 1.25,
        }
    }

    /// Cool moonlight under a dark sky, with a deep blue haze.
    pub fn night() -> Self {
        Self {
            sun_direction: Vec3::new(-0.35, 0.6, -0.45),
            sun_color: Vec3::new(0.35, 0.45, 0.85),
            zenith: Vec3::new(0.008, 0.012, 0.045),
            horizon: Vec3::new(0.07, 0.09, 0.22),
            ground: Vec3::new(0.02, 0.02, 0.04),
            ambient: 0.9,
            fog_start: 40.0,
            fog_density: 0.0028,
            fog_falloff: 0.02,
            fog_max: 0.85,
            rim: 0.8,
            specular: 1.0,
            emission: 1.0,
            exposure: 1.35,
            saturation: 1.15,
        }
    }

    /// This atmosphere when its values are usable: finite, colors and
    /// scales nonnegative, a nonzero sun direction and a fog maximum in
    /// `[0, 1]`. Otherwise the default.
    pub fn sanitized(self) -> Self {
        let colors = [self.sun_color, self.zenith, self.horizon, self.ground];
        let scales = [
            self.ambient,
            self.fog_start,
            self.fog_density,
            self.fog_falloff,
            self.rim,
            self.specular,
            self.emission,
            self.exposure,
            self.saturation,
        ];
        let usable = self.sun_direction.is_finite()
            && self.sun_direction.length_squared() > 0.0
            && colors
                .iter()
                .all(|color| color.is_finite() && color.min_element() >= 0.0)
            && scales
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && (0.0..=1.0).contains(&self.fog_max);
        if usable { self } else { Self::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_usable_and_unusable_values_fall_back() {
        for preset in [
            Atmosphere::day(),
            Atmosphere::golden_hour(),
            Atmosphere::dusk(),
            Atmosphere::night(),
        ] {
            assert_eq!(preset.sanitized(), preset);
        }
        let broken = [
            Atmosphere {
                sun_direction: Vec3::ZERO,
                ..Atmosphere::day()
            },
            Atmosphere {
                fog_density: f32::NAN,
                ..Atmosphere::day()
            },
            Atmosphere {
                horizon: Vec3::new(-1.0, 0.5, 0.5),
                ..Atmosphere::day()
            },
            Atmosphere {
                fog_max: 1.5,
                ..Atmosphere::day()
            },
        ];
        for atmosphere in broken {
            assert_eq!(atmosphere.sanitized(), Atmosphere::default());
        }
    }
}
