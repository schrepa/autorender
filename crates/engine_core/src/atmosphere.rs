//! Scene lighting and atmosphere (decision 018): a directional sun, a sky
//! gradient that also colors ambient light, reflections and fog, aerial
//! perspective (decision 053), and the grade of the final image: exposure,
//! contrast, saturation, vibrance and split toning (decision 027). Every
//! extracted frame carries a copy.

use bevy_ecs::prelude::Resource;
use glam::Vec3;

use crate::environment::InvalidSetting;

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
    /// Aerial perspective (decision 053): beyond `aerial_start` from the
    /// camera, air dims what lies behind it toward the light it scatters,
    /// with `aerial_density` extinction per unit of distance at height zero
    /// (0 turns it off), thinning by `aerial_falloff` per unit of height.
    /// Distant ridges fade into `aerial_color` away from the sun and into a
    /// glow of the sun's color, `aerial_glow` strong, toward it. Unlike fog
    /// it is unbounded and spans kilometers.
    pub aerial_start: f32,
    pub aerial_density: f32,
    pub aerial_falloff: f32,
    pub aerial_color: Vec3,
    pub aerial_glow: f32,
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
    /// Contrast around mid grey after tone mapping (decision 027): the slope
    /// of an S-curve that keeps black and white, from 0 (flat) through 1
    /// (unchanged) to its steepest at 1.5.
    pub contrast: f32,
    /// Saturation added to muted colors more than to vivid ones; 0 adds none.
    pub vibrance: f32,
    /// Hue of dark and of bright tones after tone mapping (split toning), as
    /// color gains scaled to keep luminance; white, like black, leaves tones
    /// as they are. Cool shadows against warm highlights read as vivid.
    pub shadow_tint: Vec3,
    pub highlight_tint: Vec3,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self::golden_hour()
    }
}

impl Atmosphere {
    /// A high sun under a deep blue sky, with clear air, sky-blue shadows and
    /// sunlit highlights.
    pub fn day() -> Self {
        Self {
            sun_direction: Vec3::new(0.55, 0.75, 0.35),
            sun_color: Vec3::new(2.3, 2.1, 1.75),
            zenith: Vec3::new(0.1, 0.3, 0.9),
            horizon: Vec3::new(0.55, 0.75, 1.0),
            ground: Vec3::new(0.24, 0.21, 0.18),
            ambient: 0.6,
            fog_start: 80.0,
            fog_density: 0.0011,
            fog_falloff: 0.02,
            fog_max: 0.4,
            aerial_start: 1000.0,
            aerial_density: 1.0e-4,
            aerial_falloff: 5.5e-4,
            aerial_color: Vec3::new(0.56, 0.72, 1.0),
            aerial_glow: 0.25,
            rim: 0.15,
            specular: 1.0,
            emission: 0.1,
            exposure: 0.85,
            saturation: 1.2,
            contrast: 1.12,
            vibrance: 0.35,
            shadow_tint: Vec3::new(0.85, 0.95, 1.2),
            highlight_tint: Vec3::new(1.08, 1.02, 0.92),
        }
    }

    /// A low warm sun, an amber horizon under a deep blue zenith, and violet
    /// shadows against golden highlights: the default.
    pub fn golden_hour() -> Self {
        Self {
            sun_direction: Vec3::new(0.82, 0.45, 0.1),
            sun_color: Vec3::new(2.6, 1.6, 0.85),
            zenith: Vec3::new(0.05, 0.16, 0.62),
            horizon: Vec3::new(1.15, 0.56, 0.32),
            ground: Vec3::new(0.2, 0.13, 0.12),
            ambient: 0.62,
            fog_start: 80.0,
            fog_density: 0.0014,
            fog_falloff: 0.022,
            fog_max: 0.45,
            aerial_start: 1000.0,
            aerial_density: 1.1e-4,
            aerial_falloff: 6.0e-4,
            aerial_color: Vec3::new(0.36, 0.48, 0.92),
            aerial_glow: 0.7,
            rim: 0.45,
            specular: 1.0,
            emission: 0.5,
            exposure: 0.75,
            saturation: 1.2,
            contrast: 1.15,
            vibrance: 0.4,
            shadow_tint: Vec3::new(0.75, 0.9, 1.35),
            highlight_tint: Vec3::new(1.15, 1.0, 0.85),
        }
    }

    /// The sun at the horizon under an indigo sky, with pink haze, blue-violet
    /// shadows and coral highlights.
    pub fn dusk() -> Self {
        Self {
            sun_direction: Vec3::new(0.9, 0.07, 0.4),
            sun_color: Vec3::new(2.2, 0.85, 0.5),
            zenith: Vec3::new(0.03, 0.04, 0.32),
            horizon: Vec3::new(1.0, 0.36, 0.42),
            ground: Vec3::new(0.08, 0.05, 0.09),
            ambient: 0.65,
            fog_start: 60.0,
            fog_density: 0.0016,
            fog_falloff: 0.02,
            fog_max: 0.45,
            aerial_start: 800.0,
            aerial_density: 1.4e-4,
            aerial_falloff: 6.0e-4,
            aerial_color: Vec3::new(0.3, 0.3, 0.66),
            aerial_glow: 0.9,
            rim: 0.6,
            specular: 1.0,
            emission: 1.0,
            exposure: 0.95,
            saturation: 1.15,
            contrast: 1.18,
            vibrance: 0.35,
            shadow_tint: Vec3::new(0.75, 0.8, 1.45),
            highlight_tint: Vec3::new(1.25, 0.95, 0.75),
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
            fog_density: 0.0022,
            fog_falloff: 0.02,
            fog_max: 0.7,
            aerial_start: 600.0,
            aerial_density: 1.6e-4,
            aerial_falloff: 6.0e-4,
            aerial_color: Vec3::new(0.05, 0.07, 0.16),
            aerial_glow: 0.15,
            rim: 0.8,
            specular: 1.0,
            emission: 1.0,
            exposure: 1.35,
            saturation: 1.2,
            contrast: 1.2,
            vibrance: 0.5,
            shadow_tint: Vec3::new(0.8, 0.9, 1.35),
            highlight_tint: Vec3::new(1.15, 1.0, 0.85),
        }
    }

    /// Ok when every value is usable: finite, colors and scales
    /// nonnegative, a nonzero sun direction and a fog maximum of at most 1.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        let colors = [
            ("sun_color", self.sun_color),
            ("zenith", self.zenith),
            ("horizon", self.horizon),
            ("ground", self.ground),
            ("shadow_tint", self.shadow_tint),
            ("highlight_tint", self.highlight_tint),
            ("aerial_color", self.aerial_color),
        ];
        for (field, color) in colors {
            if !color.is_finite() || color.min_element() < 0.0 {
                return Err(InvalidSetting::new(
                    field,
                    "color channels must be finite and nonnegative",
                ));
            }
        }
        if !self.sun_direction.is_finite() || self.sun_direction.length_squared() == 0.0 {
            return Err(InvalidSetting::new(
                "sun_direction",
                "must be finite and nonzero",
            ));
        }
        let scalars = [
            ("ambient", self.ambient),
            ("fog_start", self.fog_start),
            ("fog_density", self.fog_density),
            ("fog_falloff", self.fog_falloff),
            ("fog_max", self.fog_max),
            ("aerial_start", self.aerial_start),
            ("aerial_density", self.aerial_density),
            ("aerial_falloff", self.aerial_falloff),
            ("aerial_glow", self.aerial_glow),
            ("rim", self.rim),
            ("specular", self.specular),
            ("emission", self.emission),
            ("exposure", self.exposure),
            ("saturation", self.saturation),
            ("contrast", self.contrast),
            ("vibrance", self.vibrance),
        ];
        for (field, value) in scalars {
            if !value.is_finite() || value < 0.0 {
                return Err(InvalidSetting::new(field, "must be finite and nonnegative"));
            }
        }
        if self.fog_max > 1.0 {
            return Err(InvalidSetting::new("fog_max", "must be at most 1"));
        }
        Ok(())
    }

    /// This atmosphere when [`Self::validate`] accepts it, otherwise the
    /// default.
    pub fn sanitized(self) -> Self {
        if self.validate().is_ok() {
            self
        } else {
            Self::default()
        }
    }
}
