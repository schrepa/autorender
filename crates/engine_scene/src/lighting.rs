//! Scene version 6 entity fields: local lights (decision 023) and particle
//! emitters (decision 026). Each mirrors its runtime component field for
//! field, so inspected values paste back into scenes; angles are in degrees.

use engine_core::{LocalLight, ParticleBlend, ParticleEmitter, Spot, glam::Vec3};
use serde::{Deserialize, Serialize};

use crate::SceneError;

/// A point or spot light on this entity (decision 023). Requires version 6.
/// On an entity with a mesh it follows the mesh's displayed pose.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneLight {
    /// Linear color, each channel finite and nonnegative.
    pub color: [f32; 3],
    /// Light at one world unit; positive.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub intensity: f32,
    /// World distance beyond which the light has no effect; positive.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub range: f32,
    /// Position in the entity's local space, scaled with it.
    pub offset: [f32; 3],
    /// A cone; omitted for light in every direction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spot: Option<SceneSpot>,
}

impl Default for SceneLight {
    fn default() -> Self {
        Self {
            color: [1.0, 0.85, 0.6],
            intensity: 20.0,
            range: 12.0,
            offset: [0.0; 3],
            spot: None,
        }
    }
}

/// The cone of a spot light.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneSpot {
    /// Direction the cone points in the entity's local space; nonzero.
    pub direction: [f32; 3],
    /// Half-angle inside which the light is full, in degrees.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 89.0)))]
    pub inner_angle_degrees: f32,
    /// Half-angle beyond which it is dark, in degrees: at least the inner
    /// angle.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.1, max = 89.0)))]
    pub outer_angle_degrees: f32,
}

impl Default for SceneSpot {
    fn default() -> Self {
        Self {
            direction: [0.0, -1.0, 0.0],
            inner_angle_degrees: 25.0,
            outer_angle_degrees: 40.0,
        }
    }
}

fn invalid(path: &str, field: &str, rule: &str) -> SceneError {
    SceneError::new("invalid_light", format!("{path}.{field}"), rule.to_owned())
}

impl SceneLight {
    /// The component at authored path `path`, validated.
    pub fn runtime(&self, path: &str) -> Result<LocalLight, SceneError> {
        if !self
            .color
            .iter()
            .all(|&value| value.is_finite() && value >= 0.0)
        {
            return Err(invalid(
                path,
                "color",
                "channels must be finite and nonnegative",
            ));
        }
        if !(self.intensity.is_finite() && self.intensity > 0.0) {
            return Err(invalid(path, "intensity", "must be finite and positive"));
        }
        if !(self.range.is_finite() && self.range > 0.0 && self.range <= 10_000.0) {
            return Err(invalid(path, "range", "must be positive and at most 10000"));
        }
        if !self
            .offset
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 1.0e6)
        {
            return Err(invalid(path, "offset", "must be finite and within 1000000"));
        }
        let spot = match self.spot {
            Some(spot) => {
                let path = format!("{path}.spot");
                let direction = Vec3::from_array(spot.direction);
                if !direction.is_finite() || direction.length_squared() == 0.0 {
                    return Err(invalid(&path, "direction", "must be finite and nonzero"));
                }
                if !(0.0..=89.0).contains(&spot.inner_angle_degrees) {
                    return Err(invalid(&path, "inner_angle_degrees", "must be in [0, 89]"));
                }
                if !(spot.outer_angle_degrees
                    >= engine_core::math::max(spot.inner_angle_degrees, 0.1)
                    && spot.outer_angle_degrees <= 89.0)
                {
                    return Err(invalid(
                        &path,
                        "outer_angle_degrees",
                        "must be in [inner angle, 89] and at least 0.1",
                    ));
                }
                Some(Spot {
                    direction,
                    inner_angle: spot.inner_angle_degrees.to_radians(),
                    outer_angle: spot.outer_angle_degrees.to_radians(),
                })
            }
            None => None,
        };
        Ok(LocalLight {
            color: Vec3::from_array(self.color),
            intensity: self.intensity,
            range: self.range,
            offset: Vec3::from_array(self.offset),
            spot,
        })
    }
}

impl From<&LocalLight> for SceneLight {
    fn from(light: &LocalLight) -> Self {
        Self {
            color: light.color.to_array(),
            intensity: light.intensity,
            range: light.range,
            offset: light.offset.to_array(),
            spot: light.spot.map(|spot| SceneSpot {
                direction: spot.direction.to_array(),
                inner_angle_degrees: spot.inner_angle.to_degrees(),
                outer_angle_degrees: spot.outer_angle.to_degrees(),
            }),
        }
    }
}

/// How particles combine with what is behind them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneParticleBlend {
    /// Covering by alpha, lit by the sun and sky: smoke, dust, rain.
    #[default]
    Alpha,
    /// Adding light, unlit: sparks, fire.
    Additive,
}

/// A particle emitter on this entity (decision 026). Requires version 6.
/// Omitted fields take the engine's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneEmitter {
    /// Particles born per second; positive.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100000.0)))]
    pub rate: f32,
    /// Seconds each particle lives; positive.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 60.0)))]
    pub lifetime: f32,
    /// Initial speed in world units per second.
    pub speed: f32,
    /// Direction of emission in the entity's local space; nonzero.
    pub direction: [f32; 3],
    /// Half-angle of the cone of initial directions, in degrees; 180 emits
    /// in every direction.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 180.0)))]
    pub spread_degrees: f32,
    /// World acceleration: gravity, buoyancy or wind.
    pub acceleration: [f32; 3],
    /// Diameter at birth and at death, in world units.
    pub size: [f32; 2],
    /// Linear RGBA at birth and at death, channels in [0, 1]; alpha fades.
    pub color: [[f32; 4]; 2],
    /// Brightness multiple of the color, for glowing particles.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 64.0)))]
    pub emission: f32,
    pub blend: SceneParticleBlend,
    /// Seconds of travel particles stretch along their motion.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 10.0)))]
    pub stretch: f32,
    /// Birth position in the entity's local space.
    pub offset: [f32; 3],
    /// Radius of the ball particles start in.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub radius: f32,
    /// Varies random choices between otherwise identical emitters.
    pub seed: u32,
}

impl Default for SceneEmitter {
    fn default() -> Self {
        Self::from(&ParticleEmitter::default())
    }
}

impl From<&ParticleEmitter> for SceneEmitter {
    fn from(emitter: &ParticleEmitter) -> Self {
        Self {
            rate: emitter.rate,
            lifetime: emitter.lifetime,
            speed: emitter.speed,
            direction: emitter.direction.to_array(),
            spread_degrees: emitter.spread.to_degrees(),
            acceleration: emitter.acceleration.to_array(),
            size: emitter.size,
            color: emitter.color,
            emission: emitter.emission,
            blend: match emitter.blend {
                ParticleBlend::Alpha => SceneParticleBlend::Alpha,
                ParticleBlend::Additive => SceneParticleBlend::Additive,
            },
            stretch: emitter.stretch,
            offset: emitter.offset.to_array(),
            radius: emitter.radius,
            seed: emitter.seed,
        }
    }
}

impl SceneEmitter {
    /// The component at authored path `path`, validated.
    pub fn runtime(&self, path: &str) -> Result<ParticleEmitter, SceneError> {
        let invalid = |field: &str, rule: &str| {
            SceneError::new(
                "invalid_particles",
                format!("{path}.{field}"),
                rule.to_owned(),
            )
        };
        let finite3 = |value: [f32; 3]| value.iter().all(|v| v.is_finite() && v.abs() <= 1.0e6);
        if !(self.rate.is_finite() && self.rate > 0.0 && self.rate <= 100_000.0) {
            return Err(invalid("rate", "must be positive and at most 100000"));
        }
        if !(self.lifetime.is_finite() && self.lifetime > 0.0 && self.lifetime <= 60.0) {
            return Err(invalid("lifetime", "must be positive and at most 60"));
        }
        if f64::from(self.rate) * f64::from(self.lifetime)
            > f64::from(engine_core::MAX_EMITTER_PARTICLES)
        {
            return Err(invalid(
                "rate",
                "rate times lifetime must be at most the particle limit",
            ));
        }
        if !(self.speed.is_finite() && self.speed.abs() <= 1.0e4) {
            return Err(invalid("speed", "must be finite and within 10000"));
        }
        let direction = Vec3::from_array(self.direction);
        if !finite3(self.direction) || direction.length_squared() == 0.0 {
            return Err(invalid("direction", "must be finite and nonzero"));
        }
        if !(0.0..=180.0).contains(&self.spread_degrees) {
            return Err(invalid("spread_degrees", "must be in [0, 180]"));
        }
        if !finite3(self.acceleration) {
            return Err(invalid("acceleration", "must be finite"));
        }
        if !self
            .size
            .iter()
            .all(|size| size.is_finite() && (0.0..=1.0e4).contains(size))
        {
            return Err(invalid("size", "must be in [0, 10000]"));
        }
        if !self
            .color
            .iter()
            .flatten()
            .all(|channel| (0.0..=1.0).contains(channel))
        {
            return Err(invalid("color", "channels must be in [0, 1]"));
        }
        if !(0.0..=64.0).contains(&self.emission) {
            return Err(invalid("emission", "must be in [0, 64]"));
        }
        if !(0.0..=10.0).contains(&self.stretch) {
            return Err(invalid("stretch", "must be in [0, 10]"));
        }
        if !finite3(self.offset) {
            return Err(invalid("offset", "must be finite and within 1000000"));
        }
        if !(self.radius.is_finite() && (0.0..=1.0e4).contains(&self.radius)) {
            return Err(invalid("radius", "must be in [0, 10000]"));
        }
        Ok(ParticleEmitter {
            rate: self.rate,
            lifetime: self.lifetime,
            speed: self.speed,
            direction,
            spread: self.spread_degrees.to_radians(),
            acceleration: Vec3::from_array(self.acceleration),
            size: self.size,
            color: self.color,
            emission: self.emission,
            blend: match self.blend {
                SceneParticleBlend::Alpha => ParticleBlend::Alpha,
                SceneParticleBlend::Additive => ParticleBlend::Additive,
            },
            stretch: self.stretch,
            offset: Vec3::from_array(self.offset),
            radius: self.radius,
            seed: self.seed,
        })
    }
}
