use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use glam::Vec3;
use serde::{Deserialize, Serialize};

pub const MAX_EMITTER_PARTICLES: u32 = 1 << 24;

pub const MAX_SPREAD_DEGREES: f32 = 180.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParticleBlend {
    #[default]
    Alpha,
    Additive,
    Mist,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "AuthoredEmitter", into = "AuthoredEmitter")]
pub struct ParticleEmitter {
    pub rate: f32,
    pub lifetime: f32,
    pub speed: f32,
    pub direction: Vec3,
    pub spread: f32,
    pub acceleration: Vec3,
    pub size: [f32; 2],
    pub color: [[f32; 4]; 2],
    pub emission: f32,
    pub blend: ParticleBlend,
    pub stretch: f32,
    pub offset: Vec3,
    pub radius: f32,
    pub seed: u32,
    pub authored_degrees: Option<f32>,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            rate: 20.0,
            lifetime: 2.0,
            speed: 1.0,
            direction: Vec3::Y,
            spread: 0.3,
            acceleration: Vec3::ZERO,
            size: [0.3, 0.6],
            color: [[1.0, 1.0, 1.0, 0.6], [1.0, 1.0, 1.0, 0.0]],
            emission: 1.0,
            blend: ParticleBlend::Alpha,
            stretch: 0.0,
            offset: Vec3::ZERO,
            radius: 0.0,
            seed: 0,
            authored_degrees: None,
        }
    }
}

impl ParticleEmitter {
    pub fn capacity(&self) -> u32 {
        let alive = f64::from(self.rate) * f64::from(self.lifetime);
        if alive.is_finite() && alive > 0.0 {
            alive.ceil().min(f64::from(MAX_EMITTER_PARTICLES)) as u32
        } else {
            0
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AuthoredEmitter {
    rate: f32,
    lifetime: f32,
    speed: f32,
    direction: Vec3,
    spread_degrees: f32,
    acceleration: Vec3,
    size: [f32; 2],
    color: [[f32; 4]; 2],
    emission: f32,
    blend: ParticleBlend,
    stretch: f32,
    offset: Vec3,
    radius: f32,
    seed: u32,
}

impl Default for AuthoredEmitter {
    fn default() -> Self {
        Self::from(ParticleEmitter::default())
    }
}

impl From<AuthoredEmitter> for ParticleEmitter {
    fn from(authored: AuthoredEmitter) -> Self {
        Self {
            rate: authored.rate,
            lifetime: authored.lifetime,
            speed: authored.speed,
            direction: authored.direction,
            spread: authored.spread_degrees.to_radians(),
            acceleration: authored.acceleration,
            size: authored.size,
            color: authored.color,
            emission: authored.emission,
            blend: authored.blend,
            stretch: authored.stretch,
            offset: authored.offset,
            radius: authored.radius,
            seed: authored.seed,
            authored_degrees: Some(authored.spread_degrees),
        }
    }
}

impl From<ParticleEmitter> for AuthoredEmitter {
    fn from(emitter: ParticleEmitter) -> Self {
        Self {
            rate: emitter.rate,
            lifetime: emitter.lifetime,
            speed: emitter.speed,
            direction: emitter.direction,
            spread_degrees: emitter.spread.to_degrees(),
            acceleration: emitter.acceleration,
            size: emitter.size,
            color: emitter.color,
            emission: emitter.emission,
            blend: emitter.blend,
            stretch: emitter.stretch,
            offset: emitter.offset,
            radius: emitter.radius,
            seed: emitter.seed,
        }
    }
}

#[derive(Resource, Default)]
pub struct Transparency;

impl Slot for Transparency {
    const NAME: &'static str = "transparency";
    const WGSL: Option<&'static str> = Some(include_str!("places/transparency.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::TRANSPARENCY];
}
