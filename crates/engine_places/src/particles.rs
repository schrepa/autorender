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
#[serde(default, deny_unknown_fields)]
pub struct ParticleEmitter {
    pub rate: f32,
    pub lifetime: f32,
    pub speed: f32,
    pub direction: Vec3,
    #[serde(rename = "spread_degrees", with = "crate::degrees", default = "authored_spread")]
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
        }
    }
}

fn authored_spread() -> f32 {
    ParticleEmitter::default().spread.to_degrees().to_radians()
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

#[derive(Resource, Default)]
pub struct Transparency;

impl Slot for Transparency {
    const NAME: &'static str = "transparency";
    const WGSL: Option<&'static str> = Some(include_str!("places/transparency.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::TRANSPARENCY];
}
