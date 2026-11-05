use std::sync::{Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use glam::Vec3;
use serde::{Deserialize, Serialize};

pub const MAX_EMITTER_PARTICLES: u32 = 1 << 24;

pub const MAX_SPREAD_DEGREES: f32 = 180.0;

const MAX_BURSTS: usize = 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParticleBlend {
    #[default]
    Alpha,
    Additive,
    Mist,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParticleShape {
    Box,
    #[default]
    Ball,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleLook {
    pub lifetime: f32,
    pub lifetime_spread: f32,
    pub speed: f32,
    pub speed_spread: f32,
    pub direction: Vec3,
    pub spread: f32,
    pub acceleration: Vec3,
    pub shape: ParticleShape,
    pub extent: Vec3,
    pub size: [f32; 2],
    pub size_spread: f32,
    pub color: [[f32; 4]; 2],
    pub fade_in: f32,
    pub brightness: f32,
    pub blend: ParticleBlend,
    pub stretch: f32,
    pub seed: u32,
    pub formula: u32,
    pub params: [f32; 8],
    pub authored_degrees: Option<f32>,
}

impl Default for ParticleLook {
    fn default() -> Self {
        Self {
            lifetime: 2.0,
            lifetime_spread: 0.0,
            speed: 1.0,
            speed_spread: 0.25,
            direction: Vec3::Y,
            spread: 0.3,
            acceleration: Vec3::ZERO,
            shape: ParticleShape::Ball,
            extent: Vec3::splat(0.5),
            size: [0.3, 0.6],
            size_spread: 0.2,
            color: [[1.0, 1.0, 1.0, 0.6], [1.0, 1.0, 1.0, 0.0]],
            fade_in: 0.0,
            brightness: 1.0,
            blend: ParticleBlend::Alpha,
            stretch: 0.0,
            seed: 0,
            formula: 0,
            params: [0.0; 8],
            authored_degrees: None,
        }
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "AuthoredEmitter", into = "AuthoredEmitter")]
pub struct ParticleEmitter {
    pub rate: f32,
    pub offset: Vec3,
    pub emitting: bool,
    pub look: ParticleLook,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            rate: 20.0,
            offset: Vec3::ZERO,
            emitting: true,
            look: ParticleLook::default(),
        }
    }
}

impl ParticleEmitter {
    pub fn capacity(&self) -> u32 {
        let alive = f64::from(self.rate) * f64::from(self.look.lifetime);
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
    offset: Vec3,
    emitting: bool,
    lifetime: f32,
    lifetime_spread: f32,
    speed: f32,
    speed_spread: f32,
    direction: Vec3,
    spread_degrees: f32,
    acceleration: Vec3,
    shape: ParticleShape,
    extent: Vec3,
    size: [f32; 2],
    size_spread: f32,
    color: [[f32; 4]; 2],
    fade_in: f32,
    brightness: f32,
    blend: ParticleBlend,
    stretch: f32,
    seed: u32,
    formula: u32,
    params: [f32; 8],
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
            offset: authored.offset,
            emitting: authored.emitting,
            look: ParticleLook {
                lifetime: authored.lifetime,
                lifetime_spread: authored.lifetime_spread,
                speed: authored.speed,
                speed_spread: authored.speed_spread,
                direction: authored.direction,
                spread: authored.spread_degrees.to_radians(),
                acceleration: authored.acceleration,
                shape: authored.shape,
                extent: authored.extent,
                size: authored.size,
                size_spread: authored.size_spread,
                color: authored.color,
                fade_in: authored.fade_in,
                brightness: authored.brightness,
                blend: authored.blend,
                stretch: authored.stretch,
                seed: authored.seed,
                formula: authored.formula,
                params: authored.params,
                authored_degrees: Some(authored.spread_degrees),
            },
        }
    }
}

impl From<ParticleEmitter> for AuthoredEmitter {
    fn from(emitter: ParticleEmitter) -> Self {
        let look = emitter.look;
        Self {
            rate: emitter.rate,
            offset: emitter.offset,
            emitting: emitter.emitting,
            lifetime: look.lifetime,
            lifetime_spread: look.lifetime_spread,
            speed: look.speed,
            speed_spread: look.speed_spread,
            direction: look.direction,
            spread_degrees: look.spread.to_degrees(),
            acceleration: look.acceleration,
            shape: look.shape,
            extent: look.extent,
            size: look.size,
            size_spread: look.size_spread,
            color: look.color,
            fade_in: look.fade_in,
            brightness: look.brightness,
            blend: look.blend,
            stretch: look.stretch,
            seed: look.seed,
            formula: look.formula,
            params: look.params,
        }
    }
}

#[derive(Resource, Default)]
pub struct Transparency;

impl Slot for Transparency {
    const NAME: &'static str = "transparency";
    const WGSL: Option<&'static str> = Some(include_str!("places/transparency.wgsl"));
}

#[derive(Resource, Default)]
pub struct ParticleFormula;

impl Slot for ParticleFormula {
    const NAME: &'static str = "particle_formula";
    const WGSL: Option<&'static str> = Some(include_str!("places/particle_formula.wgsl"));
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct QueuedBurst {
    pub at: Vec3,
    pub count: u32,
    pub look: ParticleLook,
}

#[derive(Default)]
struct BurstQueue {
    bursts: Vec<QueuedBurst>,
    cut: u32,
}

#[derive(Resource, Default)]
pub struct ParticleBursts {
    queue: Mutex<BurstQueue>,
}

impl Slot for ParticleBursts {
    const NAME: &'static str = "particle_bursts";
}

impl ParticleBursts {
    pub fn burst(&self, at: Vec3, count: u32, look: ParticleLook) {
        let mut queue = self.lock();
        if queue.bursts.len() < MAX_BURSTS {
            queue.bursts.push(QueuedBurst { at, count, look });
        } else {
            queue.cut = queue.cut.saturating_add(count);
        }
    }

    #[doc(hidden)]
    pub fn take_bursts(&self) -> Vec<QueuedBurst> {
        std::mem::take(&mut self.lock().bursts)
    }

    #[doc(hidden)]
    pub fn take_cut(&self) -> u32 {
        std::mem::take(&mut self.lock().cut)
    }

    fn lock(&self) -> MutexGuard<'_, BurstQueue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
