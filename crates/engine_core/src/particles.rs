//! Particles: emitters on entities, simulated on the GPU.
//!
//! Every particle is a function of its emitter, its index and presentation
//! time, so frames keep no particle state: pausing freezes them, captures are
//! exact, and the CPU extracts only one record per emitter each frame.
//! Particles blend in the transparent pass, in any order.
//!
//! The [`ParticleSettings`] are inserted by default; without them, or when
//! disabled, frames draw no particles. Beyond `max_particles` the emitters
//! nearest the camera win and the others are skipped whole, counted as the
//! frame's dropped particles; without a budget only the GPU's buffer limits
//! bound them.

use bevy_ecs::{
    entity::EntityHashMap,
    prelude::{Component, Entity, Resource, World},
    query::QueryState,
};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use crate::extraction::{Nearest, Poses};
use crate::{RenderSnapshot, TransformIndex};
use engine_settings::{ExtraRules, InvalidSetting, Setting, settings};

/// Most particles one emitter keeps alive (its rate times its lifetime): a
/// validity bound far above real effects. Frames draw every emitter unless
/// [`ParticleSettings::max_particles`] sets a budget.
pub const MAX_EMITTER_PARTICLES: u32 = 1 << 24;

/// The widest an emitter's cone of directions opens, in degrees from its
/// axis: every direction.
pub const MAX_SPREAD_DEGREES: f32 = 180.0;

impl Setting for ParticleBlend {}

/// How particles combine with what is behind them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParticleBlend {
    /// Covering by their alpha, lit by the sun and sky: smoke, dust, rain.
    #[default]
    Alpha,
    /// Adding their light, unlit: sparks, fire, glowing embers.
    Additive,
}

settings! {
    unauthored;
    extra_rules;
    /// Emits particles from its entity.
    ///
    /// Particle `k` is born at `k / rate` seconds of presentation time at the
    /// emitter's position (plus `offset` and a random point within `radius`),
    /// leaves along `direction` within the `spread` cone at `speed`, accelerates
    /// by `acceleration` and dies after `lifetime`. Size and color blend from
    /// their first to their second value over its life. A moving emitter leaves
    /// its particles behind along its path. Emitters breaking their rules are
    /// skipped.
    #[derive(Component, Clone, Copy, Debug, PartialEq)]
    pub struct ParticleEmitter {
        /// Particles born per second.
        pub rate: f32 = 20.0 => above_to(0.0, 100_000.0),
        /// Seconds each particle lives.
        pub lifetime: f32 = 2.0 => above_to(0.0, 60.0),
        /// Initial speed in world units per second.
        pub speed: f32 = 1.0 => range(-10_000.0, 10_000.0),
        /// Direction of emission in the entity's local space; any nonzero length.
        pub direction: Vec3 = Vec3::Y => range(-1.0e6, 1.0e6),
        /// Half-angle in radians of the cone of initial directions; π emits
        /// in every direction.
        pub spread: f32 = 0.3 => range(0.0, MAX_SPREAD_DEGREES.to_radians()),
        /// World acceleration: gravity, buoyancy or wind.
        pub acceleration: Vec3 = Vec3::ZERO => range(-1.0e6, 1.0e6),
        /// Diameter at birth and at death, in world units.
        pub size: [f32; 2] = [0.3, 0.6] => range(0.0, 10_000.0),
        /// Linear RGBA at birth and at death; alpha fades particles.
        pub color: [[f32; 4]; 2] = [[1.0, 1.0, 1.0, 0.6], [1.0, 1.0, 1.0, 0.0]] => range(0.0, 1.0),
        /// Brightness multiple of the color, for glowing particles.
        pub emission: f32 = 1.0 => range(0.0, 64.0),
        /// How particles combine with what is behind them.
        pub blend: ParticleBlend = ParticleBlend::Alpha => any,
        /// Streaks along the motion: particles stretch by this many seconds of
        /// their travel.
        pub stretch: f32 = 0.0 => range(0.0, 10.0),
        /// Birth position in the entity's local space, scaled with it.
        pub offset: Vec3 = Vec3::ZERO => range(-1.0e6, 1.0e6),
        /// Radius of the ball around the birth position particles start in.
        pub radius: f32 = 0.0 => range(0.0, 10_000.0),
        /// Varies the random choices between otherwise identical emitters.
        pub seed: u32 = 0 => any,
    }
}

impl ExtraRules for ParticleEmitter {
    /// It points somewhere, and holds no more particles than the limit.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.direction.length_squared() == 0.0 {
            return Err(InvalidSetting::new("direction", "must be nonzero"));
        }
        if f64::from(self.rate) * f64::from(self.lifetime) > f64::from(MAX_EMITTER_PARTICLES) {
            return Err(InvalidSetting::new(
                "rate",
                "rate times lifetime must be at most the particle limit",
            ));
        }
        Ok(())
    }
}

impl ParticleEmitter {
    /// Particles alive at once: the rate times the lifetime, rounded up.
    pub fn capacity(&self) -> u32 {
        let alive = f64::from(self.rate) * f64::from(self.lifetime);
        if alive.is_finite() && alive > 0.0 {
            alive.ceil().min(f64::from(MAX_EMITTER_PARTICLES)) as u32
        } else {
            0
        }
    }
}

settings! {
    /// Particle settings. Requires version 6.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneParticles")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct ParticleSettings {
        /// Whether frames draw particles.
        pub enabled: bool = true => any,
        /// An optional budget of particles a frame draws; beyond it the emitters
        /// nearest the camera win. Omitted, every emitter draws.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_particles: Option<u32> = None => budget,
    }
}

impl ParticleSettings {
    /// What a world without these settings reads as: no particles.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// One emitter of a frame, in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderEmitter {
    /// Birth position and the emitter's velocity.
    pub origin: Vec3,
    pub velocity: Vec3,
    /// Unit emission direction.
    pub direction: Vec3,
    /// The emitter; its local `offset` and `direction` are already applied.
    pub emitter: ParticleEmitter,
    /// Particles allotted: [`ParticleEmitter::capacity`].
    pub count: u32,
}

/// Emitter records of each frame and the motion of their origins.
pub(crate) struct ParticleExtraction {
    query: QueryState<(Entity, &'static ParticleEmitter)>,
    // Pose store hints of the query's emitters, by position.
    hints: Vec<TransformIndex>,
    // Last origin, presentation time and velocity of each emitter, and the
    // extraction that saw it.
    motion: EntityHashMap<(Vec3, f32, Vec3, u64)>,
    extraction: u64,
    // This frame's usable emitters: squared distance to the eye, entity,
    // world pose and settings.
    candidates: Vec<(f32, Entity, Mat4, ParticleEmitter)>,
    nearest: Nearest,
    // The emitters of the last frame, and a revision that advances when a
    // frame's differ from them.
    previous: Vec<RenderEmitter>,
    pub(crate) revision: u64,
}

impl ParticleExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, &ParticleEmitter)>(),
            hints: Vec::new(),
            motion: EntityHashMap::default(),
            extraction: 0,
            candidates: Vec::new(),
            nearest: Nearest::default(),
            previous: Vec::new(),
            revision: 0,
        }
    }

    /// Fills the snapshot's emitters from poses prepared for this frame, or
    /// authoritative poses on `gpu` frames, and advances the revision when
    /// they changed. Beyond a particle budget, the emitters nearest the
    /// camera win.
    pub(crate) fn extract(&mut self, world: &World, snapshot: &mut RenderSnapshot, gpu: bool) {
        self.fill(world, snapshot, gpu);
        if snapshot.emitters != self.previous {
            self.revision += 1;
            self.previous.clone_from(&snapshot.emitters);
        }
    }

    fn fill(&mut self, world: &World, snapshot: &mut RenderSnapshot, gpu: bool) {
        // Extracted sanitized with the environment, before this.
        let settings = snapshot.environment.particles;
        snapshot.emitters.clear();
        snapshot.dropped_particles = 0;
        snapshot.invalid_emitters = 0;
        if !settings.enabled {
            self.motion.clear();
            return;
        }
        let poses = Poses::of(world, gpu);
        let eye = snapshot.camera_position;
        self.candidates.clear();
        let mut wanted = 0_u64;
        for (position, (entity, emitter)) in self.query.iter(world).enumerate() {
            if emitter.validate().is_err() {
                snapshot.invalid_emitters += 1;
                continue;
            }
            if self.hints.len() <= position {
                self.hints.resize(position + 1, TransformIndex::default());
            }
            let Some(model) = poses.model_indexed(entity, &mut self.hints[position]) else {
                continue;
            };
            wanted += u64::from(emitter.capacity());
            let distance = model.transform_point3(emitter.offset).distance_squared(eye);
            self.candidates.push((distance, entity, model, *emitter));
        }
        let budget = settings.max_particles.map(u64::from);
        let time = snapshot.elapsed_seconds;
        let mut total = 0_u64;
        self.extraction += 1;
        let extraction = self.extraction;
        let candidates = &self.candidates;
        let motion = &mut self.motion;
        let mut emit = |position: usize| {
            let (_, entity, model, emitter) = candidates[position];
            let count = emitter.capacity();
            if budget.is_some_and(|budget| total + u64::from(count) > budget) {
                snapshot.dropped_particles += count as usize;
                return;
            }
            total += u64::from(count);
            let origin = model.transform_point3(emitter.offset);
            // Velocity from the origin's travel since the previous frame;
            // paused frames keep the last one.
            let velocity = match motion.get(&entity) {
                Some(&(last, at, velocity, _)) => {
                    let elapsed = time - at;
                    if elapsed > 1.0e-4 && elapsed < 0.5 {
                        (origin - last) / elapsed
                    } else if elapsed.abs() <= 1.0e-4 {
                        velocity
                    } else {
                        Vec3::ZERO
                    }
                }
                None => Vec3::ZERO,
            };
            motion.insert(entity, (origin, time, velocity, extraction));
            snapshot.emitters.push(RenderEmitter {
                origin,
                velocity: if velocity.is_finite() {
                    velocity
                } else {
                    Vec3::ZERO
                },
                direction: model
                    .transform_vector3(emitter.direction)
                    .try_normalize()
                    .unwrap_or(Vec3::Y),
                emitter,
                count,
            });
        };
        if budget.is_some_and(|budget| wanted > budget) {
            // Beyond the budget the nearest emitters win, nearest first.
            self.nearest
                .nearest_first(candidates.iter().map(|candidate| candidate.0), usize::MAX)
                .for_each(&mut emit);
        } else {
            (0..candidates.len()).for_each(&mut emit);
        }
        if self.motion.len() > snapshot.emitters.len() {
            self.motion.retain(|_, motion| motion.3 == extraction);
        }
    }
}
