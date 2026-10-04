//! Particles (decision 026): emitters on entities, simulated on the GPU.
//!
//! Every particle is a function of its emitter, its index and presentation
//! time, so frames keep no particle state: pausing freezes them, captures are
//! exact, and the CPU extracts only one record per emitter each frame.
//! Particles blend in the transparent pass, in any order.
//!
//! The [`Particles`] settings are inserted by default; without them, or when
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

use crate::environment::{InvalidSetting, check_budget};
use crate::extraction::{Nearest, Poses};
use crate::{RenderSnapshot, TransformIndex};

/// Most particles one emitter keeps alive (its rate times its lifetime): a
/// validity bound far above real effects. Frames draw every emitter unless
/// [`Particles::max_particles`] sets a budget.
pub const MAX_EMITTER_PARTICLES: u32 = 1 << 24;

/// How particles combine with what is behind them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParticleBlend {
    /// Covering by their alpha, lit by the sun and sky: smoke, dust, rain.
    #[default]
    Alpha,
    /// Adding their light, unlit: sparks, fire, glowing embers.
    Additive,
}

/// Emits particles from its entity (decision 026).
///
/// Particle `k` is born at `k / rate` seconds of presentation time at the
/// emitter's position (plus `offset` and a random point within `radius`),
/// leaves along `direction` within the `spread` cone at `speed`, accelerates
/// by `acceleration` and dies after `lifetime`. Size and color blend from
/// their first to their second value over its life. A moving emitter leaves
/// its particles behind along its path. Unusable values skip the emitter.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ParticleEmitter {
    /// Particles born per second.
    pub rate: f32,
    /// Seconds each particle lives.
    pub lifetime: f32,
    /// Initial speed in world units per second.
    pub speed: f32,
    /// Direction of emission in the entity's local space; any nonzero length.
    pub direction: Vec3,
    /// Half-angle in radians of the cone of initial directions; π emits
    /// in every direction.
    pub spread: f32,
    /// World acceleration: gravity, buoyancy or wind.
    pub acceleration: Vec3,
    /// Diameter at birth and at death, in world units.
    pub size: [f32; 2],
    /// Linear RGBA at birth and at death; alpha fades particles.
    pub color: [[f32; 4]; 2],
    /// Brightness multiple of the color, for glowing particles.
    pub emission: f32,
    pub blend: ParticleBlend,
    /// Streaks along the motion: particles stretch by this many seconds of
    /// their travel.
    pub stretch: f32,
    /// Birth position in the entity's local space, scaled with it.
    pub offset: Vec3,
    /// Radius of the ball around the birth position particles start in.
    pub radius: f32,
    /// Varies the random choices between otherwise identical emitters.
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

    /// Whether renderers can draw this emitter (see the type's documentation).
    pub fn is_usable(&self) -> bool {
        let finite = |value: f32| value.is_finite();
        let colors = self
            .color
            .iter()
            .flatten()
            .all(|&value| value.is_finite() && value >= 0.0);
        self.rate.is_finite()
            && self.rate > 0.0
            && self.lifetime.is_finite()
            && self.lifetime > 0.0
            && finite(self.speed)
            && self.direction.is_finite()
            && self.direction.length_squared() > 0.0
            && finite(self.spread)
            && self.acceleration.is_finite()
            && self
                .size
                .iter()
                .all(|&size| size.is_finite() && size >= 0.0)
            && colors
            && self.emission.is_finite()
            && self.emission >= 0.0
            && finite(self.stretch)
            && self.stretch >= 0.0
            && self.offset.is_finite()
            && self.radius.is_finite()
            && self.radius >= 0.0
    }
}

/// Particle settings (decision 026). Requires version 6.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneParticles")
)]
#[serde(default, deny_unknown_fields)]
pub struct Particles {
    pub enabled: bool,
    /// An optional budget of particles a frame draws; beyond it the emitters
    /// nearest the camera win. Omitted, every emitter draws.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_particles: Option<u32>,
}

impl Default for Particles {
    fn default() -> Self {
        Self {
            enabled: true,
            max_particles: None,
        }
    }
}

impl Particles {
    /// Ok for a budget of at least one particle.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_budget("max_particles", self.max_particles)
    }

    /// A budget of at least one particle.
    pub fn sanitized(self) -> Self {
        Self {
            enabled: self.enabled,
            max_particles: self.max_particles.map(|budget| budget.max(1)),
        }
    }
}

/// One emitter of a frame (decision 026), in world space.
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
        if !settings.enabled {
            self.motion.clear();
            return;
        }
        let poses = Poses::of(world, gpu);
        let eye = snapshot.camera_position;
        self.candidates.clear();
        let mut wanted = 0_u64;
        for (position, (entity, emitter)) in self.query.iter(world).enumerate() {
            if !emitter.is_usable() {
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
