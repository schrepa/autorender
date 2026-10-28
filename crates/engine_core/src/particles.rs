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

pub const MAX_EMITTER_PARTICLES: u32 = 1 << 24;

pub const MAX_SPREAD_DEGREES: f32 = 180.0;

impl Setting for ParticleBlend {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParticleBlend {
    #[default]
    Alpha,
    Additive,
    Mist,
}

settings! {
    unauthored;
    extra_rules;
    #[derive(Component, Clone, Copy, Debug, PartialEq)]
    pub struct ParticleEmitter {
        pub rate: f32 = 20.0 => above_to(0.0, 100_000.0),
        pub lifetime: f32 = 2.0 => above_to(0.0, 60.0),
        pub speed: f32 = 1.0 => range(-10_000.0, 10_000.0),
        pub direction: Vec3 = Vec3::Y => range(-1.0e6, 1.0e6),
        pub spread: f32 = 0.3 => range(0.0, MAX_SPREAD_DEGREES.to_radians()),
        pub acceleration: Vec3 = Vec3::ZERO => range(-1.0e6, 1.0e6),
        pub size: [f32; 2] = [0.3, 0.6] => range(0.0, 10_000.0),
        pub color: [[f32; 4]; 2] = [[1.0, 1.0, 1.0, 0.6], [1.0, 1.0, 1.0, 0.0]] => range(0.0, 1.0),
        pub emission: f32 = 1.0 => range(0.0, 64.0),
        pub blend: ParticleBlend = ParticleBlend::Alpha => any,
        pub stretch: f32 = 0.0 => range(0.0, 10.0),
        pub offset: Vec3 = Vec3::ZERO => range(-1.0e6, 1.0e6),
        pub radius: f32 = 0.0 => range(0.0, 10_000.0),
        pub seed: u32 = 0 => any,
    }
}

impl ExtraRules for ParticleEmitter {
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
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneParticles")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct ParticleSettings {
        pub enabled: bool = true => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_particles: Option<u32> = None => budget,
    }
}

impl ParticleSettings {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderEmitter {
    pub origin: Vec3,
    pub velocity: Vec3,
    pub direction: Vec3,
    pub emitter: ParticleEmitter,
    pub count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct RenderParticles {
    pub list: Vec<RenderEmitter>,
    pub dropped: usize,
    pub invalid: usize,
}

pub(crate) struct ParticleExtraction {
    query: QueryState<(Entity, &'static ParticleEmitter)>,
    hints: Vec<TransformIndex>,
    motion: EntityHashMap<(Vec3, f32, Vec3, u64)>,
    extraction: u64,
    candidates: Vec<(f32, Entity, Mat4, ParticleEmitter)>,
    nearest: Nearest,
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

    pub(crate) fn extract(&mut self, world: &World, snapshot: &mut RenderSnapshot, gpu: bool) {
        self.fill(world, snapshot, gpu);
        if snapshot.particles.list != self.previous {
            self.revision += 1;
            self.previous.clone_from(&snapshot.particles.list);
        }
    }

    fn fill(&mut self, world: &World, snapshot: &mut RenderSnapshot, gpu: bool) {
        let settings = snapshot.environment.particles;
        snapshot.particles.list.clear();
        snapshot.particles.dropped = 0;
        snapshot.particles.invalid = 0;
        if !settings.enabled {
            self.motion.clear();
            return;
        }
        let poses = Poses::of(world, gpu);
        let eye = snapshot.detail_eye;
        self.candidates.clear();
        let mut wanted = 0_u64;
        for (position, (entity, emitter)) in self.query.iter(world).enumerate() {
            if emitter.validate().is_err() {
                snapshot.particles.invalid += 1;
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
                snapshot.particles.dropped += count as usize;
                return;
            }
            total += u64::from(count);
            let origin = model.transform_point3(emitter.offset);
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
            snapshot.particles.list.push(RenderEmitter {
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
            self.nearest
                .nearest_first(candidates.iter().map(|candidate| candidate.0), usize::MAX)
                .for_each(&mut emit);
        } else {
            (0..candidates.len()).for_each(&mut emit);
        }
        if self.motion.len() > snapshot.particles.list.len() {
            self.motion.retain(|_, motion| motion.3 == extraction);
        }
    }
}
