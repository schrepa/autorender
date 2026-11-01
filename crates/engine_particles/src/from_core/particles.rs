use std::sync::{Mutex, PoisonError};

use bevy_ecs::{
    entity::EntityHashMap,
    prelude::{Entity, Resource, World},
    query::QueryState,
};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use engine_core::extraction::{Nearest, Poses};
use engine_core::sections::environment_section;
use engine_core::{RenderSnapshot, TransformIndex};
use engine_settings::settings;

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

environment_section!(ParticleSettings, "particles", opt_in: false);

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
    pub cut: usize,
    pub invalid: usize,
    pub revision: u64,
}

#[derive(Resource, Default)]
pub struct ParticleExtractor(Mutex<ParticleExtraction>);

pub fn extract_particles(world: &World, snapshot: &mut RenderSnapshot) {
    world
        .resource::<ParticleExtractor>()
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .extract(world, snapshot);
}

#[derive(Default)]
struct ParticleExtraction {
    query: Option<QueryState<(Entity, &'static ParticleEmitter)>>,
    hints: Vec<TransformIndex>,
    motion: EntityHashMap<(Vec3, f32, Vec3, u64)>,
    extraction: u64,
    candidates: Vec<(f32, Entity, Mat4, ParticleEmitter)>,
    nearest: Nearest,
    previous: Vec<RenderEmitter>,
}

impl ParticleExtraction {
    fn extract(&mut self, world: &World, snapshot: &mut RenderSnapshot) {
        self.fill(world, snapshot);
        if snapshot.particles.list != self.previous {
            snapshot.particles.revision += 1;
            self.previous.clone_from(&snapshot.particles.list);
        }
    }

    fn fill(&mut self, world: &World, snapshot: &mut RenderSnapshot) {
        let settings = engine_core::section_in::<ParticleSettings>(world);
        snapshot.particles.list.clear();
        snapshot.particles.cut = 0;
        snapshot.particles.invalid = 0;
        if !settings.enabled {
            self.motion.clear();
            return;
        }
        if self.query.is_none() {
            self.query = world.try_query();
        }
        let poses = Poses::of(world);
        let eye = snapshot.detail_eye;
        self.candidates.clear();
        let mut wanted = 0_u64;
        if let Some(query) = &mut self.query {
            for (position, (entity, emitter)) in query.iter(world).enumerate() {
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
                snapshot.particles.cut += count as usize;
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
