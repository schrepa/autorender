use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{
    Allowed, Clock, Cut, Drawn, DrawnPoses, DrawnTable, Error, ErrorCode, Expected, FrameTime,
    GlobalTransforms, InvalidSetting, Nearest, Refusals, Rule, Views, fields, gpu_struct, settings,
    stats,
};
use engine_places_rendering::particles::{
    MAX_EMITTER_PARTICLES, MAX_SPREAD_DEGREES, ParticleBlend, ParticleBursts, ParticleEmitter,
    ParticleLook, ParticleShape, QueuedBurst,
};
use glam::{Mat4, Quat, Vec3};

use crate::PARTICLES;

engine_core::gpu_consts! {
    pub(crate) const NO_SLOT: u32 = u32::MAX;
}

settings! {
    #[section("particles")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneParticles")]
    pub struct ParticleSettings {
        pub enabled: bool = true => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_particles: Option<u32> = None => budget,
    }
}

stats! {
    #[section("particle_stats")]
    pub struct ParticleStats {
        pub drawn: u32,
        pub particle_cut: Cut,
    }
}

gpu_struct! {
    pub(crate) struct Emitter {
        origin: [f32; 3],
        rate: f32,
        velocity: [f32; 3],
        lifetime: f32,
        direction: [f32; 3],
        speed: f32,
        acceleration: [f32; 3],
        spread: f32,
        extent: [f32; 3],
        ball: u32,
        rotation: [f32; 4],
        start_color: [f32; 4],
        end_color: [f32; 4],
        size: [f32; 2],
        brightness: f32,
        stretch: f32,
        speed_spread: f32,
        size_spread: f32,
        lifetime_spread: f32,
        fade_in: f32,
        start: f32,
        stop: f32,
        seed: u32,
        first: u32,
        count: u32,
        formula: u32,
        now: f32,
        slot: u32,
        params: [f32; 8],
    }
}

#[derive(Clone, Copy)]
struct RenderEmitter {
    origin: Vec3,
    velocity: Vec3,
    direction: Vec3,
    rotation: Quat,
    look: ParticleLook,
    rate: f32,
    start: f32,
    stop: f32,
    now: f32,
    count: u32,
    slot: u32,
}

impl Emitter {
    fn new(emitter: &RenderEmitter, first: u32) -> Self {
        let look = &emitter.look;
        Self {
            origin: emitter.origin.to_array(),
            rate: emitter.rate,
            velocity: emitter.velocity.to_array(),
            lifetime: look.lifetime,
            direction: emitter.direction.to_array(),
            speed: look.speed,
            acceleration: look.acceleration.to_array(),
            spread: engine_core::math::cos(engine_core::math::clamp(
                look.spread,
                0.0,
                std::f32::consts::PI,
            )),
            extent: look.extent.to_array(),
            ball: u32::from(look.shape == ParticleShape::Ball),
            rotation: emitter.rotation.to_array(),
            start_color: look.color[0],
            end_color: look.color[1],
            size: look.size,
            brightness: look.brightness,
            stretch: look.stretch,
            speed_spread: look.speed_spread,
            size_spread: look.size_spread,
            lifetime_spread: look.lifetime_spread,
            fade_in: look.fade_in,
            start: emitter.start,
            stop: emitter.stop,
            seed: look.seed,
            first,
            count: emitter.count,
            formula: look.formula,
            now: emitter.now,
            slot: emitter.slot,
            params: look.params,
        }
    }
}

pub(crate) fn emitter_rule(field: &str) -> Option<Rule> {
    match field {
        "rate" => Some(fields::above_to(0.0, 100_000.0)),
        "lifetime" => Some(fields::above_to(0.0, 60.0)),
        "speed" => Some(fields::range(-10_000.0, 10_000.0)),
        "direction" | "acceleration" | "offset" => Some(fields::range(-1.0e6, 1.0e6)),
        "spread" => Some(fields::range(0.0, MAX_SPREAD_DEGREES.to_radians())),
        "size" | "extent" => Some(fields::range(0.0, 10_000.0)),
        "color" | "speed_spread" | "size_spread" | "lifetime_spread" | "fade_in" => {
            Some(fields::range(0.0, 1.0))
        }
        "brightness" => Some(fields::above_to(0.0, 64.0)),
        "stretch" => Some(fields::range(0.0, 10.0)),
        "params" => Some(fields::finite()),
        _ => None,
    }
}

pub(crate) fn emitter_extra(
    direction: Vec3,
    rate: f32,
    lifetime: f32,
) -> Result<(), InvalidSetting> {
    if direction.length_squared() == 0.0 {
        return Err(InvalidSetting::new("direction", Expected::Nonzero).got(direction.to_array()));
    }
    let particles = f64::from(rate) * f64::from(lifetime);
    if particles > f64::from(MAX_EMITTER_PARTICLES) {
        return Err(InvalidSetting::broken("rate", Rule::Any)
            .allowed(Allowed::AtMost {
                max: u64::from(MAX_EMITTER_PARTICLES),
            })
            .got(particles));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Emission {
    Stopped { at: f32 },
    Resumed { stopped: f32, at: f32 },
}

impl Emission {
    fn next(held: Option<Self>, emitting: bool, now: f32, seen: bool) -> Option<Self> {
        let stop = if seen { now } else { f32::MIN };
        match (held, emitting) {
            (None, true) => None,
            (None, false) | (Some(Self::Resumed { .. }), false) => Some(Self::Stopped { at: stop }),
            (Some(Self::Stopped { at }), true) => Some(Self::Resumed {
                stopped: at,
                at: now,
            }),
            (Some(held @ Self::Stopped { .. }), false)
            | (Some(held @ Self::Resumed { .. }), true) => Some(held),
        }
    }

    fn window(held: Option<Self>) -> (f32, f32) {
        match held {
            None => (f32::MAX, 0.0),
            Some(Self::Stopped { at }) => (at, f32::MAX),
            Some(Self::Resumed { stopped, at }) => (stopped, at),
        }
    }

    fn drained(held: Option<Self>, now: f32, lifetime: f32) -> bool {
        match held {
            Some(Self::Stopped { at }) => now - at > lifetime,
            Some(Self::Resumed { .. }) | None => false,
        }
    }
}

struct Candidate {
    distance: f32,
    entity: Option<Entity>,
    emitter: RenderEmitter,
}

fn rotation_of(model: Mat4) -> Quat {
    let (_, rotation, _) = model.to_scale_rotation_translation();
    if rotation.is_finite() {
        rotation
    } else {
        Quat::IDENTITY
    }
}

#[derive(Resource, Default)]
pub(crate) struct Gather {
    motion: HashMap<Entity, (Vec3, f32, Vec3, u64)>,
    emissions: HashMap<Entity, Emission>,
    bursts: Vec<(QueuedBurst, f32)>,
    bursts_taken: u32,
    extraction: u64,
    candidates: Vec<Candidate>,
    nearest: Nearest,
    list: Vec<RenderEmitter>,
    cut: usize,
}

#[derive(Resource, Default)]
pub(crate) struct Packed {
    pub(crate) records: Vec<Emitter>,
    pub(crate) alpha: u32,
    pub(crate) mist: u32,
    pub(crate) total: u32,
    pub(crate) emits: bool,
    cut: u64,
}

impl Gather {
    fn emission(&mut self, entity: Entity, emitting: bool, now: f32) -> Option<Emission> {
        let held = self.emissions.get(&entity).copied();
        let next = Emission::next(held, emitting, now, self.motion.contains_key(&entity));
        if let Some(next) = next {
            self.emissions.insert(entity, next);
        }
        next
    }

    fn steady(
        &mut self,
        (entity, slot, emitter): (Entity, u32, &ParticleEmitter),
        model: Mat4,
        eyes: &[Vec3],
        now: f32,
    ) -> Option<Candidate> {
        let emission = self.emission(entity, emitter.emitting, now);
        if Emission::drained(emission, now, emitter.look.lifetime) {
            return None;
        }
        let (stop, start) = Emission::window(emission);
        let origin = model.transform_point3(emitter.offset);
        Some(Candidate {
            distance: closest(origin, eyes),
            entity: Some(entity),
            emitter: RenderEmitter {
                origin,
                velocity: Vec3::ZERO,
                direction: model
                    .transform_vector3(emitter.look.direction)
                    .try_normalize()
                    .unwrap_or(Vec3::Y),
                rotation: rotation_of(model),
                look: emitter.look,
                rate: emitter.rate,
                start,
                stop,
                now,
                count: emitter.capacity(),
                slot,
            },
        })
    }

    fn gather_bursts(&mut self, bursts: &ParticleBursts, eyes: &[Vec3], time: f32) -> u64 {
        for mut burst in bursts.take_bursts() {
            self.bursts_taken = self.bursts_taken.wrapping_add(1);
            burst.look.seed = burst.look.seed.wrapping_add(self.bursts_taken);
            self.bursts.push((burst, time));
        }
        self.bursts.retain(|(burst, start)| {
            let age = time - *start;
            burst.count > 0 && age >= 0.0 && age < burst.look.lifetime
        });
        let mut wanted = 0_u64;
        for (burst, start) in &self.bursts {
            let count = burst.count.min(MAX_EMITTER_PARTICLES);
            wanted += u64::from(count);
            self.candidates.push(Candidate {
                distance: closest(burst.at, eyes),
                entity: None,
                emitter: RenderEmitter {
                    origin: burst.at,
                    velocity: Vec3::ZERO,
                    direction: burst.look.direction.try_normalize().unwrap_or(Vec3::Y),
                    rotation: Quat::IDENTITY,
                    look: burst.look,
                    rate: 0.0,
                    start: *start,
                    stop: f32::MAX,
                    now: time,
                    count,
                    slot: NO_SLOT,
                },
            });
        }
        wanted
    }

    fn fill(
        &mut self,
        settings: &ParticleSettings,
        (emitters, table): (&Query<'_, '_, &'static ParticleEmitter>, &DrawnTable),
        bursts: &ParticleBursts,
        (globals, poses): (&GlobalTransforms, &DrawnPoses),
        eyes: &[Vec3],
        (game, real): (f32, f32),
    ) {
        self.list.clear();
        self.cut = bursts.take_cut() as usize;
        self.candidates.clear();
        self.emissions
            .retain(|entity, _| emitters.contains(*entity) && globals.drawn(*entity));
        let mut wanted = 0_u64;
        for (slot, entity, _) in table.rows(PARTICLES) {
            let Ok(emitter) = emitters.get(entity) else {
                continue;
            };
            let Some(word) = globals.word(entity).filter(|word| word.drawn()) else {
                continue;
            };
            let Some(model) = poses.drawn_pose(globals, entity) else {
                continue;
            };
            let now = match word.clock {
                Clock::Game => game,
                Clock::Real => real,
            };
            if let Some(candidate) = self.steady((entity, slot, emitter), model, eyes, now) {
                wanted += u64::from(candidate.emitter.count);
                self.candidates.push(candidate);
            }
        }
        wanted += self.gather_bursts(bursts, eyes, game);
        let budget = settings.max_particles.map(u64::from);
        let mut total = 0_u64;
        let mut steady = 0_usize;
        self.extraction += 1;
        let extraction = self.extraction;
        let Self {
            motion,
            candidates,
            nearest,
            list,
            cut,
            ..
        } = self;
        let candidates = &*candidates;
        let mut emit = |position: usize| {
            let Candidate {
                entity,
                mut emitter,
                ..
            } = candidates[position];
            let count = emitter.count;
            if budget.is_some_and(|budget| total + u64::from(count) > budget) {
                *cut += count as usize;
                return;
            }
            total += u64::from(count);
            if let Some(entity) = entity {
                let origin = emitter.origin;
                let velocity = match motion.get(&entity) {
                    Some(&(last, at, velocity, _)) => {
                        let elapsed = emitter.now - at;
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
                motion.insert(entity, (origin, emitter.now, velocity, extraction));
                steady += 1;
                if velocity.is_finite() {
                    emitter.velocity = velocity;
                }
            }
            list.push(emitter);
        };
        if budget.is_some_and(|budget| wanted > budget) {
            nearest
                .nearest_first(
                    candidates.iter().map(|candidate| candidate.distance),
                    usize::MAX,
                )
                .for_each(&mut emit);
        } else {
            (0..candidates.len()).for_each(&mut emit);
        }
        if motion.len() > steady {
            motion.retain(|_, motion| motion.3 == extraction);
        }
    }
}

pub const EMITTER_PART: &str = "particles";

type JudgedEmitters = (
    With<ParticleEmitter>,
    Or<(Added<ParticleEmitter>, Changed<Drawn>)>,
);

pub(crate) fn judge_emitters(
    emitters: Query<(Entity, &Drawn), JudgedEmitters>,
    mut refusals: ResMut<Refusals>,
) {
    for (entity, drawn) in &emitters {
        let verdict = if drawn.kind == PARTICLES {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::UsageConflict, "$.particles")
                .got(drawn.kind.name())
                .nearest(PARTICLES.name()))
        };
        refusals.judge(entity, EMITTER_PART, verdict);
    }
}

pub(crate) fn forget_emitter(In(entity): In<Entity>, mut refusals: ResMut<Refusals>) {
    refusals.accept(entity, EMITTER_PART);
}

fn closest(point: Vec3, eyes: &[Vec3]) -> f32 {
    eyes.iter()
        .map(|eye| point.distance_squared(*eye))
        .fold(f32::INFINITY, engine_core::math::min)
}

type GatherFrame<'w> = (
    Res<'w, GlobalTransforms>,
    Res<'w, DrawnPoses>,
    Res<'w, Views>,
    Res<'w, FrameTime>,
    Res<'w, ParticleBursts>,
);

pub(crate) fn gather_emitters(
    settings: Res<ParticleSettings>,
    (emitters, table): (Query<&'static ParticleEmitter>, Res<DrawnTable>),
    (globals, poses, views, time, bursts): GatherFrame<'_>,
    mut gather: ResMut<Gather>,
    mut packed: ResMut<Packed>,
) {
    let gather = &mut *gather;
    let eyes: Vec<Vec3> = views
        .listed()
        .iter()
        .map(|facts| facts.detail_eye)
        .collect();
    gather.fill(
        &settings,
        (&emitters, &table),
        &bursts,
        (&globals, &poses),
        &eyes,
        (time.elapsed as f32, time.real_elapsed as f32),
    );
    let packed = &mut *packed;
    (packed.alpha, packed.mist, packed.total, packed.cut) =
        pack_emitters(&gather.list, &mut packed.records);
    packed.emits = !gather.list.is_empty();
}

pub(crate) fn particle_stats(
    gather: Res<Gather>,
    packed: Res<Packed>,
    mut stats: ResMut<ParticleStats>,
) {
    let frame = (packed.emits && packed.total > 0).then_some(&*packed);
    let drawn = frame.map_or(0, |frame| frame.total);
    let cut =
        (gather.cut as u64 + frame.map_or(0, |frame| frame.cut)).min(u64::from(u32::MAX)) as u32;
    stats.set_if_neq(ParticleStats {
        drawn,
        particle_cut: Cut::of(drawn.saturating_add(cut), drawn),
    });
}

fn pack_emitters(emitters: &[RenderEmitter], records: &mut Vec<Emitter>) -> (u32, u32, u32, u64) {
    records.clear();
    let mut first = 0_u32;
    let (mut alpha, mut mist) = (0, 0);
    let mut cut = 0;
    for blend in [
        ParticleBlend::Alpha,
        ParticleBlend::Mist,
        ParticleBlend::Additive,
    ] {
        for emitter in emitters
            .iter()
            .filter(|emitter| emitter.look.blend == blend)
        {
            if u64::from(first) + u64::from(emitter.count) > u64::from(u32::MAX) {
                cut += u64::from(emitter.count);
                continue;
            }
            records.push(Emitter::new(emitter, first));
            first += emitter.count;
        }
        match blend {
            ParticleBlend::Alpha => alpha = first,
            ParticleBlend::Mist => mist = first,
            ParticleBlend::Additive => {}
        }
    }
    (alpha, mist, first, cut)
}
