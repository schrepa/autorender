use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{
    Allowed, Cut, DeviceLimits, Expected, FrameTime, InvalidSetting, Nearest, Poses, Rule, Setting,
    View, fields, gpu_struct, settings, stats,
};
use engine_places::particles::{
    MAX_EMITTER_PARTICLES, MAX_SPREAD_DEGREES, ParticleBlend, ParticleEmitter,
};
use glam::{Mat4, Vec3};

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
        pub invalid: u32,
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
        start_color: [f32; 4],
        end_color: [f32; 4],
        size: [f32; 2],
        emission: f32,
        stretch: f32,
        radius: f32,
        seed: u32,
        first: u32,
        count: u32,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RenderEmitter {
    origin: Vec3,
    velocity: Vec3,
    direction: Vec3,
    emitter: ParticleEmitter,
    count: u32,
}

impl Emitter {
    fn new(emitter: &RenderEmitter, first: u32) -> Self {
        let settings = &emitter.emitter;
        Self {
            origin: emitter.origin.to_array(),
            rate: settings.rate,
            velocity: emitter.velocity.to_array(),
            lifetime: settings.lifetime,
            direction: emitter.direction.to_array(),
            speed: settings.speed,
            acceleration: settings.acceleration.to_array(),
            spread: engine_core::math::cos(engine_core::math::clamp(
                settings.spread,
                0.0,
                std::f32::consts::PI,
            )),
            start_color: settings.color[0],
            end_color: settings.color[1],
            size: settings.size,
            emission: settings.emission,
            stretch: settings.stretch,
            radius: settings.radius,
            seed: settings.seed,
            first,
            count: emitter.count,
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
        "size" | "radius" => Some(fields::range(0.0, 10_000.0)),
        "color" => Some(fields::range(0.0, 1.0)),
        "emission" => Some(fields::range(0.0, 64.0)),
        "stretch" => Some(fields::range(0.0, 10.0)),
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

fn ruled<T: Setting>(value: &T, field: &'static str) -> bool {
    Setting::check(value, field, fields::same(emitter_rule(field))).is_ok()
}

fn valid(emitter: &ParticleEmitter) -> bool {
    ruled(&emitter.rate, "rate")
        && ruled(&emitter.lifetime, "lifetime")
        && ruled(&emitter.speed, "speed")
        && ruled(&emitter.direction, "direction")
        && ruled(&emitter.spread, "spread")
        && ruled(&emitter.acceleration, "acceleration")
        && ruled(&emitter.size, "size")
        && ruled(&emitter.color, "color")
        && ruled(&emitter.emission, "emission")
        && ruled(&emitter.stretch, "stretch")
        && ruled(&emitter.offset, "offset")
        && ruled(&emitter.radius, "radius")
        && emitter_extra(emitter.direction, emitter.rate, emitter.lifetime).is_ok()
}

#[derive(Resource, Default)]
pub(crate) struct Gather {
    motion: HashMap<Entity, (Vec3, f32, Vec3, u64)>,
    extraction: u64,
    candidates: Vec<(f32, Entity, Mat4, ParticleEmitter)>,
    nearest: Nearest,
    list: Vec<RenderEmitter>,
    previous: Vec<RenderEmitter>,
    records: Vec<Emitter>,
    cut: usize,
    invalid: usize,
    broken: Vec<Entity>,
}

#[derive(Resource, Default)]
pub(crate) struct Packed {
    pub(crate) records: Vec<Emitter>,
    pub(crate) alpha: u32,
    pub(crate) mist: u32,
    pub(crate) total: u32,
    pub(crate) emits: bool,
    pub(crate) revision: u64,
    pub(crate) epoch: u64,
    cut: u64,
    room: u64,
}

#[derive(Resource, Default)]
pub(crate) struct Owners {
    pub(crate) owners: Vec<u32>,
}

impl Gather {
    fn fill(
        &mut self,
        settings: &ParticleSettings,
        emitters: &Query<'_, '_, (Entity, &'static ParticleEmitter)>,
        poses: &Poses,
        eye: Vec3,
        time: f32,
    ) {
        self.list.clear();
        self.cut = 0;
        self.invalid = 0;
        self.candidates.clear();
        let mut wanted = 0_u64;
        for (entity, emitter) in emitters.iter() {
            if self.broken.contains(&entity) {
                self.invalid += 1;
                continue;
            }
            let Some(model) = poses.of(entity).map(Mat4::from) else {
                continue;
            };
            wanted += u64::from(emitter.capacity());
            let distance = model.transform_point3(emitter.offset).distance_squared(eye);
            self.candidates.push((distance, entity, model, *emitter));
        }
        let budget = settings.max_particles.map(u64::from);
        let mut total = 0_u64;
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
            let (_, entity, model, emitter) = candidates[position];
            let count = emitter.capacity();
            if budget.is_some_and(|budget| total + u64::from(count) > budget) {
                *cut += count as usize;
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
            list.push(RenderEmitter {
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
            nearest
                .nearest_first(candidates.iter().map(|candidate| candidate.0), usize::MAX)
                .for_each(&mut emit);
        } else {
            (0..candidates.len()).for_each(&mut emit);
        }
        if motion.len() > list.len() {
            motion.retain(|_, motion| motion.3 == extraction);
        }
    }
}

pub(crate) fn gather_emitters(
    settings: Res<ParticleSettings>,
    emitters: Query<(Entity, &'static ParticleEmitter)>,
    changed: Query<(Entity, &'static ParticleEmitter), Changed<ParticleEmitter>>,
    (poses, view, time, limits): (Res<Poses>, Res<View>, Res<FrameTime>, Res<DeviceLimits>),
    mut gather: ResMut<Gather>,
    mut packed: ResMut<Packed>,
    mut owners: ResMut<Owners>,
) {
    let gather = &mut *gather;
    for (entity, emitter) in &changed {
        gather.broken.retain(|&broken| broken != entity);
        if !valid(emitter) {
            gather.broken.push(entity);
        }
    }
    gather.fill(
        &settings,
        &emitters,
        &poses,
        view.detail_eye(),
        time.elapsed_seconds as f32,
    );
    let room = limits.storage_room / 16 * 16;
    if gather.list == gather.previous && packed.room == room {
        return;
    }
    gather.previous.clone_from(&gather.list);
    let (alpha, mist, total, cut) = pack_emitters(&gather.list, &mut gather.records, room / 4);
    if !same_instances(&gather.records, &packed.records) {
        pack_owners(&gather.records, &mut owners.owners);
    }
    let packed = &mut *packed;
    packed.records.clone_from(&gather.records);
    (packed.alpha, packed.mist, packed.total, packed.cut) = (alpha, mist, total, cut);
    packed.emits = !gather.list.is_empty();
    packed.revision = packed.revision.wrapping_add(1);
    packed.room = room;
}

pub(crate) fn particle_stats(
    gather: Res<Gather>,
    packed: Res<Packed>,
    view: Res<View>,
    mut stats: ResMut<ParticleStats>,
) {
    let frame = (!view.debug() && packed.emits && packed.total > 0).then_some(&*packed);
    let drawn = frame.map_or(0, |frame| frame.total);
    let cut = (gather.cut as u64 + frame.map_or(0, |frame| frame.cut))
        .min(u64::from(u32::MAX)) as u32;
    stats.set_if_neq(ParticleStats {
        drawn,
        invalid: gather.invalid.min(u32::MAX as usize) as u32,
        particle_cut: Cut::of(drawn.saturating_add(cut), drawn),
    });
}

pub(crate) fn forget(
    mut gather: ResMut<Gather>,
    mut packed: ResMut<Packed>,
    mut owners: ResMut<Owners>,
    mut stats: ResMut<ParticleStats>,
) {
    let epoch = packed.epoch.wrapping_add(1);
    *gather = Gather {
        broken: std::mem::take(&mut gather.broken),
        ..Gather::default()
    };
    *packed = Packed {
        epoch,
        ..Packed::default()
    };
    *owners = Owners::default();
    stats.set_if_neq(ParticleStats::default());
}

fn pack_emitters(
    emitters: &[RenderEmitter],
    records: &mut Vec<Emitter>,
    room: u64,
) -> (u32, u32, u32, u64) {
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
            .filter(|emitter| emitter.emitter.blend == blend)
        {
            if u64::from(first) + u64::from(emitter.count) > room {
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

fn pack_owners(records: &[Emitter], owners: &mut Vec<u32>) {
    owners.clear();
    for (index, record) in records.iter().enumerate() {
        owners.extend(std::iter::repeat_n(index as u32, record.count as usize));
    }
}

pub(crate) fn same_instances(a: &[Emitter], b: &[Emitter]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.first == b.first && a.count == b.count)
}
