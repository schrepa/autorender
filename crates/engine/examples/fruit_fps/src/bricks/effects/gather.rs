use engine::math;
use engine::prelude::*;

use super::numbers::{self, EffectsNumbers, Parsed};
use super::{Device, EffectsBrick, EffectsShow, Show};
use crate::play::actors::{Actor, Body, Human, Wish};
use crate::play::effects::{CueLook, EffectsFeed};
use crate::play::events::arms::{ArmsEvents, Target, Weapon};
use crate::play::numbers::Numbers;
use crate::play::puppet::{Muzzles, ViewModel};
use crate::play::round::{DeviceSpot, Round};
use crate::play::view::Watched;
use crate::play::world::Arena;
use crate::solids::Surface;

#[derive(Clone, Debug)]
pub(super) struct Pool<T> {
    items: Vec<T>,
    next: usize,
    capacity: usize,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            next: 0,
            capacity: 0,
        }
    }
}

impl<T> Pool<T> {
    fn hold(&mut self, capacity: u32) {
        let capacity = capacity as usize;
        if capacity != self.capacity {
            self.items.clear();
            self.next = 0;
            self.capacity = capacity;
        }
    }

    fn push(&mut self, item: T) {
        if self.capacity == 0 {
            return;
        }
        if self.items.len() < self.capacity {
            self.items.push(item);
        } else {
            self.items[self.next] = item;
        }
        self.next = (self.next + 1) % self.capacity;
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Tracer {
    pub from: Vec3,
    pub to: Vec3,
    pub born: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Flash {
    pub at: Vec3,
    pub toward: Vec3,
    pub born: f64,
    pub seed: u32,
    pub scale: f32,
    pub seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum SprayKind {
    Impact(Surface),
    Juice([f32; 3]),
    Blast,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Burst {
    pub at: Vec3,
    pub normal: Vec3,
    pub born: f64,
    pub seed: u32,
    pub kind: SprayKind,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Decal {
    pub at: Vec3,
    pub normal: Vec3,
    pub born: f64,
    pub seed: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Ring {
    pub at: Vec3,
    pub radius: f32,
    pub born: f64,
    pub seconds: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Hurt {
    pub level: f32,
    pub at: f64,
}

impl Hurt {
    pub fn now(&self, now: f64, seconds: f32) -> f32 {
        let fade = 1.0 - ((now - self.at) as f32) / seconds;
        math::clamp(self.level * fade, 0.0, 1.0)
    }
}

#[derive(Resource, Default)]
pub(super) struct Setup {
    pub revision: Option<u64>,
    pub parsed: Option<Parsed>,
    pub refusal: Option<String>,
}

#[derive(Resource, Default)]
pub(super) struct Spawns {
    pub tracers: Pool<Tracer>,
    pub flashes: Pool<Flash>,
    pub sprays: Pool<Burst>,
    pub decals: Pool<Decal>,
    pub rings: Pool<Ring>,
    pub hurt: Hurt,
    pub showcase_at: f64,
    pub showcase_step: u32,
    seed: u32,
    arms_seen: Option<u64>,
    feed_seen: Option<u64>,
    device_was: DeviceSpot,
}

impl Spawns {
    fn seed(&mut self) -> u32 {
        self.seed = self.seed.wrapping_add(0x9e37_79b9);
        self.seed
    }

    fn hold(&mut self, look: &EffectsNumbers) {
        self.tracers.hold(look.pools.tracers);
        self.flashes.hold(look.pools.flashes);
        self.sprays.hold(look.pools.sprays);
        self.decals.hold(look.pools.decals);
        self.rings.hold(look.pools.rings);
    }

    fn shot(
        &mut self,
        look: &EffectsNumbers,
        from: Vec3,
        muzzle: Option<Vec3>,
        to: Vec3,
        born: f64,
    ) {
        let toward = (to - from).normalize_or_zero();
        let muzzle = muzzle.unwrap_or(from + muzzle_offset(toward, look.tracer.muzzle_offset_m));
        self.tracers.push(Tracer {
            from: muzzle,
            to,
            born,
        });
        let seed = self.seed();
        self.flashes.push(Flash {
            at: muzzle,
            toward,
            born,
            seed,
            scale: 1.0,
            seconds: look.flash.seconds,
        });
    }

    fn spray(&mut self, at: Vec3, normal: Vec3, born: f64, kind: SprayKind, scale: f32) {
        let seed = self.seed();
        self.sprays.push(Burst {
            at,
            normal,
            born,
            seed,
            kind,
            scale,
        });
    }

    fn impact(&mut self, at: Vec3, normal: Vec3, born: f64, surface: Surface) {
        self.spray(at, normal, born, SprayKind::Impact(surface), 1.0);
        let seed = self.seed();
        self.decals.push(Decal {
            at,
            normal,
            born,
            seed,
        });
    }

    fn blast(&mut self, look: &EffectsNumbers, at: Vec3, born: f64) {
        self.spray(at, Vec3::Y, born, SprayKind::Blast, 1.0);
        let seed = self.seed();
        self.flashes.push(Flash {
            at,
            toward: Vec3::Y,
            born,
            seed,
            scale: look.blast.flash_scale,
            seconds: look.blast.spray.seconds,
        });
    }

    fn hurt(&mut self, look: &EffectsNumbers, amount: f32, now: f64) {
        let level = self.hurt.now(now, look.screen.damage_seconds);
        self.hurt = Hurt {
            level: level + amount / look.screen.damage_full,
            at: now,
        };
    }
}

pub(super) fn muzzle_offset(toward: Vec3, offset: [f32; 3]) -> Vec3 {
    let right = toward.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    let up = right.cross(toward);
    right * offset[0] + up * offset[1] + toward * offset[2]
}

pub(super) fn view_direction(wish: &Wish) -> Vec3 {
    let (sin_yaw, cos_yaw) = math::sin_cos(wish.yaw_radians);
    let (sin_pitch, cos_pitch) = math::sin_cos(wish.pitch_radians);
    Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
}

pub(super) fn setup(numbers: Res<Numbers>, mut setup: ResMut<Setup>) {
    let Some(game) = numbers.get() else {
        return;
    };
    if setup.revision == Some(numbers.revision()) {
        return;
    }
    setup.revision = Some(numbers.revision());
    match numbers::parse(&game.text) {
        Ok(parsed) => {
            setup.parsed = Some(parsed);
            setup.refusal = None;
        }
        Err(error) => {
            setup.parsed = None;
            setup.refusal = Some(error);
        }
    }
}

type ActorRow<'a> = (&'a Actor, &'a Body, Has<Human>, Option<&'a Wish>);

struct Cast<'a> {
    characters: Vec<(u32, &'a str, Vec3)>,
    human: Option<(u32, &'a Body, Option<&'a Wish>)>,
}

impl<'a> Cast<'a> {
    fn of(actors: &'a Query<'_, '_, ActorRow<'_>>) -> Cast<'a> {
        let mut characters = Vec::new();
        let mut human = None;
        for (actor, body, is_human, wish) in actors.iter() {
            characters.push((
                actor.slot,
                actor.character.as_str(),
                (body.feet + body.eye) * 0.5,
            ));
            if is_human {
                human = Some((actor.slot, body, wish));
            }
        }
        Cast { characters, human }
    }

    fn character(&self, slot: u32) -> Option<(&'a str, Vec3)> {
        self.characters
            .iter()
            .find(|(known, _, _)| *known == slot)
            .map(|(_, character, middle)| (*character, *middle))
    }
}

fn juice(look: &EffectsNumbers, character: &str) -> Option<SprayKind> {
    look.juice
        .colors
        .get(character)
        .copied()
        .map(SprayKind::Juice)
}

#[derive(Clone, Copy)]
struct Origins<'a> {
    muzzles: Option<&'a Muzzles>,
    view: Option<&'a ViewModel>,
    watched: Option<u32>,
}

impl Origins<'_> {
    fn muzzle(&self, slot: u32, human: Option<u32>) -> Option<Vec3> {
        if human == Some(slot) {
            return self.view.and_then(|view| view.muzzle);
        }
        self.muzzles
            .and_then(|muzzles| muzzles.of(slot))
            .map(|muzzle| muzzle.position)
    }
}

fn take_arms(
    look: &EffectsNumbers,
    events: &ArmsEvents,
    cast: &Cast<'_>,
    origins: Origins<'_>,
    spawns: &mut Spawns,
    now: f64,
) {
    let human = cast.human.map(|(slot, _, _)| slot);
    for shot in &events.shots {
        if let Some(entry) = shot.pierced {
            spawns.impact(entry, -shot.direction, now, Surface::Stone);
        }
        if shot.weapon != Weapon::Melee {
            let muzzle = origins.muzzle(shot.shooter, human);
            spawns.shot(look, shot.origin, muzzle, shot.end, now);
        }
        match shot.target {
            Target::Nothing => {}
            Target::World { surface } => {
                let normal = shot.normal.try_normalize().unwrap_or(-shot.direction);
                spawns.impact(shot.end, normal, now, surface);
            }
            Target::Actor { slot, .. } => {
                if let Some(kind) = cast.character(slot).and_then(|(name, _)| juice(look, name)) {
                    spawns.spray(shot.end, -shot.direction, now, kind, 1.0);
                }
            }
        }
    }
    for death in &events.deaths {
        if let Some((name, middle)) = cast.character(death.victim)
            && let Some(kind) = juice(look, name)
        {
            spawns.spray(middle, Vec3::Y, now, kind, look.juice.death_scale);
        }
    }
    if let Some(slot) = origins.watched.or(human) {
        for hit in events.hits.iter().filter(|hit| hit.victim == slot) {
            spawns.hurt(look, hit.health_lost + hit.armor_lost, now);
        }
    }
}

fn take_feed(
    look: &EffectsNumbers,
    feed: &EffectsFeed,
    spawns: &mut Spawns,
    now: f64,
    tick_seconds: f32,
) {
    for cue in &feed.cues {
        match cue.look {
            CueLook::Explosion | CueLook::StrikeImpact => spawns.blast(look, cue.at, now),
            CueLook::Flash => {
                let seed = spawns.seed();
                spawns.flashes.push(Flash {
                    at: cue.at,
                    toward: Vec3::Y,
                    born: now,
                    seed,
                    scale: look.blast.flash_scale,
                    seconds: look.blast.spray.seconds,
                });
            }
            CueLook::Pulse => spawns.rings.push(Ring {
                at: cue.at,
                radius: cue.size,
                born: now,
                seconds: cue.ticks as f32 * tick_seconds,
            }),
        }
    }
    if feed.human.damage > 0.0 {
        spawns.hurt(look, feed.human.damage, now);
    }
}

fn showcase(
    look: &EffectsNumbers,
    show: Show,
    cast: &Cast<'_>,
    arena: Option<&Arena>,
    spawns: &mut Spawns,
    now: f64,
) {
    let Some((_, body, Some(wish))) = cast.human else {
        return;
    };
    if now - spawns.showcase_at < f64::from(look.showcase.every_seconds) {
        return;
    }
    spawns.showcase_at = now;
    spawns.showcase_step = spawns.showcase_step.wrapping_add(1);
    let step = spawns.showcase_step;
    let toward = view_direction(wish);
    let ahead = body.eye + toward * look.showcase.distance_m;
    let all = show == Show::All;
    if all || show == Show::Shots || show == Show::Impacts {
        let found = arena
            .filter(|arena| arena.ready())
            .and_then(|arena| arena.solids().ray(body.eye, toward, crate::names::FAR));
        let (end, normal, surface) = match found {
            Some(hit) => (hit.point, hit.normal, hit.surface),
            None => (ahead, -toward, Surface::Dirt),
        };
        let surfaces = [Surface::Stone, Surface::Wood, Surface::Metal, Surface::Dirt];
        let surface = if show == Show::Impacts {
            surfaces[step as usize % surfaces.len()]
        } else {
            surface
        };
        spawns.shot(look, body.eye, None, end, now);
        spawns.impact(end, normal, now, surface);
    }
    if all || show == Show::Juice {
        let colors: Vec<[f32; 3]> = look.juice.colors.values().copied().collect();
        if let Some(color) = colors.get(step as usize % colors.len().max(1)) {
            spawns.spray(ahead, -toward, now, SprayKind::Juice(*color), 1.0);
        }
    }
    let feet = body.feet + Vec3::new(toward.x, 0.0, toward.z) * look.showcase.distance_m;
    if all || show == Show::Pulse {
        spawns.rings.push(Ring {
            at: feet,
            radius: look.showcase.radius_m,
            born: now,
            seconds: look.showcase.every_seconds,
        });
    }
    if show == Show::Strike {
        spawns.blast(look, feet, now);
    }
    if show == Show::Damage {
        spawns.hurt(look, look.screen.damage_full * 0.5, now);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the tick's events come from every producer"
)]
pub(super) fn gather(
    time: Res<FixedTime>,
    setup: Res<Setup>,
    show: Res<EffectsShow>,
    arms: Option<Res<ArmsEvents>>,
    round: Option<Res<Round>>,
    feed: Option<Res<EffectsFeed>>,
    arena: Option<Res<Arena>>,
    muzzles: Option<Res<Muzzles>>,
    watched: Option<Res<Watched>>,
    view: Option<Res<ViewModel>>,
    actors: Query<ActorRow<'_>>,
    mut spawns: ResMut<Spawns>,
) {
    let Some(parsed) = setup.parsed.as_ref() else {
        return;
    };
    let look = &parsed.look;
    spawns.hold(look);
    let now = time.elapsed_seconds;
    let cast = Cast::of(&actors);
    if let Some(events) = arms.as_deref()
        && spawns.arms_seen != Some(events.tick)
    {
        spawns.arms_seen = Some(events.tick);
        let origins = Origins {
            muzzles: muzzles.as_deref(),
            view: view.as_deref(),
            watched: watched.and_then(|watched| watched.slot),
        };
        take_arms(look, events, &cast, origins, &mut spawns, now);
    }
    if let Some(feed) = feed.as_deref()
        && spawns.feed_seen != Some(feed.tick)
    {
        spawns.feed_seen = Some(feed.tick);
        take_feed(look, feed, &mut spawns, now, time.delta_seconds);
    }
    if let Some(round) = round.as_deref() {
        let exploded = |spot: &DeviceSpot| matches!(spot, DeviceSpot::Exploded { .. });
        if exploded(&round.device)
            && !exploded(&spawns.device_was)
            && let Some(at) = round.device.at()
        {
            spawns.blast(look, at, now);
        }
        spawns.device_was = round.device;
    }
    if show.show != Show::Off {
        showcase(look, show.show, &cast, arena.as_deref(), &mut spawns, now);
    }
}

pub(super) fn device_spot(
    round: Option<&Round>,
    show: Show,
    human: Option<(&Body, &Wish)>,
    look: &EffectsNumbers,
) -> Option<(Vec3, Option<u32>)> {
    if show == Show::Device || show == Show::All {
        let (body, wish) = human?;
        let toward = view_direction(wish);
        let flat = Vec3::new(toward.x, 0.0, toward.z).normalize_or_zero();
        return Some((body.feet + flat * look.showcase.distance_m * 0.5, None));
    }
    match round?.device {
        DeviceSpot::Planted {
            at,
            fuse_ticks_left,
            ..
        } => Some((at, Some(fuse_ticks_left))),
        DeviceSpot::Dropped { at } | DeviceSpot::Defused { at, .. } => Some((at, None)),
        DeviceSpot::Nowhere | DeviceSpot::Carried { .. } | DeviceSpot::Exploded { .. } => None,
    }
}

pub(super) fn place_device(
    setup: Res<Setup>,
    show: Res<EffectsShow>,
    round: Option<Res<Round>>,
    humans: Query<(&Body, &Wish), With<Human>>,
    devices: Query<(Entity, &Node), With<Device>>,
    mut moves: Moves<EffectsBrick>,
    mut spawner: Spawner<EffectsBrick>,
) {
    let Some(parsed) = setup.parsed.as_ref() else {
        return;
    };
    let spot = device_spot(
        round.as_deref(),
        show.show,
        humans.iter().next(),
        &parsed.look,
    );
    for (entity, node) in &devices {
        if let Some((at, _)) = spot
            && let Some(mut transform) = moves.get_mut(entity)
        {
            let placed = Transform {
                translation: at,
                ..*transform
            };
            transform.set_if_neq(placed);
        }
        if node.visible != spot.is_some() {
            spawner.insert(
                entity,
                Node {
                    visible: spot.is_some(),
                    ..*node
                },
            );
        }
    }
}
