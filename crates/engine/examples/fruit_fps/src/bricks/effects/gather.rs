use engine::math;
use engine::prelude::*;

use super::numbers::{self, EffectsNumbers, Parsed};
use super::{Device, EffectsBrick, EffectsShow, Show};
use crate::abilities::{CueKind, Event as PowerEvent, OrbEvent};
use crate::play::actors::{Actor, Body, Human, Wish};
use crate::play::effects::{CueLook, EffectsFeed, Look, PropPlace};
use crate::play::events::arms::{ArmsEvents, Target, Weapon};
use crate::play::events::powers::PowerEvents;
use crate::play::numbers::Numbers;
use crate::play::puppet::{Muzzles, ViewModel};
use crate::play::round::{DeviceSpot, Round};
use crate::play::view::Watched;
use crate::play::world::Arena;
use crate::rules::Team;
use crate::solids::Kind;

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
    Impact(Option<Kind>),
    Juice([f32; 3]),
    Blast,
    Pieces([f32; 3]),
    Sparks([f32; 3]),
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
    resting: Vec<u32>,
    pub puffs: u32,
    powers_seen: Option<u64>,
    looks: Vec<(u32, Look, Team)>,
    pub vanished: Vec<(usize, f64)>,
    pub shatters: u32,
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

    fn impact(&mut self, at: Vec3, normal: Vec3, born: f64, kind: Option<Kind>) {
        self.spray(at, normal, born, SprayKind::Impact(kind), 1.0);
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

    fn shatter(
        &mut self,
        look: &EffectsNumbers,
        at: Vec3,
        size: f32,
        colors: [[f32; 3]; 2],
        born: f64,
    ) {
        let shatter = &look.shatter;
        let scale = math::clamp(
            size / math::max(shatter.reference_m, 1.0e-3),
            shatter.least_scale,
            shatter.most_scale,
        );
        self.spray(at, Vec3::Y, born, SprayKind::Pieces(colors[0]), scale);
        self.spray(at, Vec3::Y, born, SprayKind::Sparks(colors[1]), scale);
        self.shatters = self.shatters.wrapping_add(1);
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
            spawns.impact(entry, -shot.direction, now, shot.pierced_kind);
        }
        if shot.weapon != Weapon::Melee {
            let muzzle = origins.muzzle(shot.shooter, human);
            spawns.shot(look, shot.origin, muzzle, shot.end, now);
        }
        match shot.target {
            Target::Nothing => {}
            Target::World => {
                let normal = shot.normal.try_normalize().unwrap_or(-shot.direction);
                spawns.impact(shot.end, normal, now, shot.kind);
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

fn broken_colors(
    look: &EffectsNumbers,
    seen: Option<(Look, Team)>,
    (team, viewer): (usize, Option<usize>),
) -> [[f32; 3]; 2] {
    let ally = viewer.is_none_or(|viewer| viewer == team);
    let edge = look.teams[usize::from(!ally)];
    match seen {
        Some((Look::Wall, _)) => [look.panels.rind.flesh, look.panels.rind.skin],
        Some((Look::Curtain, _)) => [look.panels.curtain.peel, look.panels.curtain.pulp],
        Some((Look::Wire, _)) => [look.props.wire_color, edge],
        Some(_) | None => [edge, edge],
    }
}

fn take_powers(
    look: &EffectsNumbers,
    (events, viewer): (&PowerEvents, Option<usize>),
    spawns: &mut Spawns,
    now: f64,
) {
    let mut broken: Option<(Option<(Look, Team)>, usize)> = None;
    for event in &events.events {
        if let PowerEvent::Destroyed { effect, team, .. } = *event {
            let seen = spawns
                .looks
                .iter()
                .find(|(known, _, _)| *known == effect.0)
                .map(|&(_, look, team)| (look, team));
            broken = Some((seen, team));
        } else if let PowerEvent::Cue {
            cue: CueKind::Destroyed,
            at,
            size,
            ..
        } = *event
        {
            let (seen, team) = broken.take().unwrap_or((None, 0));
            let colors = broken_colors(look, seen, (team, viewer));
            spawns.shatter(look, at, size, colors, now);
        }
    }
    for event in &events.orbs {
        match *event {
            OrbEvent::Vanished { orb } => {
                spawns.vanished.retain(|(known, _)| *known != orb);
                spawns.vanished.push((orb, now));
            }
            OrbEvent::Restored { orb } => spawns.vanished.retain(|(known, _)| *known != orb),
            OrbEvent::Started { .. } | OrbEvent::Stopped { .. } | OrbEvent::Taken { .. } => {}
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
    spawns.looks = feed
        .shown
        .iter()
        .map(|shown| (shown.effect, shown.look, shown.team))
        .collect();
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
    let mut resting = Vec::new();
    for prop in &feed.props {
        if let PropPlace::Thrown {
            centre,
            resting: true,
            ..
        } = prop.place
        {
            if !spawns.resting.contains(&prop.effect) {
                let kind = SprayKind::Impact(Some(Kind::named("dirt")));
                spawns.spray(centre, Vec3::Y, now, kind, look.props.puff_scale);
                spawns.puffs = spawns.puffs.wrapping_add(1);
            }
            resting.push(prop.effect);
        }
    }
    spawns.resting = resting;
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
    let every = if show == Show::Shatter {
        math::min(
            look.showcase.every_seconds,
            look.shatter.pieces.seconds * 0.8,
        )
    } else {
        look.showcase.every_seconds
    };
    if now - spawns.showcase_at < f64::from(every) {
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
        let (end, normal, kind) = match found {
            Some(hit) => (hit.point, hit.normal, Some(hit.kind)),
            None => (ahead, -toward, None),
        };
        let names: Vec<&String> = look.impacts.kinds.keys().collect();
        let kind = if show == Show::Impacts && !names.is_empty() {
            Some(Kind::named(names[step as usize % names.len()]))
        } else {
            kind
        };
        spawns.shot(look, body.eye, None, end, now);
        spawns.impact(end, normal, now, kind);
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
    if show == Show::Shatter {
        let cases = [
            (Look::Wire, look.traps.wire_width_m * 5.0),
            (Look::Trap, look.traps.size_m),
            (Look::Wall, look.showcase.wall_m[0] * 0.5),
        ];
        let (seen, size) = cases[step as usize % cases.len()];
        let team = (step as usize / cases.len()) % 2;
        let colors = broken_colors(look, Some((seen, Team::BOTH[team])), (team, Some(0)));
        spawns.shatter(look, ahead, size, colors, now);
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
    (feed, powers): (Option<Res<EffectsFeed>>, Option<Res<PowerEvents>>),
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
            watched: watched.as_deref().and_then(|watched| watched.slot),
        };
        take_arms(look, events, &cast, origins, &mut spawns, now);
    }
    if let Some(events) = powers.as_deref()
        && spawns.powers_seen != Some(events.tick)
    {
        spawns.powers_seen = Some(events.tick);
        let slot = watched.as_deref().and_then(|watched| watched.slot);
        let slot = slot.or(cast.human.map(|(slot, ..)| slot));
        let viewer = actors
            .iter()
            .find(|(actor, ..)| Some(actor.slot) == slot)
            .map(|(actor, ..)| actor.team.index());
        take_powers(look, (events, viewer), &mut spawns, now);
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
