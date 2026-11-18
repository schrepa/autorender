use glam::Vec3;

use super::{Loaded, Reach};
use crate::bots::{
    AbilitySense, Block, DeviceSense, GamePhase, GunClass, GunSense, Heard, MatchSense, Places,
    Seen, Side as BotSide, SoundKind, WeaponChoice,
};
use crate::play::actors::Motion;
use crate::play::arms::Arms;
use crate::play::events::arms::{ArmsEvents, Handling};
use crate::play::events::body::BodyEvents;
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::powers::Powers;
use crate::play::round::{DeviceSpot, Round};
use crate::play::world::Arena;
use crate::rules::{Device, Event, Phase, Side, Team, View};
use crate::weapons::{GunKind, Slot, Stance, WeaponNumbers, forward};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Who {
    pub slot: u32,
    pub team: Team,
    pub alive: bool,
    pub human: bool,
    pub feet: Vec3,
    pub velocity: Vec3,
    pub eye: Vec3,
    pub crouched: bool,
    pub grounded: bool,
    pub pickable: Option<u32>,
    pub yaw: f32,
    pub pitch: f32,
    pub recoil_yaw: f32,
    pub recoil_pitch: f32,
    pub health: f32,
    pub armour: f32,
    pub gun: GunSense,
    pub character: Option<usize>,
    pub interact: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Noise {
    pub slot: u32,
    pub kind: SoundKind,
    pub at: Vec3,
    pub range: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ReadTicks {
    body: Option<u64>,
    arms: Option<u64>,
    powers: Option<u64>,
    round: Option<u64>,
}

pub(super) struct Events<'a> {
    pub body: Option<&'a BodyEvents>,
    pub arms: Option<&'a ArmsEvents>,
    pub powers: Option<&'a PowerEvents>,
    pub round: Option<&'a RoundEvents>,
}

fn stance(motion: Motion) -> Stance {
    match motion {
        Motion::Standing => Stance::Stand,
        Motion::Walking => Stance::Walk,
        Motion::Running => Stance::Run,
        Motion::Crouching => Stance::Crouch,
        Motion::Air => Stance::Air,
    }
}

fn class_of(kind: GunKind) -> GunClass {
    match kind {
        GunKind::Sidearm => GunClass::Sidearm,
        GunKind::Smg => GunClass::Smg,
        GunKind::Rifle => GunClass::Rifle,
        GunKind::Sniper => GunClass::Sniper,
    }
}

fn holding(slot: Slot) -> WeaponChoice {
    match slot {
        Slot::Primary => WeaponChoice::Primary,
        Slot::Sidearm => WeaponChoice::Sidearm,
        Slot::Melee => WeaponChoice::Melee,
    }
}

pub(super) fn gun_sense(
    arms: Option<&Arms>,
    weapons: &WeaponNumbers,
    motion: Motion,
    recoil: (f32, f32),
) -> GunSense {
    let Some(arms) = arms else {
        let sidearm = weapons.gun(GunKind::Sidearm);
        return GunSense {
            class: GunClass::Sidearm,
            auto: sidearm.auto,
            magazine: 0,
            magazine_size: sidearm.magazine,
            reserve: 0,
            reloading: false,
            ready: false,
            spread_deg: 0.0,
            scope_level: 0,
            recoil_pitch: recoil.0,
            recoil_yaw: recoil.1,
            range_m: sidearm.range_m(),
            has_primary: false,
            holding: WeaponChoice::Sidearm,
            has_scope: sidearm.scope.is_some(),
        };
    };
    let loadout = &arms.loadout;
    let view = loadout.view(weapons, stance(motion));
    let gun = loadout.active_gun().map(|gun| weapons.gun(gun.kind()));
    GunSense {
        class: loadout
            .active_gun()
            .map_or(GunClass::Melee, |gun| class_of(gun.kind())),
        auto: gun.is_some_and(|gun| gun.auto),
        magazine: view.magazine.unwrap_or(0),
        magazine_size: gun.map_or(0, |gun| gun.magazine),
        reserve: view.reserve.unwrap_or(0),
        reloading: view.reload_progress.is_some(),
        ready: view.ready,
        spread_deg: view.spread_deg,
        scope_level: view.scope_level,
        recoil_pitch: recoil.0,
        recoil_yaw: recoil.1,
        range_m: gun.map_or(weapons.melee.range_m, |gun| gun.range_m()),
        has_primary: loadout.primary.is_some(),
        holding: holding(loadout.active),
        has_scope: gun.is_some_and(|gun| gun.scope.is_some()),
    }
}

pub(super) fn abilities_of(
    powers: Option<&Powers>,
    slot: u32,
    reach: [Reach; 4],
) -> [AbilitySense; 4] {
    let actor = powers.and_then(|powers| powers.actor(slot));
    [0, 1, 2, 3].map(|index| {
        let Some(view) = actor.map(|actor| &actor.slots[index]) else {
            return AbilitySense {
                block: None,
                ready: false,
                charges: 0,
                range_m: reach[index].range_m,
                radius_m: reach[index].radius_m,
                thrown: reach[index].thrown,
            };
        };
        let ultimate = index == 3;
        let charges = if ultimate {
            u32::from(view.ready)
        } else {
            view.charges
        };
        AbilitySense {
            block: Block::from_data_name(&view.block),
            ready: view.ready && charges > 0,
            charges,
            range_m: reach[index].range_m,
            radius_m: reach[index].radius_m,
            thrown: reach[index].thrown,
        }
    })
}

fn points(who: &Who, loaded: &Loaded) -> [Vec3; 3] {
    let share = (who.eye.y - who.feet.y) / loaded.eye_height_m;
    let joints = loaded.joints;
    [
        who.feet + Vec3::Y * (joints.head * share),
        who.feet + Vec3::Y * (joints.spine * share),
        who.feet + Vec3::Y * loaded.leg_radius_m,
    ]
}

pub(super) struct Sight<'a> {
    pub arena: &'a Arena,
    pub powers: Option<&'a Powers>,
    pub loaded: &'a Loaded,
}

pub(super) fn seen(
    sight: &Sight<'_>,
    viewer: &Who,
    target: &Who,
    carrier: Option<u32>,
) -> Option<Seen> {
    let senses = sight.loaded.senses;
    let view = forward(
        viewer.yaw + viewer.recoil_yaw,
        viewer.pitch + viewer.recoil_pitch,
    );
    let limit = sight
        .powers
        .and_then(|powers| powers.sight_limit(viewer.slot));
    let blind = sight
        .powers
        .map_or(0.0, |powers| powers.blindness(viewer.slot));
    let marks = points(target, sight.loaded);
    let mut shown = 0u32;
    for mark in marks {
        let to = mark - viewer.eye;
        let distance = to.length();
        if limit.is_some_and(|limit| distance > limit) {
            continue;
        }
        let ahead = distance <= senses.touch_m || to.dot(view) >= distance * sight.loaded.view_cos;
        let clear = ahead
            && sight.arena.sees(viewer.eye, mark)
            && !sight
                .powers
                .is_some_and(|powers| powers.sight_blocked(viewer.eye, mark));
        shown += u32::from(clear);
    }
    let shows = shown as f32 / marks.len() as f32 * (1.0 - blind);
    (shows > 0.0).then_some(Seen {
        id: target.slot,
        feet: target.feet,
        head: marks[0],
        chest: marks[1],
        velocity: target.velocity,
        yaw: target.yaw,
        shows,
        carries_device: carrier == Some(target.slot),
    })
}

fn fresh(read: &mut Option<u64>, tick: Option<u64>) -> bool {
    let fresh = tick.is_some() && *read != tick;
    if fresh {
        *read = tick;
    }
    fresh
}

fn feet_of(whos: &[Who], slot: u32) -> Option<Vec3> {
    whos.iter().find(|who| who.slot == slot).map(|who| who.feet)
}

pub(super) fn noises(
    events: &Events<'_>,
    read: &mut ReadTicks,
    whos: &[Who],
    loaded: &Loaded,
    out: &mut Vec<Noise>,
) {
    out.clear();
    let senses = loaded.senses;
    let step_range = loaded.bots.player.footstep_heard_m;
    if let Some(body) = events
        .body
        .filter(|body| fresh(&mut read.body, Some(body.tick)))
    {
        out.extend(
            body.footsteps
                .iter()
                .filter(|step| step.loud)
                .map(|step| Noise {
                    slot: step.slot,
                    kind: SoundKind::Step,
                    at: step.position,
                    range: step_range,
                }),
        );
    }
    if let Some(arms) = events
        .arms
        .filter(|arms| fresh(&mut read.arms, Some(arms.tick)))
    {
        out.extend(arms.shots.iter().map(|shot| Noise {
            slot: shot.shooter,
            kind: SoundKind::Shot,
            at: shot.origin,
            range: senses.shot_heard_m,
        }));
        for handling in arms
            .handling
            .iter()
            .filter(|h| h.kind == Handling::ReloadStarted)
        {
            if let Some(at) = feet_of(whos, handling.slot) {
                out.push(Noise {
                    slot: handling.slot,
                    kind: SoundKind::Step,
                    at,
                    range: senses.reload_heard_m,
                });
            }
        }
    }
    if let Some(powers) = events
        .powers
        .filter(|powers| fresh(&mut read.powers, Some(powers.tick)))
    {
        for cast in &powers.casts {
            if let Some(at) = feet_of(whos, cast.slot) {
                out.push(Noise {
                    slot: cast.slot,
                    kind: SoundKind::Ability,
                    at,
                    range: senses.ability_heard_m,
                });
            }
        }
    }
    if let Some(round) = events
        .round
        .filter(|round| fresh(&mut read.round, Some(round.tick)))
    {
        for event in &round.events {
            let (player, kind) = if let Event::PlantStarted { player, .. } = event {
                (*player, SoundKind::Plant)
            } else if let Event::DefuseStarted { player, .. } = event {
                (*player, SoundKind::Defuse)
            } else {
                continue;
            };
            let slot = player as u32;
            if let Some(at) = feet_of(whos, slot) {
                out.push(Noise {
                    slot,
                    kind,
                    at,
                    range: senses.device_heard_m,
                });
            }
        }
    }
}

pub(super) fn heard(noises: &[Noise], whos: &[Who], hearer: &Who, out: &mut Vec<Heard>) {
    for noise in noises {
        let enemy = whos
            .iter()
            .any(|who| who.slot == noise.slot && who.team != hearer.team);
        let distance = (noise.at - hearer.feet).length();
        if enemy && distance <= noise.range && noise.range > 0.0 {
            out.push(Heard {
                kind: noise.kind,
                at: noise.at,
                loudness: 1.0 - distance / noise.range,
            });
        }
    }
}

pub(super) fn revealed(powers: Option<&Powers>, whos: &[Who], hearer: &Who, out: &mut Vec<Heard>) {
    let Some(powers) = powers else {
        return;
    };
    for who in whos {
        if who.alive && who.team != hearer.team && powers.revealed(who.slot, hearer.team) {
            out.push(Heard {
                kind: SoundKind::Ability,
                at: who.feet,
                loudness: 1.0,
            });
        }
    }
}

pub(super) fn carrier(round: &Round) -> Option<u32> {
    match round.device {
        DeviceSpot::Carried { slot, .. } => Some(slot as u32),
        DeviceSpot::Nowhere
        | DeviceSpot::Dropped { .. }
        | DeviceSpot::Planted { .. }
        | DeviceSpot::Exploded { .. }
        | DeviceSpot::Defused { .. } => None,
    }
}

fn device_for(round: &Round, view: &View<'_>, places: &Places, side: Side) -> DeviceSense {
    let attacking = side == Side::Attack;
    match round.device {
        DeviceSpot::Carried { slot, .. } if attacking => DeviceSense::Carried {
            carrier: slot as u32,
        },
        DeviceSpot::Dropped { at } if attacking => DeviceSense::Dropped { at },
        DeviceSpot::Planted {
            site,
            at,
            fuse_ticks_left,
            ..
        } => {
            let defuser = match *view.device {
                Device::Planted { defusing, .. } => defusing.player.map(|player| player as u32),
                Device::Carried { .. }
                | Device::Dropped { .. }
                | Device::Exploded { .. }
                | Device::Defused { .. } => None,
            };
            DeviceSense::Planted {
                site: places.site_at(at).unwrap_or(site),
                at,
                fuse_ticks_left,
                defuser,
            }
        }
        DeviceSpot::Exploded { .. } | DeviceSpot::Defused { .. } => DeviceSense::Over,
        DeviceSpot::Nowhere | DeviceSpot::Carried { .. } | DeviceSpot::Dropped { .. } => {
            DeviceSense::Unknown
        }
    }
}

pub(super) fn match_sense(
    round: &Round,
    view: &View<'_>,
    places: &Places,
    side: Side,
) -> MatchSense {
    let phase = match view.phase {
        Phase::Buy => GamePhase::Buy,
        Phase::Live => GamePhase::Live,
        Phase::Planted => GamePhase::Planted,
        Phase::RoundEnd | Phase::Over => GamePhase::Ended,
    };
    MatchSense {
        phase,
        side: match side {
            Side::Attack => BotSide::Attack,
            Side::Defend => BotSide::Defend,
        },
        ticks_left: view.ticks_left,
        round: view.round,
        device: device_for(round, view, places, side),
    }
}
