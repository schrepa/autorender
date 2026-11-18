use std::collections::{BTreeMap, BTreeSet};

use engine::prelude::*;
use engine_core::math::{clamp, max, min, sin_cos};
use engine_core::{Error, ErrorCode, Refusals};

use super::locomotion::{Moved, moved};
use super::{
    ActorStats, AimStats, ClipWeight, EyeFitStats, HeadStats, LocomotionStats, Memories,
    PUPPET_PART, PuppetBrick, PuppetStats,
};
use crate::abilities::{AbilityBook, AbilityId, AbilitySpec, Delivery, Slot};
use crate::names::{HEAD, LEFT_HAND, RIGHT_HAND, SPINE};
use crate::play::actors::{Actor, Alive, Body, Motion, Wish};
use crate::play::arms::Arms;
use crate::play::camera::TestCamera;
use crate::play::events::arms::{ArmsEvents, Handling, Weapon};
use crate::play::events::body::BodyEvents;
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::figures::{
    Aim, BaseLayer, Figure, FigureJoints, FigurePlay, HIP, HandTarget, Hold, HoldSpace, LEFT_LEG,
    Offset, RIGHT_LEG, UpperLayer,
};
use crate::play::numbers::{GameNumbers, Numbers};
use crate::play::powers::Powers;
use crate::play::puppet::{
    BaseClip, DEVICE, InHand, OneShot, PuppetBook, PuppetFile, PuppetTable, Task, UpperTable,
    ability_index, leave_tick, swing_clip, weapon_key,
};
use crate::play::view::{ViewOffsets, Watched};
use crate::rules::{AbilitySlot, Event};
use crate::weapons::BodyPart;
use crate::weapons::Stance;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base {
    Idle,
    Walk,
    Run,
    Turn,
    Rise,
    Fall,
    CrouchIdle,
    CrouchWalk,
    Kneel(Task),
    Busy(Task),
    Stand,
    Dead(From),
}

impl Base {
    fn on_device(self) -> bool {
        matches!(self, Base::Kneel(_) | Base::Busy(_) | Base::Stand)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum From {
    Front,
    Back,
    Side,
    Head,
}

#[derive(Clone, Debug, PartialEq)]
struct Upper {
    clip: String,
    from: String,
    feather: u32,
    weight: f32,
    speed: f32,
    cue: u32,
    lasting: bool,
    window: Option<[f32; 2]>,
    looping: bool,
    fit_seconds: Option<f32>,
    added: Option<Added>,
    next: Option<String>,
    ease: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
struct Added {
    played_from: Option<String>,
    fade_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Carry {
    pub(super) slot: u32,
    pub(super) actor: Entity,
    pub(super) weapon: &'static str,
    pub(super) offset: Offset,
    pub(super) muzzle: Option<(Vec3, Vec3)>,
    pub(super) shown: bool,
}

#[derive(Clone, Debug)]
pub(super) struct Memory {
    base: Base,
    base_cue: u32,
    upper: Option<Upper>,
    upper_cue: u32,
    hold: f32,
    fallen: From,
    aiming: f32,
    feet_yaw: Option<f32>,
    turning: bool,
    weapon: Option<&'static str>,
    switch_left: f32,
    switch_total: f32,
    planting: bool,
    defusing: bool,
    strike: Option<f32>,
    feet_before: Option<[Vec3; 2]>,
    slide: f32,
    moving: Option<Moved>,
    clips: BTreeSet<String>,
    missing: Vec<String>,
    pub(super) carry: Option<Carry>,
    pub(super) hand: Option<InHand>,
    pub(super) stats: ActorStats,
}

impl Memory {
    fn locomotion(&self, body: &Body) -> LocomotionStats {
        let Some(moved) = self.moving.as_ref() else {
            let speed = Vec3::new(body.velocity.x, 0.0, body.velocity.z).length();
            return LocomotionStats {
                slot: self.stats.slot,
                state: "standing".to_owned(),
                clips: vec![ClipWeight {
                    clip: self.stats.base.clone(),
                    weight: 1.0,
                }],
                rate: self.stats.rate,
                stride: 1.0,
                hip_turn: 0.0,
                speed_mps: speed,
                planted_mps: 0.0,
                planned_slide_mps: speed,
                slide_mps: self.slide,
            };
        };
        let held = 1.0
            - moved
                .blend
                .iter()
                .map(|blended| blended.weight)
                .sum::<f32>();
        let clips = std::iter::once(ClipWeight {
            clip: moved.clip.clone(),
            weight: held,
        })
        .chain(moved.blend.iter().map(|blended| ClipWeight {
            clip: blended.clip.clone(),
            weight: blended.weight,
        }))
        .collect();
        LocomotionStats {
            slot: self.stats.slot,
            state: "moving".to_owned(),
            clips,
            rate: moved.rate,
            stride: moved.stride.scale,
            hip_turn: moved.hip_turn,
            speed_mps: moved.speed,
            planted_mps: moved.planted,
            planned_slide_mps: moved.planned_slide,
            slide_mps: self.slide,
        }
    }

    pub(super) fn task(&self) -> Option<Task> {
        if self.defusing {
            Some(Task::Defusing)
        } else if self.planting {
            Some(Task::Planting)
        } else {
            None
        }
    }
}

fn tasked(memory: &Memory, book: &PuppetBook, arms: Option<&Arms>) -> Option<&'static str> {
    match memory.task() {
        Some(Task::Planting) => book.weapons.contains_key(DEVICE).then_some(DEVICE),
        Some(Task::Defusing) => None,
        None => arms.map(|arms| weapon_key(&arms.loadout)),
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            base: Base::Idle,
            base_cue: 0,
            upper: None,
            upper_cue: 0,
            hold: 0.0,
            fallen: From::Front,
            aiming: 0.0,
            feet_yaw: None,
            turning: false,
            weapon: None,
            switch_left: 0.0,
            switch_total: 0.0,
            planting: false,
            defusing: false,
            strike: None,
            feet_before: None,
            slide: 0.0,
            moving: None,
            clips: BTreeSet::new(),
            missing: Vec::new(),
            carry: None,
            hand: None,
            stats: ActorStats::default(),
        }
    }
}

fn named_by(table: &PuppetTable) -> Vec<String> {
    let mut names: Vec<&str> = table
        .locomotion
        .gaits
        .iter()
        .map(|gait| gait.clip.as_str())
        .collect();
    names.push(&table.rise.clip);
    names.push(&table.crouch.clips.idle.clip);
    names.push(&table.land.clip);
    names.push(&table.death.headshot);
    names.extend(table.upper.clips());
    names.extend(table.melee.clips());
    names.extend(table.draw.clips.values().map(String::as_str));
    names.extend(table.device.clips());
    let mut names: Vec<String> = names.into_iter().map(str::to_owned).collect();
    names.extend(table.casts.clips());
    names.sort_unstable();
    names.dedup();
    names
}

fn known(memory: &mut Memory, table: &PuppetTable, figure: &Figure) {
    if memory.clips.len() == figure.clips.len() && figure.clips.keys().eq(memory.clips.iter()) {
        return;
    }
    memory.clips = figure.clips.keys().cloned().collect();
    memory.missing = named_by(table)
        .into_iter()
        .filter(|name| !memory.clips.contains(name))
        .collect();
}

const JOINT_NAMES: [&str; 9] = [
    HEAD,
    SPINE,
    HIP,
    LEFT_LEG[2],
    RIGHT_LEG[2],
    RIGHT_HAND,
    LEFT_HAND,
    LEFT_EYE,
    RIGHT_EYE,
];

const LEFT_EYE: &str = "L_Eye";

const RIGHT_EYE: &str = "R_Eye";

fn eye_fits(actors: &Actors, numbers: Option<&GameNumbers>) -> Vec<EyeFitStats> {
    let Some(eye) = numbers.map(|numbers| numbers.player.eye_height_m) else {
        return Vec::new();
    };
    let mut fits: Vec<EyeFitStats> = Vec::new();
    for ((_, actor, ..), _, _, joints, _) in actors.iter() {
        let Some(joints) = joints.filter(|joints| joints.rest_eye_m.is_some()) else {
            continue;
        };
        if fits.iter().any(|known| known.character == actor.character) {
            continue;
        }
        let idle = joints.rest_eye_m.unwrap_or_default();
        fits.push(EyeFitStats {
            character: actor.character.clone(),
            idle_eye_m: idle,
            body_eye_m: eye,
            gap_m: idle - eye,
            fit: joints.fit,
        });
    }
    fits.sort_by(|a, b| a.character.cmp(&b.character));
    fits
}

fn aim_check(
    memory: &Memory,
    (slot, wish, offsets): (u32, &Wish, Option<&ViewOffsets>),
    joints: &FigureJoints,
) -> Option<AimStats> {
    let carry = memory
        .carry
        .as_ref()
        .filter(|carry| !carry.weapon.is_empty())?;
    let at = JOINT_NAMES.iter().position(|known| *known == RIGHT_HAND)?;
    let hand = joints.joints.get(at).filter(|joint| joint.found)?;
    let barrel = hand.rotation * Quat::from_array(carry.offset.rotation).normalize() * Vec3::Z;
    let yaw = wish.yaw_radians + offsets.map_or(0.0, ViewOffsets::yaw_radians);
    let pitch = wish.pitch_radians + offsets.map_or(0.0, ViewOffsets::pitch_radians);
    let (sin_yaw, cos_yaw) = sin_cos(yaw);
    let (sin_pitch, cos_pitch) = sin_cos(pitch);
    let view = Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch);
    let cosine = clamp(barrel.normalize_or_zero().dot(view), -1.0, 1.0);
    Some(AimStats {
        slot,
        weapon: carry.weapon.to_owned(),
        barrel_error_deg: cosine.acos().to_degrees(),
        hold: memory.hold,
    })
}

fn head_check(
    numbers: Option<&GameNumbers>,
    (slot, body): (u32, &Body),
    joints: Option<&FigureJoints>,
) -> Option<HeadStats> {
    let player = &numbers?.player;
    let joints = joints?;
    let top = joints.head_top?;
    let eye_at = |name: &str| {
        let at = JOINT_NAMES.iter().position(|known| *known == name)?;
        joints.joints.get(at).filter(|joint| joint.found)
    };
    let eye = (eye_at(LEFT_EYE)?.position + eye_at(RIGHT_EYE)?.position) * 0.5;
    let standing = player.moving.height_m;
    let height = standing + (player.moving.crouch_height_m - standing) * body.crouch;
    let head_top = top.y - body.feet.y;
    let crown = joints.crown.map_or(head_top, |crown| crown.y - body.feet.y);
    let eye_m = eye.y - body.feet.y;
    let body_eye = body.eye.y - body.feet.y;
    Some(HeadStats {
        slot,
        head_top_m: head_top,
        height_m: height,
        top_gap_m: head_top - height,
        crown_m: crown,
        crown_gap_m: crown - height,
        eye_m,
        body_eye_m: body_eye,
        eye_gap_m: eye_m - body_eye,
    })
}

fn toward(value: f32, goal: f32, step: f32) -> f32 {
    if value < goal {
        min(value + step, goal)
    } else {
        max(value - step, goal)
    }
}

fn pitch_turn(pitch: f32) -> Quat {
    let (sin, cos) = sin_cos(-pitch * 0.5);
    Quat::from_xyzw(sin, 0.0, 0.0, cos)
}

fn yaw_turn(yaw: f32) -> Quat {
    let (sin, cos) = sin_cos(yaw * 0.5);
    Quat::from_xyzw(0.0, sin, 0.0, cos)
}

fn wrapped(angle: f32) -> f32 {
    let turn = std::f32::consts::TAU;
    (angle + std::f32::consts::PI).rem_euclid(turn) - std::f32::consts::PI
}

fn shot(one: &OneShot, lasting: bool) -> Upper {
    Upper {
        clip: one.clip.clone(),
        from: one.from.clone(),
        feather: one.feather,
        weight: one.weight,
        speed: one.speed,
        cue: 0,
        lasting,
        window: one.window,
        looping: false,
        fit_seconds: None,
        added: None,
        next: None,
        ease: None,
    }
}

fn layer_shot(
    table: &UpperTable,
    clips: &BTreeSet<String>,
    clip: Option<&String>,
) -> Option<Upper> {
    let clip = clip.filter(|clip| clips.contains(*clip))?;
    Some(Upper {
        clip: clip.clone(),
        from: table.from.clone(),
        feather: table.feather,
        weight: table.weight,
        speed: 1.0,
        cue: 0,
        lasting: false,
        window: None,
        looping: false,
        fit_seconds: None,
        added: Some(Added {
            played_from: table
                .played
                .contains(clip)
                .then(|| table.played_from.clone()),
            fade_seconds: table.fade_seconds,
        }),
        next: None,
        ease: None,
    })
}

fn throws(upper: Option<&Upper>, table: &UpperTable) -> bool {
    upper.is_some_and(|upper| {
        upper.clip == table.throw.hold || table.throw.ready.values().any(|clip| *clip == upper.clip)
    })
}

fn melee_upper(memory: &mut Memory, facts: &Facts<'_>, slot: u32) -> Option<Upper> {
    let melee = &facts.book.table.melee;
    let swing = facts.arms.swings.iter().rfind(|swing| swing.slot == slot)?;
    let clip = swing_clip((&melee.light, &melee.heavy), swing)
        .filter(|clip| memory.clips.contains(*clip))?;
    memory.strike = Some(swing.strike_tick.saturating_sub(facts.arms.tick) as f32 * facts.delta);
    Some(Upper {
        clip: clip.clone(),
        from: melee.from.clone(),
        feather: melee.feather,
        weight: melee.weight,
        speed: melee.speed,
        cue: 0,
        lasting: false,
        window: None,
        looping: false,
        fit_seconds: None,
        added: None,
        next: None,
        ease: Some(melee.fade_seconds),
    })
}

fn draw_upper(memory: &Memory, facts: &Facts<'_>, slot: u32) -> Option<Upper> {
    let draw = &facts.book.table.draw;
    let handled = facts
        .arms
        .handling
        .iter()
        .rfind(|handled| handled.slot == slot && handled.kind == Handling::Equip)?;
    let Weapon::Gun(kind) = handled.weapon else {
        return None;
    };
    let clip = draw
        .clips
        .get(handled.weapon.key())
        .filter(|clip| memory.clips.contains(*clip))?;
    Some(Upper {
        clip: clip.clone(),
        from: draw.from.clone(),
        feather: draw.feather,
        weight: draw.weight,
        speed: 1.0,
        cue: 0,
        lasting: false,
        window: None,
        looping: false,
        fit_seconds: facts
            .numbers
            .map(|numbers| numbers.weapons.gun(kind).equip_seconds),
        added: None,
        next: None,
        ease: Some(draw.fade_seconds),
    })
}

fn drawn(memory: &Memory, table: &PuppetTable, weapon: Option<&str>) -> bool {
    weapon
        .and_then(|weapon| table.draw.clips.get(weapon))
        .is_some_and(|clip| memory.clips.contains(clip))
}

struct Facts<'a> {
    book: &'a PuppetBook,
    tick: u64,
    delta: f32,
    feet: &'a BTreeMap<u32, Vec3>,
    arms: &'a ArmsEvents,
    round: &'a RoundEvents,
    powers: Option<&'a PowerEvents>,
    body: Option<&'a BodyEvents>,
    abilities: Option<&'a AbilityBook>,
    numbers: Option<&'a GameNumbers>,
    held_powers: Option<&'a Powers>,
}

fn ability_fall(facts: &Facts<'_>, victim: (u32, Vec3, f32)) -> Option<From> {
    let (slot, feet, yaw) = victim;
    let death = facts
        .arms
        .deaths
        .iter()
        .find(|death| death.victim == slot && death.ability.is_some())?;
    let killer = death.killer.filter(|killer| *killer != slot)?;
    Some(from_where((feet, yaw), facts.feet.get(&killer).copied()))
}

fn from_motion(body: &Body, wish: &Wish, slot: u32, still: f32) -> From {
    let flat = Vec3::new(body.velocity.x, 0.0, body.velocity.z);
    if flat.length() < still {
        return match slot % 3 {
            0 => From::Front,
            1 => From::Back,
            _ => From::Side,
        };
    }
    let (sin, cos) = sin_cos(wish.yaw_radians);
    let forward = Vec3::new(-sin, 0.0, -cos);
    let ahead = flat.dot(forward);
    let aside = flat.cross(forward).y;
    if ahead >= aside.abs() {
        From::Back
    } else if -ahead >= aside.abs() {
        From::Front
    } else {
        From::Side
    }
}

fn spec_of<'a>(facts: &Facts<'a>, actor: &Actor, slot: AbilitySlot) -> Option<&'a AbilitySpec> {
    let index = match slot {
        AbilitySlot::C => Slot::C,
        AbilitySlot::Q => Slot::Q,
        AbilitySlot::E => Slot::E,
        AbilitySlot::X => Slot::X,
    };
    let book = facts.abilities?;
    let character = book.character(&actor.character)?;
    Some(book.ability(AbilityId {
        character,
        slot: index,
    }))
}

fn thrown(spec: Option<&AbilitySpec>) -> bool {
    spec.is_some_and(|spec| matches!(spec.delivery, Delivery::Throw(_)))
}

fn cast_upper(
    (memory, still): (&Memory, bool),
    facts: &Facts<'_>,
    actor: &Actor,
    slot: AbilitySlot,
) -> Option<Upper> {
    let table = &facts.book.table;
    let spec = spec_of(facts, actor, slot);
    match spec.map(|spec| spec.delivery) {
        Some(Delivery::Throw(_)) => {
            let throw = &table.upper.throw;
            let rolled = spec.is_some_and(|spec| throw.rolled.contains(&spec.name));
            let clips = if rolled { &throw.roll } else { &throw.release };
            let clip = memory.weapon.and_then(|weapon| clips.get(weapon));
            layer_shot(&table.upper, &memory.clips, clip)
        }
        Some(Delivery::Place(_) | Delivery::Caster) => {
            let casts = &table.casts;
            let row = table.view.hands.abilities.get(&spec?.name)?;
            let cast = casts.clips.get(&row.clip)?;
            let clip = casts
                .named(cast, memory.weapon?)
                .filter(|clip| memory.clips.contains(clip))?;
            let from = cast
                .still_from
                .as_ref()
                .filter(|_| still)
                .unwrap_or(&casts.from);
            Some(Upper {
                clip,
                from: from.clone(),
                feather: casts.feather,
                weight: casts.weight,
                speed: row.speed,
                cue: 0,
                lasting: false,
                window: None,
                looping: false,
                fit_seconds: None,
                added: None,
                next: None,
                ease: Some(casts.fade_seconds),
            })
        }
        None => None,
    }
}

fn cast_event(table: &PuppetTable, clip: &str) -> Option<f32> {
    let casts = &table.casts;
    casts
        .clips
        .values()
        .find(|cast| {
            casts
                .suffix
                .values()
                .any(|suffix| clip.strip_prefix(cast.clip.as_str()) == Some(suffix.as_str()))
        })
        .map(|cast| cast.event_seconds)
}

fn gun_upper(memory: &Memory, facts: &Facts<'_>, slot: u32) -> Option<Upper> {
    let table = &facts.book.table.upper;
    let mut wanted = None;
    for fired in &facts.arms.shots {
        if fired.shooter == slot && fired.weapon != Weapon::Melee {
            wanted = layer_shot(table, &memory.clips, table.fire.get(fired.weapon.key()));
        }
    }
    for handled in &facts.arms.handling {
        let Weapon::Gun(kind) = handled.weapon else {
            continue;
        };
        if handled.slot != slot || handled.kind != Handling::ReloadStarted {
            continue;
        }
        let pace = facts
            .held_powers
            .map_or(1.0, |powers| powers.modifiers(slot).reload);
        let seconds = facts
            .numbers
            .map(|numbers| numbers.weapons.gun(kind).reload_seconds / max(pace, 1.0e-3));
        wanted =
            layer_shot(table, &memory.clips, table.reload.get(handled.weapon.key())).map(|upper| {
                Upper {
                    fit_seconds: seconds,
                    ..upper
                }
            });
    }
    wanted
}

fn throw_upper(memory: &mut Memory, facts: &Facts<'_>, actor: &Actor) -> Option<Upper> {
    let table = &facts.book.table.upper;
    let holding = facts
        .held_powers
        .and_then(|powers| powers.held(actor.slot))
        .is_some_and(|held| thrown(spec_of(facts, actor, held.ability)));
    let in_throw = throws(memory.upper.as_ref(), table);
    if in_throw && !holding {
        memory.upper = None;
    }
    let free = memory
        .upper
        .as_ref()
        .is_none_or(|upper| upper.added.is_some() && !in_throw);
    if !(holding && free) {
        return None;
    }
    let ready = memory
        .weapon
        .and_then(|weapon| table.throw.ready.get(weapon));
    layer_shot(table, &memory.clips, ready).map(|upper| Upper {
        next: Some(table.throw.hold.clone()),
        ..upper
    })
}

fn hand_of(
    was: Option<InHand>,
    facts: &Facts<'_>,
    (entity, actor): (Entity, &Actor),
) -> Option<InHand> {
    let table = &facts.book.table;
    let throw = &table.upper.throw;
    let cast = facts
        .powers
        .iter()
        .flat_map(|powers| &powers.casts)
        .rfind(|cast| cast.slot == actor.slot);
    let held = facts
        .held_powers
        .and_then(|powers| powers.held(actor.slot))
        .map(|held| held.ability);
    let Some(ability) = cast.map(|cast| cast.ability).or(held) else {
        return was.filter(|hand| hand.cast.is_some() && hand.holds(facts.tick));
    };
    let spec = spec_of(facts, actor, ability);
    let tossed = thrown(spec);
    let prop = spec.and_then(|spec| {
        let row = table.view.hands.abilities.get(&spec.name)?;
        Some((spec, row, row.prop.as_ref()?))
    });
    let Some((spec, row, prop)) = prop.filter(|_| tossed || cast.is_some()) else {
        return was.filter(|hand| hand.cast.is_some() && hand.holds(facts.tick));
    };
    let index = ability_index(ability);
    let kept = was.filter(|hand| hand.ability == index && hand.prop == prop.key);
    let (cast, release) = match (cast, kept.as_ref()) {
        (Some(_), _) => {
            let leave = if !tossed {
                table
                    .casts
                    .clips
                    .get(&row.clip)
                    .map_or(prop.leave_seconds, |cast| cast.event_seconds)
                    / max(row.speed, 1.0e-3)
            } else if throw.rolled.contains(&spec.name) {
                throw.roll_leave_seconds
            } else {
                throw.release_leave_seconds
            };
            (
                Some(facts.tick),
                Some(leave_tick(facts.tick, leave, facts.delta)),
            )
        }
        (None, Some(hand)) => (hand.cast, hand.release),
        (None, None) => (None, None),
    };
    Some(InHand {
        holder: entity,
        socket: throw.hand.joint.clone(),
        seat: table.hand_axes.left.seat(&throw.hand),
        prop: prop.key.clone(),
        size_m: prop.size_m,
        owner: actor.slot,
        team: actor.team.index(),
        ability: index,
        cast,
        release,
    })
    .filter(|hand| hand.holds(facts.tick))
}

fn feet_follow(
    memory: &mut Memory,
    table: &PuppetTable,
    (body, wish): (&Body, &Wish),
    delta: f32,
) -> f32 {
    let view = wish.yaw_radians;
    let feet = memory.feet_yaw.unwrap_or(view);
    let moving = !body.grounded
        || Vec3::new(body.velocity.x, 0.0, body.velocity.z).length() >= table.still_below_mps;
    let offset = wrapped(view - feet);
    if moving || offset.abs() > table.look.limit_radians {
        memory.turning = true;
    }
    let mut feet = feet;
    if memory.turning {
        let step = table.look.catch_up_radians_per_second * delta;
        let left = wrapped(view - feet);
        if left.abs() <= step {
            feet = view;
            memory.turning = moving;
        } else {
            feet += step * left.signum();
        }
    }
    memory.feet_yaw = Some(feet);
    wrapped(view - feet)
}

struct Gun {
    reload: Option<f32>,
    dip: f32,
    aim: f32,
}

fn gun_state(
    memory: &mut Memory,
    facts: &Facts<'_>,
    (weapon, arms, slot): (Option<&'static str>, Option<&Arms>, u32),
) -> Gun {
    if weapon != memory.weapon {
        if drawn(memory, &facts.book.table, weapon) {
            memory.switch_total = 0.0;
            memory.switch_left = 0.0;
        } else if memory.weapon.is_some() && weapon.is_some() {
            let seconds = facts.numbers.zip(arms).map_or(0.0, |(numbers, arms)| {
                arms.loadout
                    .active_gun()
                    .map_or(numbers.weapons.melee.equip_seconds, |gun| {
                        numbers.weapons.gun(gun.kind()).equip_seconds
                    })
            });
            memory.switch_total = seconds;
            memory.switch_left = seconds;
        }
        memory.weapon = weapon;
    }
    memory.switch_left = max(memory.switch_left - facts.delta, 0.0);
    let dip = if memory.switch_total > 0.0 {
        let done = 1.0 - memory.switch_left / memory.switch_total;
        4.0 * done * (1.0 - done)
    } else {
        0.0
    };
    let reload = facts.numbers.zip(arms).and_then(|(numbers, arms)| {
        arms.loadout
            .view(&numbers.weapons, Stance::Stand)
            .reload_progress
    });
    let holding = facts
        .held_powers
        .is_some_and(|powers| powers.held(slot).is_some());
    let table = &facts.book.table;
    let step = facts.delta / max(table.aiming.seconds, facts.delta);
    memory.aiming = toward(memory.aiming, f32::from(u8::from(holding)), step);
    Gun {
        reload,
        dip,
        aim: memory.aiming,
    }
}

fn from_where(victim: (Vec3, f32), from: Option<Vec3>) -> From {
    let Some(from) = from else {
        return From::Front;
    };
    let (feet, yaw) = victim;
    let (sin, cos) = sin_cos(yaw);
    let forward = Vec3::new(-sin, 0.0, -cos);
    let to = Vec3::new(from.x - feet.x, 0.0, from.z - feet.z);
    let ahead = to.dot(forward);
    let aside = to.cross(forward).y;
    if ahead >= aside.abs() {
        From::Front
    } else if -ahead >= aside.abs() {
        From::Back
    } else {
        From::Side
    }
}

fn hit_clip(table: &PuppetTable, part: BodyPart, from: From) -> String {
    let hits = &table.hits;
    match (part, from) {
        (BodyPart::Head, _) => hits.head.clone(),
        (BodyPart::Leg, _) => hits.low.clone(),
        (BodyPart::Body, From::Front | From::Head) => hits.front.clone(),
        (BodyPart::Body, From::Back) => hits.back.clone(),
        (BodyPart::Body, From::Side) => hits.side.clone(),
    }
}

fn round_state(memory: &mut Memory, facts: &Facts<'_>, slot: usize) {
    for event in &facts.round.events {
        match event {
            Event::Respawn { player, .. } if *player == slot => {
                memory.upper = None;
                memory.planting = false;
                memory.defusing = false;
            }
            Event::PlantStarted { player, .. } if *player == slot => memory.planting = true,
            Event::PlantCancelled { player, .. } | Event::PlantCompleted { player, .. }
                if *player == slot =>
            {
                memory.planting = false;
            }
            Event::DefuseStarted { player, .. } if *player == slot => memory.defusing = true,
            Event::DefuseCancelled { player, .. } | Event::DefuseCompleted { player }
                if *player == slot =>
            {
                memory.defusing = false;
            }
            Event::Phase { .. }
            | Event::Respawn { .. }
            | Event::GiveWeapon { .. }
            | Event::TakeWeapon { .. }
            | Event::Armor { .. }
            | Event::Charges { .. }
            | Event::Credits { .. }
            | Event::BuyRefused { .. }
            | Event::AbilityUsed { .. }
            | Event::AbilityRefused { .. }
            | Event::DeviceGiven { .. }
            | Event::DeviceDropped { .. }
            | Event::DevicePickedUp { .. }
            | Event::PlantStarted { .. }
            | Event::PlantCancelled { .. }
            | Event::PlantCompleted { .. }
            | Event::DefuseStarted { .. }
            | Event::DefuseHalf { .. }
            | Event::DefuseCancelled { .. }
            | Event::DefuseCompleted { .. }
            | Event::DeviceExploded { .. }
            | Event::KillFeed { .. }
            | Event::UltimatePoint { .. }
            | Event::UltimateReady { .. }
            | Event::RoundEnd { .. }
            | Event::SideSwap { .. }
            | Event::MatchEnd { .. } => {}
        }
    }
}

fn note_hits(
    memory: &mut Memory,
    facts: &Facts<'_>,
    (slot, body, wish): (u32, &Body, &Wish),
) -> Option<(BodyPart, From)> {
    let mut reaction = None;
    for hit in &facts.arms.hits {
        if hit.victim == slot {
            let from = from_where(
                (body.feet, wish.yaw_radians),
                facts.feet.get(&hit.shooter).copied(),
            );
            let headshot = hit.killed
                && hit.part == BodyPart::Head
                && memory.clips.contains(&facts.book.table.death.headshot);
            memory.fallen = if headshot { From::Head } else { from };
            if !hit.killed {
                reaction = Some((hit.part, from));
            }
        }
    }
    reaction
}

fn upper_from_events(
    memory: &mut Memory,
    facts: &Facts<'_>,
    actor: &Actor,
    (body, wish): (&Body, &Wish),
) {
    let table = &facts.book.table;
    let slot = actor.slot as usize;
    round_state(memory, facts, slot);
    let reaction = note_hits(memory, facts, (actor.slot, body, wish));
    if memory.upper.as_ref().is_some_and(|upper| upper.lasting) {
        return;
    }
    let switched = facts
        .arms
        .handling
        .iter()
        .any(|handled| handled.slot == actor.slot && handled.kind == Handling::Equip);
    if switched
        && memory
            .upper
            .as_ref()
            .is_some_and(|upper| table.upper.reload.values().any(|clip| *clip == upper.clip))
    {
        memory.upper = None;
    }
    let mut wanted = draw_upper(memory, facts, actor.slot);
    if let Some(throw) = throw_upper(memory, facts, actor) {
        wanted = Some(throw);
    }
    let still = body.grounded
        && Vec3::new(body.velocity.x, 0.0, body.velocity.z).length() < table.still_below_mps;
    for cast in facts.powers.iter().flat_map(|powers| &powers.casts) {
        if cast.slot == actor.slot {
            wanted = cast_upper((memory, still), facts, actor, cast.ability).or(wanted);
        }
    }
    for event in &facts.round.events {
        if let Event::MatchEnd { winner, .. } = event
            && *winner == actor.team
        {
            wanted = Some(shot(&table.victory, true));
        }
    }
    if wanted.as_ref().is_some_and(|upper| upper.lasting) {
        start(memory, wanted);
        return;
    }
    let landed = facts
        .body
        .iter()
        .flat_map(|events| &events.landings)
        .any(|landing| landing.slot == actor.slot && landing.hard);
    if landed && body.grounded && memory.clips.contains(&table.land.clip) {
        wanted = Some(shot(&table.land, false));
    }
    if let Some(gun) = gun_upper(memory, facts, actor.slot) {
        wanted = Some(gun);
    }
    if let Some(swing) = melee_upper(memory, facts, actor.slot) {
        wanted = Some(swing);
    }
    if let Some((part, from)) = reaction {
        let hits = &table.hits;
        wanted = Some(Upper {
            clip: hit_clip(table, part, from),
            from: hits.from.clone(),
            feather: hits.feather,
            weight: hits.weight,
            speed: hits.speed,
            cue: 0,
            lasting: false,
            window: None,
            looping: false,
            fit_seconds: None,
            added: None,
            next: None,
            ease: None,
        });
    }
    start(memory, wanted);
}

fn start(memory: &mut Memory, wanted: Option<Upper>) {
    if let Some(mut upper) = wanted {
        memory.upper_cue = memory.upper_cue.wrapping_add(1);
        upper.cue = memory.upper_cue;
        memory.upper = Some(upper);
    }
}

fn base_done(memory: &Memory, joints: Option<&FigureJoints>) -> bool {
    joints.is_some_and(|joints| joints.base_cue == memory.base_cue && joints.base_done)
}

fn knelt(memory: &Memory, table: &PuppetTable, task: Task, joints: Option<&FigureJoints>) -> Base {
    let kneeling = match memory.base {
        Base::Busy(was) => was != task,
        Base::Kneel(was) => was != task || !base_done(memory, joints),
        Base::Idle
        | Base::Walk
        | Base::Run
        | Base::Turn
        | Base::Rise
        | Base::Fall
        | Base::CrouchIdle
        | Base::CrouchWalk
        | Base::Stand
        | Base::Dead(_) => true,
    };
    if kneeling && memory.clips.contains(&table.device.kneel.clip) {
        Base::Kneel(task)
    } else {
        Base::Busy(task)
    }
}

fn stood(
    memory: &Memory,
    table: &PuppetTable,
    (body, joints): (&Body, Option<&FigureJoints>),
) -> bool {
    let still = body.grounded
        && Vec3::new(body.velocity.x, 0.0, body.velocity.z).length() < table.still_below_mps;
    let stands = memory.clips.contains(&table.device.stand.clip);
    match memory.base {
        Base::Kneel(_) | Base::Busy(_) => still && stands,
        Base::Stand => still && !base_done(memory, joints),
        Base::Idle
        | Base::Walk
        | Base::Run
        | Base::Turn
        | Base::Rise
        | Base::Fall
        | Base::CrouchIdle
        | Base::CrouchWalk
        | Base::Dead(_) => false,
    }
}

fn base_of(
    memory: &Memory,
    table: &PuppetTable,
    (body, joints): (&Body, Option<&FigureJoints>),
) -> Base {
    if let Some(task) = memory.task() {
        return knelt(memory, table, task, joints);
    }
    if stood(memory, table, (body, joints)) {
        return Base::Stand;
    }
    if !body.grounded {
        let rising = body.velocity.y > 0.0;
        let was_rising = memory.base == Base::Rise;
        let in_air = matches!(memory.base, Base::Rise | Base::Fall);
        return if rising && (was_rising || !in_air) {
            Base::Rise
        } else {
            Base::Fall
        };
    }
    let crouched = body.crouch >= table.crouch.clips.from;
    let crouching = crouched && table.crouch.chosen().is_some();
    let speed = Vec3::new(body.velocity.x, 0.0, body.velocity.z).length();
    if speed >= table.still_below_mps {
        return match (crouched, body.motion) {
            (true, _) => Base::CrouchWalk,
            (false, Motion::Running) => Base::Run,
            (false, Motion::Standing | Motion::Walking | Motion::Crouching | Motion::Air) => {
                Base::Walk
            }
        };
    }
    if crouching {
        return Base::CrouchIdle;
    }
    let turning = memory.turning;
    if turning { Base::Turn } else { Base::Idle }
}

fn base_clip(table: &PuppetTable, base: Base) -> (String, f32, bool) {
    let of = |clip: &BaseClip, looping: bool| {
        (
            clip.clip.clone(),
            clip.fade_seconds,
            clip.looping.unwrap_or(looping),
        )
    };
    let crouched = |standing: &BaseClip| {
        table
            .crouch
            .chosen()
            .map_or_else(|| of(standing, true), |clips| of(&clips.idle, true))
    };
    let death = &table.death;
    match base {
        Base::Idle => of(&table.idle, true),
        Base::Turn => of(&table.turn, true),
        Base::Rise => of(&table.rise, false),
        Base::Fall => of(&table.fall, true),
        Base::CrouchIdle | Base::Walk | Base::Run | Base::CrouchWalk => crouched(&table.idle),
        Base::Kneel(_) => of(&table.device.kneel, false),
        Base::Busy(Task::Planting) => of(&table.device.plant, true),
        Base::Busy(Task::Defusing) => of(&table.device.defuse, true),
        Base::Stand => of(&table.device.stand, false),
        Base::Dead(from) => (
            match from {
                From::Front => death.from_front.clone(),
                From::Back => death.from_back.clone(),
                From::Side => death.from_side.clone(),
                From::Head => death.headshot.clone(),
            },
            death.fade_seconds,
            false,
        ),
    }
}

fn rate_of(table: &PuppetTable, base: Base) -> f32 {
    match base {
        Base::Dead(_) => table.death.speed,
        Base::Idle
        | Base::Walk
        | Base::Run
        | Base::CrouchWalk
        | Base::Turn
        | Base::Rise
        | Base::Fall
        | Base::CrouchIdle
        | Base::Kneel(_)
        | Base::Busy(_)
        | Base::Stand => 1.0,
    }
}

pub(super) struct Shaped {
    clip: String,
    fade_seconds: f32,
    looping: bool,
    rate: f32,
    crouch: f32,
    aim: f32,
    moving: Option<Moved>,
}

fn shaped(
    table: &PuppetTable,
    base: Base,
    (alive, body, wish): (bool, &Body, &Wish),
    (pitch, clips): (f32, &BTreeSet<String>),
) -> Shaped {
    let has = |clip: &str| clips.contains(clip);
    let moving = matches!(base, Base::Walk | Base::Run | Base::CrouchWalk)
        .then(|| {
            moved(
                &table.locomotion,
                base == Base::CrouchWalk,
                (body, wish),
                &has,
            )
        })
        .flatten();
    let (clip, fade_seconds, looping) = match &moving {
        Some(moved) => (moved.clip.clone(), table.locomotion.fade_seconds, true),
        None if base.on_device() && !has(&base_clip(table, base).0) => base_clip(table, Base::Idle),
        None => base_clip(table, base),
    };
    let rate = moving
        .as_ref()
        .map_or_else(|| rate_of(table, base), |moved| moved.rate);
    let posed_crouched = match base {
        Base::CrouchIdle => table.crouch.chosen().is_some(),
        Base::CrouchWalk => moving.as_ref().is_some_and(|moved| !moved.dropped),
        Base::Idle
        | Base::Walk
        | Base::Run
        | Base::Turn
        | Base::Rise
        | Base::Fall
        | Base::Kneel(_)
        | Base::Busy(_)
        | Base::Stand
        | Base::Dead(_) => false,
    };
    let (crouch, aim) = match (alive, posed_crouched) {
        (false, _) => (0.0, 0.0),
        (true, true) => (0.0, pitch * table.aim.share),
        (true, false) => (body.crouch * table.crouch.drop_m, pitch * table.aim.share),
    };
    Shaped {
        clip,
        fade_seconds,
        looping,
        rate,
        crouch,
        aim,
        moving,
    }
}

fn slid(memory: &mut Memory, joints: Option<&FigureJoints>, delta: f32) -> f32 {
    let feet = joints.and_then(|joints| {
        let at = |name: &str| {
            let index = joints.names.iter().position(|known| known == name)?;
            joints
                .joints
                .get(index)
                .filter(|joint| joint.found)
                .map(|joint| joint.position)
        };
        Some([at(LEFT_LEG[2])?, at(RIGHT_LEG[2])?])
    });
    let before = std::mem::replace(&mut memory.feet_before, feet);
    let (Some(now), Some(was)) = (feet, before) else {
        return 0.0;
    };
    let stance = usize::from(now[1].y < now[0].y);
    let moved = Vec3::new(
        now[stance].x - was[stance].x,
        0.0,
        now[stance].z - was[stance].z,
    );
    if delta > 0.0 {
        moved.length() / delta
    } else {
        0.0
    }
}

pub(super) struct Posed {
    pub(super) hold: Option<Hold>,
    pub(super) carry: Option<(&'static str, Offset, Option<Vec3>, Vec3)>,
}

struct Pose {
    pitch: f32,
    weight: f32,
    gun: Gun,
    played: bool,
}

fn left_played(memory: &Memory) -> bool {
    memory
        .upper
        .as_ref()
        .and_then(|upper| upper.added.as_ref())
        .is_some_and(|added| added.played_from.is_some())
}

fn held_pose(book: &PuppetBook, weapon: &'static str, given: &Pose) -> Option<Posed> {
    let table = &book.table;
    let pose = table.holds.get(table.weapons.get(weapon)?)?;
    let Pose { pitch, weight, .. } = *given;
    let lowered = max(given.gun.dip, given.gun.aim * table.aiming.lower);
    let pitch = pitch - table.switch.dip_radians * lowered;
    let drop = table.switch.dip_m * lowered;
    let markers = book
        .weapons
        .get(weapon)
        .map(|look| look.markers)
        .unwrap_or_default();
    let turn = pitch_turn(pitch * pose.pitch_share);
    let pivot = Vec3::from_array(pose.pivot) - Vec3::new(0.0, drop, 0.0);
    let grip = Vec3::from_array(pose.grip);
    let grip_r = markers.grip_r.unwrap_or(Vec3::ZERO);
    let placed = |point: Vec3| pivot + turn * (grip + point - grip_r);
    let axes = &table.hand_axes;
    let right_hand = pose.right.hand(&axes.right);
    let right_wrist = pose.right.wrist(grip_r);
    let right = HandTarget {
        position: placed(right_wrist).to_array(),
        rotation: (turn * right_hand).normalize().to_array(),
        pole: pose.right.pole,
    };
    let raised = Vec3::from_array(table.aiming.hand) - Vec3::new(0.0, drop, 0.0);
    let left = pose
        .left
        .zip(markers.grip_l)
        .filter(|_| !given.played)
        .map(|(grip, at)| HandTarget {
            position: placed(grip.wrist(at))
                .lerp(raised, given.gun.aim)
                .to_array(),
            rotation: (turn * grip.hand(&axes.left)).normalize().to_array(),
            pole: grip.pole,
        });
    let back = right_hand.inverse();
    let carry = book.weapons.get(weapon).map(|_| {
        let offset = Offset {
            translation: (back * -right_wrist).to_array(),
            rotation: back.normalize().to_array(),
            scale: 1.0,
        };
        (weapon, offset, markers.muzzle.map(placed), turn * Vec3::Z)
    });
    Some(Posed {
        hold: Some(Hold {
            space: HoldSpace::Figure,
            weight,
            chest: table.aim.chest.clone(),
            right: Some(right),
            left,
        }),
        carry,
    })
}

fn stepped(
    memory: &mut Memory,
    facts: &Facts<'_>,
    (actor, body, wish): (&Actor, &Body, &Wish),
    (alive, joints): (bool, Option<&FigureJoints>),
) -> Base {
    if let Some(upper) = &memory.upper
        && !upper.lasting
        && joints.is_some_and(|joints| joints.upper_cue == Some(upper.cue) && joints.upper_done)
    {
        let next = upper.next.as_ref();
        let then = layer_shot(&facts.book.table.upper, &memory.clips, next).map(|upper| Upper {
            looping: true,
            ..upper
        });
        memory.upper = None;
        start(memory, then);
    }
    let dead = matches!(memory.base, Base::Dead(_));
    let base = if alive {
        if dead {
            memory.upper = None;
        }
        upper_from_events(memory, facts, actor, (body, wish));
        base_of(memory, &facts.book.table, (body, joints))
    } else {
        if !dead {
            note_hits(memory, facts, (actor.slot, body, wish));
            if !facts.arms.hits.iter().any(|hit| hit.victim == actor.slot) {
                memory.fallen = ability_fall(facts, (actor.slot, body.feet, wish.yaw_radians))
                    .unwrap_or_else(|| {
                        from_motion(body, wish, actor.slot, facts.book.table.still_below_mps)
                    });
            }
            memory.upper = None;
            memory.planting = false;
            memory.defusing = false;
            memory.feet_yaw = None;
            memory.turning = false;
        }
        match memory.base {
            Base::Dead(from) => Base::Dead(from),
            Base::Idle
            | Base::Walk
            | Base::Run
            | Base::Turn
            | Base::Rise
            | Base::Fall
            | Base::CrouchIdle
            | Base::CrouchWalk
            | Base::Kneel(_)
            | Base::Busy(_)
            | Base::Stand => Base::Dead(memory.fallen),
        }
    };
    if base != memory.base
        && matches!(
            base,
            Base::Rise | Base::Dead(_) | Base::Kneel(_) | Base::Stand
        )
    {
        memory.base_cue = memory.base_cue.wrapping_add(1);
    }
    memory.base = base;
    base
}

fn carried(
    (posed, (shift, fit)): (Option<&Posed>, (Vec3, f32)),
    (slot, actor): (u32, Entity),
    (body, wish): (&Body, &Wish),
    may_show: bool,
) -> Carry {
    let facing = yaw_turn(wish.yaw_radians + std::f32::consts::PI);
    let held = posed.and_then(|posed| posed.carry);
    Carry {
        slot,
        actor,
        weapon: held.map_or("", |(weapon, ..)| weapon),
        offset: held.map_or(Offset::STILL, |(_, offset, ..)| offset),
        muzzle: held.and_then(|(_, _, muzzle, direction)| {
            let fit = if fit > 0.0 { fit } else { 1.0 };
            muzzle.map(|muzzle| {
                (
                    body.feet + facing * ((muzzle + shift) * fit),
                    facing * direction,
                )
            })
        }),
        shown: may_show && held.is_some(),
    }
}

fn upper_layer(memory: &Memory) -> Option<UpperLayer> {
    memory.upper.as_ref().map(|upper| UpperLayer {
        clip: upper.clip.clone(),
        weight: upper.weight,
        looping: upper.looping,
        from: upper.from.clone(),
        feather: upper.feather,
        speed: upper.speed,
        cue: upper.cue,
        window: upper.window,
        fit_seconds: upper.fit_seconds,
        additive: upper.added.is_some(),
        played_from: upper
            .added
            .as_ref()
            .and_then(|added| added.played_from.clone()),
        fade_seconds: upper
            .added
            .as_ref()
            .map_or(upper.ease.unwrap_or(0.0), |added| added.fade_seconds),
    })
}

fn judged(refusals: &mut Refusals, entity: Entity, weapon: Option<&str>, posed: bool) {
    match weapon {
        Some(weapon) if !posed => refusals.refuse(
            entity,
            PUPPET_PART,
            Error::new(ErrorCode::UnknownReference, "$.puppet.weapons").got(weapon),
        ),
        Some(_) | None => refusals.accept(entity, PUPPET_PART),
    }
}

fn abilities_follow(memories: &mut Memories, numbers: &Numbers, delta: f32) {
    let revision = numbers.revision();
    if memories
        .abilities
        .as_ref()
        .is_some_and(|(known, _)| *known == revision)
    {
        return;
    }
    let ticks_per_second = if delta > 0.0 {
        (1.0 / delta).round() as u32
    } else {
        0
    };
    let book = numbers
        .get()
        .and_then(|numbers| AbilityBook::parse(&numbers.text, ticks_per_second).ok())
        .map(std::sync::Arc::new);
    memories.abilities = Some((revision, book));
}

fn figure_play(
    table: &PuppetTable,
    memory: &Memory,
    shaped: &Shaped,
    (look_yaw, hold): (f32, Option<Hold>),
) -> FigurePlay {
    let pinned = &table.pin_feet.clips;
    let pin_feet = (pinned.contains(&shaped.clip)
        || memory
            .upper
            .as_ref()
            .is_some_and(|upper| pinned.contains(&upper.clip) && upper.from == table.look.hips))
    .then_some(table.pin_feet.contact_m);
    let moving = shaped.moving.as_ref();
    FigurePlay {
        base: BaseLayer {
            clip: shaped.clip.clone(),
            speed: shaped.rate,
            looping: shaped.looping,
            fade_seconds: shaped.fade_seconds,
            cue: memory.base_cue,
            fit_seconds: None,
            blend: moving.map_or_else(Vec::new, |moved| moved.blend.clone()),
            stride: moving.map(|moved| moved.stride),
            synced: moving.is_some(),
        },
        upper: upper_layer(memory),
        aim: Some(Aim {
            pitch: shaped.aim,
            chain: table.aim.chain.clone(),
            yaw: look_yaw - moving.map_or(0.0, |moved| moved.hip_turn),
            hips: Some(table.look.hips.clone()),
        }),
        hold,
        crouch: Some(shaped.crouch),
        pin_feet,
    }
}

fn joints_named() -> FigureJoints {
    FigureJoints {
        names: JOINT_NAMES.iter().map(|name| (*name).to_owned()).collect(),
        ..FigureJoints::default()
    }
}

type Reads<'w, 's> = (
    (Res<'w, FixedTime>, Res<'w, Numbers>),
    (
        Res<'w, ArmsEvents>,
        Res<'w, RoundEvents>,
        Option<Res<'w, PowerEvents>>,
        Option<Res<'w, BodyEvents>>,
        Option<Res<'w, Powers>>,
    ),
    Query<'w, 's, &'static PuppetFile>,
    (Option<Res<'w, Watched>>, Option<Res<'w, TestCamera>>),
);

type Actors<'w, 's> = Query<
    'w,
    's,
    (
        (Entity, &'static Actor, &'static Body, &'static Wish),
        (Option<&'static Arms>, Option<&'static ViewOffsets>),
        (Has<Alive>, Option<&'static Node>, &'static Figure),
        Option<&'static FigureJoints>,
        Option<&'static mut FigurePlay>,
    ),
    With<Figure>,
>;

type Writes<'w, 's> = (
    ResMut<'w, Memories>,
    ResMut<'w, PuppetStats>,
    ResMut<'w, Refusals>,
    Spawner<'w, 's, PuppetBrick>,
);

type Stated<'a> = (Option<&'static str>, bool, Option<f32>, f32);

fn actor_stats(
    (memory, table): (&Memory, &PuppetTable),
    (actor, body, joints): (&Actor, &Body, Option<&FigureJoints>),
    (shape, holding, look_yaw): (&Shaped, bool, f32),
    (weapon, alive, reload, dip): Stated<'_>,
) -> ActorStats {
    let playing = joints.filter(|joints| {
        memory
            .upper
            .as_ref()
            .is_some_and(|upper| joints.upper_cue == Some(upper.cue))
    });
    ActorStats {
        slot: actor.slot,
        base: shape.clip.clone(),
        rate: shape.rate,
        upper: memory.upper.as_ref().map(|upper| upper.clip.clone()),
        upper_seconds: playing.map_or(0.0, |joints| joints.upper_seconds),
        upper_rate: playing.map_or(0.0, |joints| joints.upper_rate),
        upper_active: playing.is_some_and(|joints| !joints.upper_done),
        event_s: memory.upper.as_ref().and_then(|upper| {
            let event = memory
                .strike
                .filter(|_| table.melee.clips().any(|clip| clip == upper.clip))
                .or_else(|| cast_event(table, &upper.clip))?;
            Some(event / max(upper.speed, 1.0e-3))
        }),
        hold: if holding { memory.hold } else { 0.0 },
        crouch: shape.crouch,
        aim: shape.aim,
        weapon: weapon.map(str::to_owned),
        dead: !alive,
        look_yaw,
        step_m: joints
            .and_then(|joints| joints.steps.get(&shape.clip))
            .copied()
            .unwrap_or_default(),
        drift_m: joints
            .and_then(|joints| joints.drifts.get(&shape.clip))
            .copied()
            .unwrap_or_default(),
        reload,
        dip,
        in_hand: memory.hand.as_ref().map(|hand| hand.prop.clone()),
        team: actor.team.index() as u32,
        feet: body.feet.to_array(),
        eye: body.eye.to_array(),
    }
}

fn held_blend(memory: &mut Memory, table: &PuppetTable, delta: f32) {
    let free = memory
        .upper
        .as_ref()
        .is_none_or(|upper| upper.added.is_some())
        && !memory.base.on_device();
    let goal = f32::from(u8::from(free));
    let seconds = memory
        .upper
        .as_ref()
        .and_then(|upper| upper.ease)
        .filter(|_| !free)
        .unwrap_or(table.hold_blend_seconds);
    let blend = delta / max(seconds, delta);
    memory.hold = toward(memory.hold, goal, blend);
}

pub(super) fn animate(
    (
        (fixed, numbers),
        (arms_events, round_events, powers, body_events, held_powers),
        files,
        (watched, test),
    ): Reads,
    mut actors: Actors,
    (mut memories, mut stats, mut refusals, mut spawner): Writes,
) {
    let Some(book) = files.iter().find_map(PuppetFile::book).cloned() else {
        let refusal = Some("no prepared `puppet` component (scene entity `game`)".to_owned());
        if stats.refusal != refusal {
            stats.refusal = refusal;
        }
        return;
    };
    if stats.refusal.is_some() {
        stats.refusal = None;
    }
    let outside = test.as_deref().is_some_and(|test| test.outside);
    let followed = watched.and_then(|watched| watched.entity);
    let watched = followed.filter(|_| !outside);
    let feet: BTreeMap<u32, Vec3> = actors
        .iter()
        .map(|((_, actor, body, _), ..)| (actor.slot, body.feet))
        .collect();
    abilities_follow(&mut memories, &numbers, fixed.delta_seconds);
    let abilities = memories
        .abilities
        .as_ref()
        .and_then(|(_, book)| book.clone());
    let facts = Facts {
        book: &book,
        tick: fixed.tick,
        delta: fixed.delta_seconds,
        feet: &feet,
        arms: &arms_events,
        round: &round_events,
        powers: powers.as_deref(),
        body: body_events.as_deref(),
        abilities: abilities.as_deref(),
        numbers: numbers.get(),
        held_powers: held_powers.as_deref(),
    };
    let table = &book.table;
    let mut seen = Vec::new();
    let mut listed = Vec::new();
    for ((entity, actor, body, wish), (arms, offsets), (alive, _, figure), joints, play) in
        &mut actors
    {
        seen.push(entity);
        let memory = memories.held.entry(entity).or_default();
        known(memory, table, figure);
        if joints.is_none() {
            spawner.insert(entity, joints_named());
        }
        let viewed = watched == Some(entity);
        let look_yaw = if alive {
            feet_follow(memory, table, (body, wish), facts.delta)
        } else {
            0.0
        };
        let base = stepped(memory, &facts, (actor, body, wish), (alive, joints));
        memory.hand = alive
            .then(|| hand_of(memory.hand.take(), &facts, (entity, actor)))
            .flatten()
            .filter(|_| !viewed);
        let pitch = wish.pitch_radians + offsets.map_or(0.0, |offsets| offsets.pitch_radians());
        let shape = shaped(table, base, (alive, body, wish), (pitch, &memory.clips));
        memory.slide = slid(memory, joints, facts.delta);
        let weapon = tasked(memory, &book, arms.filter(|_| alive));
        held_blend(memory, table, facts.delta);
        let pose = Pose {
            pitch: shape.aim,
            weight: memory.hold,
            gun: gun_state(memory, &facts, (weapon, arms, actor.slot)),
            played: left_played(memory),
        };
        let (reload, dip) = (pose.gun.reload, pose.gun.dip);
        let posed = weapon.and_then(|weapon| held_pose(&book, weapon, &pose));
        judged(&mut refusals, entity, weapon, posed.is_some());
        let hold = posed.as_ref().and_then(|posed| posed.hold.clone());
        memory.carry = Some(carried(
            (
                posed.as_ref(),
                joints.map_or((Vec3::ZERO, 1.0), |joints| (joints.hold_shift, joints.fit)),
            ),
            (actor.slot, entity),
            (body, wish),
            alive && !viewed,
        ));
        let wanted = figure_play(table, memory, &shape, (look_yaw, hold));
        memory.stats = actor_stats(
            (memory, table),
            (actor, body, joints),
            (&shape, wanted.hold.is_some(), look_yaw),
            (weapon, alive, reload, dip),
        );
        memory.moving = shape.moving;
        listed.push((actor.slot, memory.stats.clone()));
        match play {
            Some(mut play) => {
                play.set_if_neq(wanted);
            }
            None => spawner.insert(entity, wanted),
        }
    }
    memories.held.retain(|entity, _| seen.contains(entity));
    refusals.keep_part(PUPPET_PART, |entity| seen.contains(&entity));
    listed.sort_unstable_by_key(|(slot, _)| *slot);
    let reported: Vec<ActorStats> = listed.into_iter().map(|(_, stats)| stats).collect();
    if stats.actors != reported {
        stats.actors = reported;
    }
    watched_report(&mut stats, &memories, (&actors, followed), facts.numbers);
}

fn watched_report(
    stats: &mut PuppetStats,
    memories: &Memories,
    (actors, followed): (&Actors, Option<Entity>),
    numbers: Option<&GameNumbers>,
) {
    let watched = followed.and_then(|entity| Some((entity, actors.get(entity).ok()?)));
    let locomotion = watched.and_then(|(entity, ((_, _, body, _), ..))| {
        Some(memories.held.get(&entity)?.locomotion(body))
    });
    let eyes = eye_fits(actors, numbers);
    if stats.eyes != eyes {
        stats.eyes = eyes;
    }
    let head = watched.and_then(|(_, ((_, actor, body, _), _, _, joints, _))| {
        head_check(numbers, (actor.slot, body), joints)
    });
    if stats.head != head {
        stats.head = head;
    }
    let aim = watched.and_then(
        |(entity, ((_, actor, _, wish), (_, offsets), _, joints, _))| {
            aim_check(
                memories.held.get(&entity)?,
                (actor.slot, wish, offsets),
                joints?,
            )
        },
    );
    if stats.aim != aim {
        stats.aim = aim;
    }
    if stats.locomotion != locomotion {
        stats.locomotion = locomotion;
    }
    let mut missing: Vec<String> = memories
        .held
        .values()
        .flat_map(|memory| {
            memory
                .missing
                .iter()
                .map(|clip| format!("slot {}: {clip}", memory.stats.slot))
        })
        .collect();
    missing.sort_unstable();
    let count = missing.len() as u32;
    missing.truncate(8);
    if stats.missing_clips != missing || stats.missing_count != count {
        stats.missing_clips = missing;
        stats.missing_count = count;
    }
}
