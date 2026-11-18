use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::math::{clamp, max, min, sin_cos};
use engine_core::{Error, ErrorCode, Refusals};

use super::locomotion::{Moved, moved};
use super::{
    ActorStats, ClipWeight, HeadStats, LocomotionStats, Memories, PUPPET_PART, PuppetBrick,
    PuppetStats,
};
use crate::abilities::{AbilityBook, AbilityId, Delivery, Slot};
use crate::names::{HEAD, LEFT_HAND, RIGHT_HAND, SPINE};
use crate::play::actors::{Actor, Alive, Body, Motion, Wish};
use crate::play::arms::Arms;
use crate::play::camera::TestCamera;
use crate::play::events::arms::{ArmsEvents, Weapon};
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
    BaseClip, DEVICE, OneShot, PuppetBook, PuppetFile, PuppetTable, Task, weapon_key,
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
    PlantIn,
    PlantLoop,
    Defuse,
    Dead(From),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum From {
    Front,
    Back,
    Side,
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
    feet_before: Option<[Vec3; 2]>,
    slide: f32,
    moving: Option<Moved>,
    pub(super) carry: Option<Carry>,
    pub(super) stats: ActorStats,
}

impl Memory {
    fn locomotion(&self) -> Option<LocomotionStats> {
        let moved = self.moving.as_ref()?;
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
        Some(LocomotionStats {
            slot: self.stats.slot,
            clips,
            rate: moved.rate,
            stride: moved.stride.scale,
            hip_turn: moved.hip_turn,
            speed_mps: moved.speed,
            planted_mps: moved.planted,
            planned_slide_mps: moved.planned_slide,
            slide_mps: self.slide,
        })
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
            feet_before: None,
            slide: 0.0,
            moving: None,
            carry: None,
            stats: ActorStats::default(),
        }
    }
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
    let eye_m = eye.y - body.feet.y;
    let body_eye = body.eye.y - body.feet.y;
    Some(HeadStats {
        slot,
        head_top_m: head_top,
        height_m: height,
        top_gap_m: head_top - height,
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
    }
}

struct Facts<'a> {
    book: &'a PuppetBook,
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

fn cast_shot<'a>(
    table: &'a PuppetTable,
    facts: &Facts<'_>,
    actor: &Actor,
    slot: AbilitySlot,
) -> &'a OneShot {
    let index = match slot {
        AbilitySlot::C => Slot::C,
        AbilitySlot::Q => Slot::Q,
        AbilitySlot::E => Slot::E,
        AbilitySlot::X => Slot::X,
    };
    let delivery = facts.abilities.and_then(|book| {
        let character = book.character(&actor.character)?;
        Some(
            book.ability(AbilityId {
                character,
                slot: index,
            })
            .delivery,
        )
    });
    match delivery {
        Some(Delivery::Throw(_)) => &table.casts.throw,
        Some(Delivery::Place(_)) => &table.casts.place,
        Some(Delivery::Caster) | None => &table.casts.caster,
    }
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
        if memory.weapon.is_some() && weapon.is_some() {
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
        (BodyPart::Body, From::Front) => hits.front.clone(),
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
            memory.fallen = from;
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
    let mut wanted: Option<Upper> = None;
    for cast in facts.powers.iter().flat_map(|powers| &powers.casts) {
        if cast.slot == actor.slot {
            wanted = Some(shot(cast_shot(table, facts, actor, cast.ability), false));
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
        .filter(|landing| landing.slot == actor.slot && landing.hard)
        .map(|landing| landing.fall_speed)
        .reduce(max);
    if let Some(speed) = landed.filter(|_| body.grounded) {
        let hard = table
            .land_hard
            .as_ref()
            .filter(|hard| speed >= hard.from_mps)
            .map_or(&table.land, |hard| &hard.shot);
        wanted = Some(shot(hard, false));
    }
    for fired in &facts.arms.shots {
        if fired.shooter == actor.slot && fired.weapon == Weapon::Melee {
            let swing = if fired.heavy {
                &table.melee_heavy
            } else {
                &table.melee
            };
            wanted = Some(shot(swing, false));
        }
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

fn planted(memory: &Memory, joints: Option<&FigureJoints>) -> Base {
    let started =
        joints.is_some_and(|joints| joints.base_cue == memory.base_cue && joints.base_done);
    match memory.base {
        Base::PlantIn if started => Base::PlantLoop,
        Base::PlantLoop => Base::PlantLoop,
        Base::PlantIn
        | Base::Idle
        | Base::Walk
        | Base::Run
        | Base::Turn
        | Base::Rise
        | Base::Fall
        | Base::CrouchIdle
        | Base::CrouchWalk
        | Base::Defuse
        | Base::Dead(_) => Base::PlantIn,
    }
}

fn base_of(
    memory: &Memory,
    table: &PuppetTable,
    (body, joints): (&Body, Option<&FigureJoints>),
) -> Base {
    if memory.defusing {
        return Base::Defuse;
    }
    if memory.planting {
        return planted(memory, joints);
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
        Base::PlantIn => of(&table.plant.start, false),
        Base::PlantLoop => of(&table.plant.hold, true),
        Base::Defuse => of(&table.defuse, true),
        Base::Dead(from) => (
            match from {
                From::Front => death.from_front.clone(),
                From::Back => death.from_back.clone(),
                From::Side => death.from_side.clone(),
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
        | Base::PlantIn
        | Base::PlantLoop
        | Base::Defuse => 1.0,
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
    pitch: f32,
) -> Shaped {
    let moving = matches!(base, Base::Walk | Base::Run | Base::CrouchWalk)
        .then(|| moved(&table.locomotion, base == Base::CrouchWalk, (body, wish)))
        .flatten();
    let (clip, fade_seconds, looping) = match &moving {
        Some(moved) => (moved.clip.clone(), table.locomotion.fade_seconds, true),
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
        | Base::PlantIn
        | Base::PlantLoop
        | Base::Defuse
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
    drop: f32,
    weight: f32,
    gun: Gun,
}

fn reload_weight(progress: f32, share: f32) -> f32 {
    let share = max(share, 0.01);
    let ease = |value: f32| {
        let value = clamp(value, 0.0, 1.0);
        value * value * (3.0 - 2.0 * value)
    };
    min(ease(progress / share), ease((1.0 - progress) / share))
}

fn held_pose(book: &PuppetBook, weapon: &'static str, given: &Pose) -> Option<Posed> {
    let table = &book.table;
    let pose = table.holds.get(table.weapons.get(weapon)?)?;
    let Pose {
        pitch,
        drop,
        weight,
        ..
    } = *given;
    let lowered = max(given.gun.dip, given.gun.aim * table.aiming.lower);
    let pitch = pitch - table.switch.dip_radians * lowered;
    let drop = drop + table.switch.dip_m * lowered;
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
    let away = given.gun.reload.map_or(0.0, |progress| {
        reload_weight(progress, table.reload_hand.share)
    });
    let hip = Vec3::from_array(table.reload_hand.to) - Vec3::new(0.0, drop, 0.0);
    let raised = Vec3::from_array(table.aiming.hand) - Vec3::new(0.0, drop, 0.0);
    let left = pose.left.zip(markers.grip_l).map(|(grip, at)| HandTarget {
        position: placed(grip.wrist(at))
            .lerp(hip, away)
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
        };
        (weapon, offset, markers.muzzle.map(placed), turn * Vec3::Z)
    });
    Some(Posed {
        hold: Some(Hold {
            space: HoldSpace::Figure,
            weight,
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
        memory.upper = None;
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
            | Base::PlantIn
            | Base::PlantLoop
            | Base::Defuse => Base::Dead(memory.fallen),
        }
    };
    if base != memory.base && matches!(base, Base::Rise | Base::Dead(_) | Base::PlantIn) {
        memory.base_cue = memory.base_cue.wrapping_add(1);
    }
    memory.base = base;
    base
}

fn carried(
    posed: Option<&Posed>,
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
        offset: held.map_or(
            Offset {
                translation: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            |(_, offset, ..)| offset,
        ),
        muzzle: held.and_then(|(_, _, muzzle, direction)| {
            muzzle.map(|muzzle| (body.feet + facing * muzzle, facing * direction))
        }),
        shown: may_show && held.is_some(),
    }
}

fn upper_layer(memory: &Memory) -> Option<UpperLayer> {
    memory.upper.as_ref().map(|upper| UpperLayer {
        clip: upper.clip.clone(),
        weight: upper.weight,
        looping: false,
        from: upper.from.clone(),
        feather: upper.feather,
        speed: upper.speed,
        cue: upper.cue,
        window: upper.window,
    })
}

fn shown_body(
    spawner: &mut Spawner<PuppetBrick>,
    entity: Entity,
    node: Option<&Node>,
    viewed: bool,
) {
    if let Some(node) = node
        && node.visible == viewed
    {
        spawner.insert(
            entity,
            Node {
                visible: !viewed,
                ..*node
            },
        );
    }
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
            .is_some_and(|upper| pinned.contains(&upper.clip)))
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
        (Has<Alive>, Option<&'static Node>),
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

fn held_blend(memory: &mut Memory, table: &PuppetTable, delta: f32) {
    let free = memory.upper.is_none() && memory.task().is_none();
    let goal = f32::from(u8::from(free));
    let blend = delta / max(table.hold_blend_seconds, delta);
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
    for ((entity, actor, body, wish), (arms, offsets), (alive, node), joints, play) in &mut actors {
        seen.push(entity);
        let memory = memories.held.entry(entity).or_default();
        if joints.is_none() {
            spawner.insert(entity, joints_named());
        }
        let viewed = watched == Some(entity);
        shown_body(&mut spawner, entity, node, viewed);
        let look_yaw = if alive {
            feet_follow(memory, table, (body, wish), facts.delta)
        } else {
            0.0
        };
        let base = stepped(memory, &facts, (actor, body, wish), (alive, joints));
        let pitch = wish.pitch_radians + offsets.map_or(0.0, |offsets| offsets.pitch_radians());
        let shape = shaped(table, base, (alive, body, wish), pitch);
        let (clip, rate, crouch, aim) = (shape.clip.clone(), shape.rate, shape.crouch, shape.aim);
        memory.slide = slid(memory, joints, facts.delta);
        let weapon = tasked(memory, &book, arms.filter(|_| alive));
        held_blend(memory, table, facts.delta);
        let pose = Pose {
            pitch: aim,
            drop: crouch,
            weight: memory.hold,
            gun: gun_state(memory, &facts, (weapon, arms, actor.slot)),
        };
        let (reload, dip) = (pose.gun.reload, pose.gun.dip);
        let posed = weapon.and_then(|weapon| held_pose(&book, weapon, &pose));
        judged(&mut refusals, entity, weapon, posed.is_some());
        let hold = posed.as_ref().and_then(|posed| posed.hold.clone());
        memory.carry = Some(carried(
            posed.as_ref(),
            (actor.slot, entity),
            (body, wish),
            alive && !viewed,
        ));
        let wanted = figure_play(table, memory, &shape, (look_yaw, hold));
        memory.moving = shape.moving;
        memory.stats = ActorStats {
            slot: actor.slot,
            base: clip.clone(),
            rate,
            upper: memory.upper.as_ref().map(|upper| upper.clip.clone()),
            hold: if wanted.hold.is_some() {
                memory.hold
            } else {
                0.0
            },
            crouch,
            aim,
            weapon: weapon.map(str::to_owned),
            dead: !alive,
            look_yaw,
            step_m: joints
                .and_then(|joints| joints.steps.get(&clip))
                .copied()
                .unwrap_or_default(),
            reload,
            dip,
        };
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
    let locomotion = followed
        .and_then(|entity| memories.held.get(&entity))
        .and_then(Memory::locomotion);
    let head = followed
        .and_then(|entity| actors.get(entity).ok())
        .and_then(|((_, actor, body, _), _, _, joints, _)| {
            head_check(numbers, (actor.slot, body), joints)
        });
    if stats.head != head {
        stats.head = head;
    }
    if stats.locomotion != locomotion {
        stats.locomotion = locomotion;
    }
}
