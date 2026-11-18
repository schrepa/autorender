use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::math::{clamp, max, min, sin_cos};
use engine_core::{Error, ErrorCode, Refusals};

use super::{ActorStats, Memories, PUPPET_PART, PuppetBrick, PuppetStats};
use crate::abilities::{AbilityBook, AbilityId, Delivery, Slot};
use crate::names::{HEAD, LEFT_HAND, RIGHT_HAND, SPINE};
use crate::play::actors::{Actor, Alive, Body, Motion, Wish};
use crate::play::arms::Arms;
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
    BaseClip, CrouchClips, OneShot, PuppetBook, PuppetFile, PuppetTable, weapon_key,
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
    pub(super) carry: Option<Carry>,
    pub(super) stats: ActorStats,
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
            carry: None,
            stats: ActorStats::default(),
        }
    }
}

const JOINT_NAMES: [&str; 7] = [
    HEAD,
    SPINE,
    HIP,
    LEFT_LEG[2],
    RIGHT_LEG[2],
    RIGHT_HAND,
    LEFT_HAND,
];

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
        .any(|landing| landing.slot == actor.slot && landing.hard);
    if landed && body.grounded {
        wanted = Some(shot(&table.land, false));
    }
    for fired in &facts.arms.shots {
        if fired.shooter == actor.slot && fired.weapon == Weapon::Melee {
            wanted = Some(shot(&table.melee, false));
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
    let crouching = table
        .crouch
        .chosen()
        .is_some_and(|clips| body.crouch >= clips.from);
    let speed = Vec3::new(body.velocity.x, 0.0, body.velocity.z).length();
    if speed >= table.still_below_mps {
        return match (crouching, body.motion) {
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
    let crouched = |pick: fn(&CrouchClips) -> &BaseClip, standing: &BaseClip| {
        table
            .crouch
            .chosen()
            .map_or_else(|| of(standing, true), |clips| of(pick(clips), true))
    };
    let death = &table.death;
    match base {
        Base::Idle => of(&table.idle, true),
        Base::Walk => of(&table.walk, true),
        Base::Run => of(&table.run, true),
        Base::Turn => of(&table.turn, true),
        Base::Rise => of(&table.rise, false),
        Base::Fall => of(&table.fall, true),
        Base::CrouchIdle => crouched(|clips| &clips.idle, &table.idle),
        Base::CrouchWalk => crouched(|clips| &clips.walk, &table.walk),
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

fn rate_of(
    table: &PuppetTable,
    base: Base,
    clip: &str,
    body: &Body,
    joints: Option<&FigureJoints>,
) -> f32 {
    match base {
        Base::Walk | Base::Run | Base::CrouchWalk => {
            let speed = Vec3::new(body.velocity.x, 0.0, body.velocity.z).length();
            let stride = joints.and_then(|joints| joints.strides.get(clip)).copied();
            match stride {
                Some(stride) if stride > table.rate.stride_floor_mps => {
                    clamp(speed / stride, table.rate.min, table.rate.max)
                }
                Some(_) | None => 1.0,
            }
        }
        Base::Dead(_) => table.death.speed,
        Base::Idle
        | Base::Turn
        | Base::Rise
        | Base::Fall
        | Base::CrouchIdle
        | Base::PlantIn
        | Base::PlantLoop
        | Base::Defuse => 1.0,
    }
}

pub(super) struct Posed {
    pub(super) hold: Option<Hold>,
    pub(super) carry: Option<(&'static str, Offset, Vec3, Vec3)>,
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
        let muzzle = markers.muzzle.unwrap_or(Vec3::ZERO);
        (weapon, offset, placed(muzzle), turn * Vec3::Z)
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
        muzzle: held
            .map(|(_, _, muzzle, direction)| (body.feet + facing * muzzle, facing * direction)),
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

fn stance(table: &PuppetTable, base: Base, (alive, body): (bool, &Body), pitch: f32) -> (f32, f32) {
    let clipped =
        matches!(base, Base::CrouchIdle | Base::CrouchWalk) && table.crouch.chosen().is_some();
    match (alive, clipped) {
        (false, _) => (0.0, 0.0),
        (true, true) => (0.0, pitch * table.aim.share),
        (true, false) => (body.crouch * table.crouch.drop_m, pitch * table.aim.share),
    }
}

fn figure_play(
    table: &PuppetTable,
    memory: &Memory,
    (clip, rate, looping, fade_seconds): (String, f32, bool, f32),
    (aim, look_yaw, crouch, hold): (f32, f32, f32, Option<Hold>),
) -> FigurePlay {
    let pin_feet = table
        .pin_feet
        .clips
        .contains(&clip)
        .then_some(table.pin_feet.contact_m);
    FigurePlay {
        base: BaseLayer {
            clip,
            speed: rate,
            looping,
            fade_seconds,
            cue: memory.base_cue,
            fit_seconds: None,
        },
        upper: upper_layer(memory),
        aim: Some(Aim {
            pitch: aim,
            chain: table.aim.chain.clone(),
            yaw: look_yaw,
            hips: Some(table.look.hips.clone()),
        }),
        hold,
        crouch: Some(crouch),
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
    Option<Res<'w, Watched>>,
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

pub(super) fn animate(
    (
        (fixed, numbers),
        (arms_events, round_events, powers, body_events, held_powers),
        files,
        watched,
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
    let watched = watched.and_then(|watched| watched.entity);
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
        let (clip, fade_seconds, looping) = base_clip(table, base);
        let rate = rate_of(table, base, &clip, body, joints);
        let pitch = wish.pitch_radians + offsets.map_or(0.0, |offsets| offsets.pitch_radians());
        let (crouch, aim) = stance(table, base, (alive, body), pitch);
        let weapon = arms.filter(|_| alive).map(|arms| weapon_key(&arms.loadout));
        let goal = if memory.upper.is_some() { 0.0 } else { 1.0 };
        let blend = facts.delta / max(table.hold_blend_seconds, facts.delta);
        memory.hold = toward(memory.hold, goal, blend);
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
        let wanted = figure_play(
            table,
            memory,
            (clip.clone(), rate, looping, fade_seconds),
            (aim, look_yaw, crouch, hold),
        );
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
    let actors: Vec<ActorStats> = listed.into_iter().map(|(_, stats)| stats).collect();
    if stats.actors != actors {
        stats.actors = actors;
    }
}
