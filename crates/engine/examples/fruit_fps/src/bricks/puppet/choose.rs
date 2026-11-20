use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::*;
use engine_core::math::{clamp, max, sin_cos};
use engine_core::{Assets, Drawn, Error, ErrorCode, Refusals, Stats};
use engine_math::wrap_angle;

use super::heads::Crowns;
use super::{
    ActorStats, AimStats, ClipWeight, EyeFitStats, HeadStats, LocomotionStats, Memories,
    PUPPET_PART, PuppetBrick, PuppetStats,
};
use crate::names::RIGHT_HAND;
use engine_rig::Skeleton;
use crate::abilities::{AbilityBook, AbilityId, AbilitySpec, Delivery};
use crate::play::actors::{Actor, Alive, Body, Motion, Wish};
use crate::play::arms::Arms;
use crate::play::camera::TestCamera;
use crate::play::events::arms::{ArmsEvents, Handling, Weapon};
use crate::play::events::body::BodyEvents;
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::figures::biped;
use crate::play::gait::{Gaited, gait};
use crate::play::numbers::{GameNumbers, Numbers};
use crate::play::powers::Powers;
use crate::play::puppet::{
    BaseClip, DEVICE, HeadCheck, InHand, OneShot, Puppet, PuppetBook, PuppetTable, Task,
    UpperTable, leave_tick, weapon_key,
};
use crate::play::skinned::{
    Aim, BaseLayer, Chest, ClipNo, Figure, FigureClock, FigureFit, FigureJoints, FigureLimbs,
    FigurePlay, HandTarget, Hold, HoldSpace, JointAt, JointNo, Offset, POSED_JOINTS, StepJoints,
    StepPose, UpperLayer, posed,
};
use crate::play::turns::{about_x, about_y, approach};
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
    clip: ClipNo,
    from: JointNo,
    feather: u32,
    weight: f32,
    speed: f32,
    cue: u32,
    lasting: bool,
    window: Option<[f32; 2]>,
    looping: bool,
    fit_seconds: Option<f32>,
    added: Option<Added>,
    next: Option<ClipNo>,
    ease: Option<f32>,
    whole: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct Added {
    played_from: Option<JointNo>,
    fade_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Carry {
    pub(super) slot: u32,
    pub(super) actor: Entity,
    pub(super) hand: JointNo,
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
    look: f32,
    gaze: [f32; 2],
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
    moving: Option<Gaited>,
    bound: Option<(Arc<PuppetBook>, Bound)>,
    missing: Vec<String>,
    pub(super) carry: Option<Carry>,
    pub(super) hand: Option<InHand>,
}

impl Memory {
    fn locomotion(&self, (body, figure): (&Body, &Figure), shown: &ActorStats) -> LocomotionStats {
        let name = |clip: ClipNo| figure.clip_name(clip).unwrap_or_default().to_owned();
        let Some(moved) = self.moving.as_ref() else {
            let speed = Vec3::new(body.velocity.x, 0.0, body.velocity.z).length();
            return LocomotionStats {
                slot: shown.slot,
                state: "standing".to_owned(),
                clips: vec![ClipWeight {
                    clip: shown.base.clone(),
                    weight: 1.0,
                }],
                rate: shown.rate,
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
            clip: name(moved.clip),
            weight: held,
        })
        .chain(moved.blend.iter().map(|blended| ClipWeight {
            clip: name(blended.clip),
            weight: blended.weight,
        }))
        .collect();
        LocomotionStats {
            slot: shown.slot,
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
            look: 1.0,
            gaze: [0.0; 2],
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
            bound: None,
            missing: Vec::new(),
            carry: None,
            hand: None,
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

#[derive(Clone, Debug, Default)]
struct Bound {
    gaits: Vec<Option<ClipNo>>,
    idle: Option<ClipNo>,
    turn: Option<ClipNo>,
    rise: Option<ClipNo>,
    fall: Option<ClipNo>,
    crouch_idle: Option<ClipNo>,
    kneel: Option<ClipNo>,
    plant: Option<ClipNo>,
    defuse: Option<ClipNo>,
    stand: Option<ClipNo>,
    land: Option<ClipNo>,
    victory: Option<ClipNo>,
    death: [Option<ClipNo>; 4],
    hits: [Option<ClipNo>; 5],
    fire: BTreeMap<String, ClipNo>,
    reload: BTreeMap<String, ClipNo>,
    ready: BTreeMap<String, ClipNo>,
    release: BTreeMap<String, ClipNo>,
    roll: BTreeMap<String, ClipNo>,
    draw: BTreeMap<String, ClipNo>,
    hold: Option<ClipNo>,
    light: Vec<Option<ClipNo>>,
    heavy: Option<ClipNo>,
    casts: BTreeMap<String, BTreeMap<String, ClipNo>>,
    played: Vec<ClipNo>,
    pinned: Vec<ClipNo>,
    upper_from: Option<JointNo>,
    played_from: Option<JointNo>,
    melee_from: Option<JointNo>,
    draw_from: Option<JointNo>,
    casts_from: Option<JointNo>,
    still_from: BTreeMap<String, JointNo>,
    hits_from: Option<JointNo>,
    land_from: Option<JointNo>,
    victory_from: Option<JointNo>,
    hips: Option<JointNo>,
    socket: Option<JointNo>,
    hand: Option<JointNo>,
    chain: Vec<JointNo>,
    chest: Option<Chest>,
    asked: Vec<Option<JointNo>>,
}

fn table_joints(table: &PuppetTable) -> impl Iterator<Item = &str> {
    let casts = &table.casts;
    [
        &table.upper.from,
        &table.upper.played_from,
        &table.melee.from,
        &table.draw.from,
        &casts.from,
        &table.hits.from,
        &table.land.from,
        &table.victory.from,
        &table.look.hips,
        &table.upper.throw.hand.joint,
    ]
    .into_iter()
    .chain(casts.clips.values().filter_map(|cast| cast.still_from.as_ref()))
    .chain(&table.aim.chain)
    .chain(table.aim.chest.as_ref().map(|chest| &chest.joint))
    .map(String::as_str)
}

fn bound(table: &PuppetTable, figure: &Figure, skeleton: &Skeleton) -> (Bound, Vec<String>) {
    let clip = |name: &str| figure.clip(name);
    let joint = |name: &str| skeleton.find(name).map(JointNo);
    let each = |map: &BTreeMap<String, String>| -> BTreeMap<String, ClipNo> {
        (map.iter())
            .filter_map(|(key, name)| Some((key.clone(), clip(name)?)))
            .collect()
    };
    let casts = &table.casts;
    let (upper, throw, death, hits) = (&table.upper, &table.upper.throw, &table.death, &table.hits);
    let made = Bound {
        gaits: (table.locomotion.gaits.iter()).map(|gait| clip(&gait.clip)).collect(),
        idle: clip(&table.idle.clip),
        turn: clip(&table.turn.clip),
        rise: clip(&table.rise.clip),
        fall: clip(&table.fall.clip),
        crouch_idle: clip(&table.crouch.clips.idle.clip),
        kneel: clip(&table.device.kneel.clip),
        plant: clip(&table.device.plant.clip),
        defuse: clip(&table.device.defuse.clip),
        stand: clip(&table.device.stand.clip),
        land: clip(&table.land.clip),
        victory: clip(&table.victory.clip),
        death: [&death.from_front, &death.from_back, &death.from_side, &death.headshot]
            .map(|name| clip(name)),
        hits: [&hits.head, &hits.low, &hits.front, &hits.back, &hits.side].map(|name| clip(name)),
        fire: each(&upper.fire),
        reload: each(&upper.reload),
        ready: each(&throw.ready),
        release: each(&throw.release),
        roll: each(&throw.roll),
        draw: each(&table.draw.clips),
        hold: clip(&throw.hold),
        light: table.melee.light.iter().map(|name| clip(name)).collect(),
        heavy: clip(&table.melee.heavy),
        casts: (casts.clips.iter())
            .map(|(key, cast)| {
                let named = (casts.suffix.iter())
                    .filter_map(|(weapon, suffix)| {
                        Some((weapon.clone(), clip(&format!("{}{suffix}", cast.clip))?))
                    })
                    .collect();
                (key.clone(), named)
            })
            .collect(),
        played: upper.played.iter().filter_map(|name| clip(name)).collect(),
        pinned: (table.pin_feet.clips.iter()).filter_map(|name| clip(name)).collect(),
        upper_from: joint(&upper.from),
        played_from: joint(&upper.played_from),
        melee_from: joint(&table.melee.from),
        draw_from: joint(&table.draw.from),
        casts_from: joint(&casts.from),
        still_from: (casts.clips.iter())
            .filter_map(|(key, cast)| Some((key.clone(), joint(cast.still_from.as_ref()?)?)))
            .collect(),
        hits_from: joint(&hits.from),
        land_from: joint(&table.land.from),
        victory_from: joint(&table.victory.from),
        hips: joint(&table.look.hips),
        socket: joint(&throw.hand.joint),
        hand: joint(RIGHT_HAND),
        chain: table.aim.chain.iter().filter_map(|name| joint(name)).collect(),
        chest: (table.aim.chest.as_ref()).and_then(|chest| {
            Some(Chest {
                joint: joint(&chest.joint)?,
                reference: clip(&chest.reference)?,
            })
        }),
        asked: POSED_JOINTS.map(joint).to_vec(),
    };
    let mut missing: Vec<String> = (named_by(table).into_iter())
        .filter(|name| clip(name).is_none())
        .collect();
    missing.extend(
        table_joints(table)
            .filter(|name| joint(name).is_none())
            .map(|name| format!("joint {name}")),
    );
    (made, missing)
}

fn bind(
    memory: &mut Memory,
    book: &Arc<PuppetBook>,
    (figure, assets): (&Ref<Figure>, &Assets),
) -> Option<(Arc<PuppetBook>, Bound)> {
    let fresh = memory
        .bound
        .as_ref()
        .is_some_and(|(held, _)| Arc::ptr_eq(held, book));
    if !fresh || figure.is_changed() {
        memory.bound = None;
        let skeleton = assets.get(figure.skeleton).ok()?;
        let (made, missing) = bound(&book.table, figure, skeleton);
        memory.missing = missing;
        return Some((Arc::clone(book), made));
    }
    memory.bound.take()
}

fn written(
    spawner: &mut Spawner<PuppetBrick>,
    entity: Entity,
    play: Option<Mut<FigurePlay>>,
    wanted: Option<FigurePlay>,
) {
    match (play, wanted) {
        (Some(mut play), Some(wanted)) => {
            play.set_if_neq(wanted);
        }
        (None, Some(wanted)) => spawner.insert(entity, wanted),
        (_, None) => {}
    }
}

fn eye_fits(actors: &Actors, numbers: Option<&GameNumbers>) -> Vec<EyeFitStats> {
    let Some(eye) = numbers.map(|numbers| numbers.player.eye_height_m) else {
        return Vec::new();
    };
    let mut fits: Vec<EyeFitStats> = Vec::new();
    for ((_, actor, ..), _, _, (_, joints), _) in actors.iter() {
        let Some(joints) = joints.filter(|joints| joints.fit_rest_m.is_some()) else {
            continue;
        };
        if fits.iter().any(|known| known.character == actor.character) {
            continue;
        }
        let idle = joints.fit_rest_m.unwrap_or_default();
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
    joints: &StepPose,
) -> Option<AimStats> {
    let carry = memory
        .carry
        .as_ref()
        .filter(|carry| !carry.weapon.is_empty())?;
    let hand = joints.joints.get(posed::RIGHT_HAND).filter(|joint| joint.found)?;
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
    (joints, fit): (Option<&StepPose>, f32),
    crowned: impl FnOnce(&JointAt, f32) -> Option<Vec3>,
) -> Option<HeadStats> {
    let player = &numbers?.player;
    let joints = joints?;
    let top = joints.top?;
    let eye_at = |at: usize| joints.joints.get(at).filter(|joint| joint.found);
    let eye = (eye_at(posed::LEFT_EYE)?.position + eye_at(posed::RIGHT_EYE)?.position) * 0.5;
    let standing = player.moving.height_m;
    let height = standing + (player.moving.crouch_height_m - standing) * body.crouch;
    let head_top = top.y - body.feet.y;
    let crown = eye_at(posed::HEAD)
        .and_then(|head| crowned(head, fit))
        .map_or(head_top, |crown| crown.y - body.feet.y);
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

fn shot(one: &OneShot, (clip, from): (Option<ClipNo>, Option<JointNo>), lasting: bool) -> Option<Upper> {
    Some(Upper {
        clip: clip?,
        from: from?,
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
        whole: one.whole,
    })
}

fn layer_shot(table: &UpperTable, bound: &Bound, clip: Option<&ClipNo>) -> Option<Upper> {
    let clip = *clip?;
    Some(Upper {
        clip,
        from: bound.upper_from?,
        feather: table.feather,
        weight: table.weight,
        speed: 1.0,
        cue: 0,
        lasting: false,
        window: None,
        looping: false,
        fit_seconds: None,
        added: Some(Added {
            played_from: bound.played.contains(&clip).then_some(bound.played_from).flatten(),
            fade_seconds: table.fade_seconds,
        }),
        next: None,
        ease: None,
        whole: false,
    })
}

fn throws(upper: Option<&Upper>, bound: &Bound) -> bool {
    upper.is_some_and(|upper| {
        Some(upper.clip) == bound.hold || bound.ready.values().any(|clip| *clip == upper.clip)
    })
}

fn melee_upper(memory: &mut Memory, facts: &Facts<'_>, slot: u32) -> Option<Upper> {
    let melee = &facts.book.table.melee;
    let swing = facts.arms.swings.iter().rfind(|swing| swing.slot == slot)?;
    let bound = facts.bound;
    let clip = if swing.heavy {
        bound.heavy
    } else {
        let light = &bound.light;
        light.get(usize::from(swing.step) % light.len().max(1)).copied().flatten()
    }?;
    let from = bound.melee_from?;
    memory.strike = Some(swing.strike_tick.saturating_sub(facts.arms.tick) as f32 * facts.delta);
    Some(Upper {
        clip,
        from,
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
        whole: false,
    })
}

fn draw_upper(facts: &Facts<'_>, slot: u32) -> Option<Upper> {
    let draw = &facts.book.table.draw;
    let handled = facts
        .arms
        .handling
        .iter()
        .rfind(|handled| handled.slot == slot && handled.kind == Handling::Equip)?;
    let Weapon::Gun(kind) = handled.weapon else {
        return None;
    };
    let clip = *facts.bound.draw.get(handled.weapon.key())?;
    Some(Upper {
        clip,
        from: facts.bound.draw_from?,
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
        whole: false,
    })
}

fn drawn(bound: &Bound, weapon: Option<&str>) -> bool {
    weapon.is_some_and(|weapon| bound.draw.contains_key(weapon))
}

#[derive(Clone, Copy)]
struct Facts<'a> {
    book: &'a PuppetBook,
    bound: &'a Bound,
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
    let book = facts.abilities?;
    let character = book.character(&actor.character)?;
    Some(book.ability(AbilityId { character, slot }))
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
            let bound = facts.bound;
            let clips = if rolled { &bound.roll } else { &bound.release };
            let clip = memory.weapon.and_then(|weapon| clips.get(weapon));
            layer_shot(&table.upper, bound, clip)
        }
        Some(Delivery::Place(_) | Delivery::Caster) => {
            let casts = &table.casts;
            let row = table.view.hands.abilities.get(&spec?.name)?;
            let cast = casts.clips.get(&row.clip)?;
            let bound = facts.bound;
            let clip = *bound.casts.get(&row.clip)?.get(memory.weapon?)?;
            let still_from = bound.still_from.get(&row.clip).filter(|_| still);
            let from = still_from.copied().or(bound.casts_from)?;
            Some(Upper {
                clip,
                from,
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
                whole: cast.whole,
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

fn gun_upper(facts: &Facts<'_>, slot: u32) -> Option<Upper> {
    let table = &facts.book.table.upper;
    let mut wanted = None;
    for fired in &facts.arms.shots {
        if fired.shooter == slot && fired.weapon != Weapon::Melee {
            wanted = layer_shot(table, facts.bound, facts.bound.fire.get(fired.weapon.key()));
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
        let reload = facts.bound.reload.get(handled.weapon.key());
        wanted = layer_shot(table, facts.bound, reload).map(|upper| Upper {
            fit_seconds: seconds,
            ..upper
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
    let in_throw = throws(memory.upper.as_ref(), facts.bound);
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
    let ready = (memory.weapon).and_then(|weapon| facts.bound.ready.get(weapon));
    layer_shot(table, facts.bound, ready).map(|upper| Upper {
        next: facts.bound.hold,
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
    let index = ability.index();
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
        socket: facts.bound.socket?,
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
    let offset = wrap_angle(view - feet);
    if moving || offset.abs() > table.look.limit_radians {
        memory.turning = true;
    }
    let mut feet = feet;
    if memory.turning {
        let step = table.look.catch_up_radians_per_second * delta;
        let left = wrap_angle(view - feet);
        if left.abs() <= step {
            feet = view;
            memory.turning = moving;
        } else {
            feet += step * left.signum();
        }
    }
    memory.feet_yaw = Some(feet);
    wrap_angle(view - feet)
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
        if drawn(facts.bound, weapon) {
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
    memory.aiming = approach(memory.aiming, f32::from(u8::from(holding)), step);
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

fn hit_clip(bound: &Bound, part: BodyPart, from: From) -> Option<ClipNo> {
    let [head, low, front, back, side] = bound.hits;
    match (part, from) {
        (BodyPart::Head, _) => head,
        (BodyPart::Leg, _) => low,
        (BodyPart::Body, From::Front | From::Head) => front,
        (BodyPart::Body, From::Back) => back,
        (BodyPart::Body, From::Side) => side,
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
            let headshot = hit.killed && hit.part == BodyPart::Head && facts.bound.death[3].is_some();
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
            .is_some_and(|upper| facts.bound.reload.values().any(|clip| *clip == upper.clip))
    {
        memory.upper = None;
    }
    let bound = facts.bound;
    let mut wanted = draw_upper(facts, actor.slot);
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
            wanted = shot(&table.victory, (bound.victory, bound.victory_from), true).or(wanted);
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
        wanted = shot(&table.land, (bound.land, bound.land_from), false).or(wanted);
    }
    if let Some(gun) = gun_upper(facts, actor.slot) {
        wanted = Some(gun);
    }
    if let Some(swing) = melee_upper(memory, facts, actor.slot) {
        wanted = Some(swing);
    }
    if let Some((part, from)) = reaction
        && let (Some(clip), Some(joint)) = (hit_clip(bound, part, from), bound.hits_from)
    {
        let hits = &table.hits;
        wanted = Some(Upper {
            clip,
            from: joint,
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
            whole: false,
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

fn base_done(memory: &Memory, joints: Option<&FigureClock>) -> bool {
    joints.is_some_and(|joints| joints.base_cue == memory.base_cue && joints.base_done)
}

fn knelt(memory: &Memory, bound: &Bound, task: Task, joints: Option<&FigureClock>) -> Base {
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
    if kneeling && bound.kneel.is_some() {
        Base::Kneel(task)
    } else {
        Base::Busy(task)
    }
}

fn stood(
    memory: &Memory,
    (table, bound): (&PuppetTable, &Bound),
    (body, joints): (&Body, Option<&FigureClock>),
) -> bool {
    let still = body.grounded
        && Vec3::new(body.velocity.x, 0.0, body.velocity.z).length() < table.still_below_mps;
    let stands = bound.stand.is_some();
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
    (table, bound): (&PuppetTable, &Bound),
    (body, joints): (&Body, Option<&FigureClock>),
) -> Base {
    if let Some(task) = memory.task() {
        return knelt(memory, bound, task, joints);
    }
    if stood(memory, (table, bound), (body, joints)) {
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

fn base_clip(
    (table, bound): (&PuppetTable, &Bound),
    base: Base,
) -> (Option<ClipNo>, f32, bool, bool) {
    let of = |(clip, number): (&BaseClip, Option<ClipNo>), looping: bool| {
        (
            number,
            clip.fade_seconds,
            clip.looping.unwrap_or(looping),
            clip.whole,
        )
    };
    let crouched = |standing: (&BaseClip, Option<ClipNo>)| {
        table
            .crouch
            .chosen()
            .map_or_else(|| of(standing, true), |clips| of((&clips.idle, bound.crouch_idle), true))
    };
    let death = &table.death;
    match base {
        Base::Idle => of((&table.idle, bound.idle), true),
        Base::Turn => of((&table.turn, bound.turn), true),
        Base::Rise => of((&table.rise, bound.rise), false),
        Base::Fall => of((&table.fall, bound.fall), true),
        Base::CrouchIdle | Base::Walk | Base::Run | Base::CrouchWalk => {
            crouched((&table.idle, bound.idle))
        }
        Base::Kneel(_) => of((&table.device.kneel, bound.kneel), false),
        Base::Busy(Task::Planting) => of((&table.device.plant, bound.plant), true),
        Base::Busy(Task::Defusing) => of((&table.device.defuse, bound.defuse), true),
        Base::Stand => of((&table.device.stand, bound.stand), false),
        Base::Dead(from) => (
            bound.death[match from {
                From::Front => 0,
                From::Back => 1,
                From::Side => 2,
                From::Head => 3,
            }],
            death.fade_seconds,
            false,
            death.whole,
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
    clip: Option<ClipNo>,
    fade_seconds: f32,
    looping: bool,
    rate: f32,
    crouch: f32,
    aim: f32,
    whole: bool,
    moving: Option<Gaited>,
}

fn shaped(
    (table, bound): (&PuppetTable, &Bound),
    base: Base,
    (alive, body, wish): (bool, &Body, &Wish),
    pitch: f32,
) -> Shaped {
    let moving = matches!(base, Base::Walk | Base::Run | Base::CrouchWalk)
        .then(|| {
            let (sin, cos) = sin_cos(wish.yaw_radians);
            gait(
                &table.locomotion,
                base == Base::CrouchWalk,
                (body.velocity, Vec3::new(-sin, 0.0, -cos)),
                &bound.gaits,
            )
        })
        .flatten();
    let (clip, fade_seconds, looping, whole) = match &moving {
        Some(moved) => (Some(moved.clip), table.locomotion.fade_seconds, true, false),
        None if base.on_device() && base_clip((table, bound), base).0.is_none() => {
            base_clip((table, bound), Base::Idle)
        }
        None => base_clip((table, bound), base),
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
    let crouch = if alive && !posed_crouched {
        body.crouch * table.crouch.drop_m
    } else {
        0.0
    };
    Shaped {
        clip,
        fade_seconds,
        looping,
        rate,
        crouch,
        aim: pitch * table.aim.share,
        whole,
        moving,
    }
}

fn step_pose<'a>(
    (steps, stats): (&'a mut StepJoints, &mut Stats<PuppetStats>),
    (entity, play, joints): (Entity, Option<&FigurePlay>, Option<&FigureJoints>),
    (facts, slot, followed): (&Facts<'_>, u32, Option<Entity>),
) -> Option<&'a StepPose> {
    let asked = joints.map_or(&[][..], |joints| joints.asked.as_slice());
    if facts.arms.shots.iter().any(|shot| shot.shooter == slot) {
        steps.posed(entity, play, asked);
    }
    if followed == Some(entity) {
        stats.edit(|_| {
            steps.posed(entity, play, asked);
        });
    }
    steps.kept(entity)
}

fn shift_of(step: Option<&StepPose>, joints: Option<&FigureJoints>) -> Vec3 {
    (step.map(|step| step.hold_shift))
        .or(joints.map(|joints| joints.hold_shift))
        .unwrap_or(Vec3::ZERO)
}

fn slid(
    memory: &mut Memory,
    (drawn, step): (Option<&FigureJoints>, Option<&StepPose>),
    delta: f32,
) -> f32 {
    let joints = (step.map(|step| &step.joints[..])).or(drawn.map(|drawn| &drawn.joints[..]));
    let feet = joints.and_then(|joints| {
        let at = |index: usize| {
            joints
                .get(index)
                .filter(|joint| joint.found)
                .map(|joint| joint.position)
        };
        Some([at(posed::LEFT_FOOT)?, at(posed::RIGHT_FOOT)?])
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

fn held_pose(
    (book, bound): (&PuppetBook, &Bound),
    weapon: &'static str,
    given: &Pose,
) -> Option<Posed> {
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
    let turn = about_x(-(pitch * pose.pitch_share));
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
            chest: bound.chest,
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
    (alive, joints): (bool, Option<&FigureClock>),
) -> Base {
    if let Some(upper) = &memory.upper
        && !upper.lasting
        && joints.is_some_and(|joints| joints.upper_cue == Some(upper.cue) && joints.upper_done)
    {
        let next = upper.next.as_ref();
        let then = layer_shot(&facts.book.table.upper, facts.bound, next).map(|upper| Upper {
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
        base_of(memory, (&facts.book.table, facts.bound), (body, joints))
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
    (slot, actor, hand): (u32, Entity, JointNo),
    (body, wish): (&Body, &Wish),
    may_show: bool,
) -> Carry {
    let facing = about_y(wish.yaw_radians + std::f32::consts::PI);
    let held = posed.and_then(|posed| posed.carry);
    Carry {
        slot,
        actor,
        hand,
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
        clip: upper.clip,
        weight: upper.weight,
        looping: upper.looping,
        from: upper.from,
        feather: upper.feather,
        speed: upper.speed,
        cue: upper.cue,
        window: upper.window,
        fit_seconds: upper.fit_seconds,
        additive: upper.added.is_some(),
        played_from: upper
            .added
            .as_ref()
            .and_then(|added| added.played_from),
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

fn abilities_follow(memories: &mut Memories, numbers: &Numbers, ticks_per_second: u32) {
    let revision = numbers.revision();
    if memories
        .abilities
        .as_ref()
        .is_some_and(|(known, _)| *known == revision)
    {
        return;
    }
    let book = numbers
        .get()
        .and_then(|numbers| AbilityBook::parse(&numbers.whole, ticks_per_second).ok())
        .map(std::sync::Arc::new);
    memories.abilities = Some((revision, book));
}

fn figure_play(
    (table, bound): (&PuppetTable, &Bound),
    memory: &Memory,
    shaped: &Shaped,
    (look_yaw, hold): (f32, Option<Hold>),
) -> Option<FigurePlay> {
    let clip = shaped.clip?;
    let pinned = &bound.pinned;
    let pin_feet = (pinned.contains(&clip)
        || memory.upper.as_ref().is_some_and(|upper| {
            pinned.contains(&upper.clip) && Some(upper.from) == bound.hips
        }))
    .then_some(table.pin_feet.contact_m);
    let moving = shaped.moving.as_ref();
    Some(FigurePlay {
        base: BaseLayer {
            clip,
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
            chain: bound.chain.clone(),
            yaw: look_yaw - moving.map_or(0.0, |moved| moved.hip_turn),
            hips: bound.hips,
        }),
        hold,
        crouch: Some(shaped.crouch),
        pin_feet,
    })
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
    (Res<'w, Puppet>, Res<'w, Assets>),
    (Option<Res<'w, Watched>>, Option<Res<'w, TestCamera>>),
);

type Actors<'w, 's> = Query<
    'w,
    's,
    (
        (Entity, &'static Actor, &'static Body, &'static Wish),
        (Option<&'static Arms>, Option<&'static ViewOffsets>),
        (
            Has<Alive>,
            Option<&'static Node>,
            Ref<'static, Figure>,
            Option<&'static Drawn>,
            Option<&'static FigureLimbs>,
            Option<&'static FigureFit>,
        ),
        (Option<&'static FigureJoints>, Option<&'static FigureClock>),
        Option<&'static mut FigurePlay>,
    ),
    With<Figure>,
>;

type Writes<'w, 's> = (
    ResMut<'w, Memories>,
    Stats<'w, PuppetStats>,
    ResMut<'w, Refusals>,
    (Spawner<'w, 's, PuppetBrick>, ResMut<'w, StepJoints>),
);

type Stated<'a> = (Option<&'static str>, bool, Option<f32>, f32);

fn actor_stats(
    (memory, table, bound, figure): (&Memory, &PuppetTable, &Bound, &Figure),
    (actor, body, joints): (&Actor, &Body, Option<&FigureClock>),
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
        base: (shape.clip.and_then(|clip| figure.clip_name(clip)))
            .unwrap_or_default()
            .to_owned(),
        rate: shape.rate,
        upper: (memory.upper.as_ref())
            .and_then(|upper| figure.clip_name(upper.clip))
            .map(str::to_owned),
        upper_seconds: playing.map_or(0.0, |joints| joints.upper_seconds),
        upper_rate: playing.map_or(0.0, |joints| joints.upper_rate),
        upper_active: playing.is_some_and(|joints| !joints.upper_done),
        event_s: memory.upper.as_ref().and_then(|upper| {
            let melee = bound.heavy == Some(upper.clip) || bound.light.contains(&Some(upper.clip));
            let event = (memory.strike.filter(|_| melee))
                .or_else(|| cast_event(table, figure.clip_name(upper.clip)?))?;
            Some(event / max(upper.speed, 1.0e-3))
        }),
        hold: if holding { memory.hold } else { 0.0 },
        crouch: shape.crouch,
        aim: shape.aim,
        look_weight: memory.look,
        weapon: weapon.map(str::to_owned),
        dead: !alive,
        look_yaw,
        step_m: joints
            .zip(shape.clip)
            .and_then(|(joints, clip)| joints.steps.get(clip.0))
            .copied()
            .unwrap_or_default(),
        drift_m: joints
            .zip(shape.clip)
            .and_then(|(joints, clip)| joints.drifts.get(clip.0))
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
    memory.hold = approach(memory.hold, goal, blend);
}

fn pitch_of(memory: &Memory, alive: bool, (wish, offsets): (&Wish, Option<&ViewOffsets>)) -> f32 {
    if alive {
        wish.pitch_radians + offsets.map_or(0.0, ViewOffsets::pitch_radians)
    } else {
        memory.gaze[0]
    }
}

fn gazed(
    memory: &mut Memory,
    (table, shape): (&PuppetTable, &mut Shaped),
    (alive, pitch, look_yaw): (bool, f32, f32),
    delta: f32,
) -> f32 {
    if alive {
        memory.gaze = [pitch, look_yaw];
    }
    let owner = memory.upper.as_ref().filter(|upper| upper.whole);
    let goal = f32::from(u8::from(!shape.whole && owner.is_none()));
    let seconds = owner
        .and_then(|upper| upper.ease)
        .filter(|_| goal == 0.0)
        .unwrap_or(table.hold_blend_seconds);
    memory.look = approach(memory.look, goal, delta / max(seconds, delta));
    shape.aim *= memory.look;
    look_yaw * memory.look
}

fn equipped(
    spawner: &mut Spawner<PuppetBrick>,
    entity: Entity,
    (joints, limbs): (Option<&FigureJoints>, bool),
    (held, fit, asked): (Option<&FigureFit>, Option<&FigureFit>, &[Option<JointNo>]),
) {
    if joints.is_none_or(|joints| joints.asked != asked) {
        let joints = FigureJoints {
            asked: asked.to_vec(),
            ..FigureJoints::default()
        };
        spawner.insert(entity, joints);
    }
    if !limbs {
        spawner.insert(entity, biped());
    }
    if held != fit {
        match fit {
            Some(fit) => spawner.insert(entity, fit.clone()),
            None => spawner.remove::<FigureFit>(entity),
        }
    }
}

pub(super) fn animate(
    (
        (fixed, numbers),
        (arms_events, round_events, powers, body_events, held_powers),
        (puppet, assets),
        (watched, test),
    ): Reads,
    mut actors: Actors,
    (mut memories, mut stats, mut refusals, (mut spawner, mut steps)): Writes,
    mut crowns: Local<Crowns>,
) {
    let Some(book) = puppet.book.clone() else {
        let refusal = puppet.refusal.clone().unwrap_or_else(|| {
            "no prepared `puppet` component (scene entity `game`)".to_owned()
        });
        stats.edit(|stats| stats.refusal = Some(refusal));
        return;
    };
    stats.edit(|stats| stats.refusal = None);
    let followed = watched.and_then(|watched| watched.entity);
    let watched = followed.filter(|_| !test.as_deref().is_some_and(|test| test.outside));
    let feet: BTreeMap<u32, Vec3> = actors
        .iter()
        .map(|((_, actor, body, _), ..)| (actor.slot, body.feet))
        .collect();
    abilities_follow(&mut memories, &numbers, fixed.rate);
    let abilities = memories
        .abilities
        .as_ref()
        .and_then(|(_, book)| book.clone());
    let unbound = Bound::default();
    let shared = Facts {
        book: &book,
        bound: &unbound,
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
    let fit = puppet.eye_fit(&numbers);
    let mut seen = Vec::new();
    stats.edit(|stats| stats.actors.clear());
    for (
        (entity, actor, body, wish),
        (arms, offsets),
        (alive, _, figure, _, limbs, held_fit),
        (joints, clock),
        play,
    ) in &mut actors
    {
        seen.push(entity);
        let memory = memories.held.entry(entity).or_default();
        let Some((bound_for, bound)) = bind(memory, &book, (&figure, &assets)) else {
            continue;
        };
        let facts = Facts {
            bound: &bound,
            ..shared
        };
        equipped(
            &mut spawner,
            entity,
            (joints, limbs.is_some()),
            (held_fit, fit.as_ref(), &bound.asked),
        );
        let viewed = watched == Some(entity);
        let look_yaw = if alive {
            feet_follow(memory, table, (body, wish), facts.delta)
        } else {
            memory.gaze[1]
        };
        let base = stepped(memory, &facts, (actor, body, wish), (alive, clock));
        memory.hand = alive
            .then(|| hand_of(memory.hand.take(), &facts, (entity, actor)))
            .flatten()
            .filter(|_| !viewed);
        let pitch = pitch_of(memory, alive, (wish, offsets));
        let mut shape = shaped((table, &bound), base, (alive, body, wish), pitch);
        let asked = (entity, play.as_deref(), joints);
        let step = step_pose((&mut steps, &mut stats), asked, (&facts, actor.slot, followed));
        memory.slide = slid(memory, (joints, step), facts.delta);
        let weapon = tasked(memory, &book, arms.filter(|_| alive));
        held_blend(memory, table, facts.delta);
        let gaze = (alive, pitch, look_yaw);
        let look_yaw = gazed(memory, (table, &mut shape), gaze, facts.delta);
        let pose = Pose {
            pitch: shape.aim,
            weight: memory.hold,
            gun: gun_state(memory, &facts, (weapon, arms, actor.slot)),
            played: left_played(memory),
        };
        let (reload, dip) = (pose.gun.reload, pose.gun.dip);
        let posed = weapon.and_then(|weapon| held_pose((&book, &bound), weapon, &pose));
        judged(&mut refusals, entity, weapon, posed.is_some());
        let hold = posed.as_ref().and_then(|posed| posed.hold);
        let fitted = (shift_of(step, joints), clock.map_or(1.0, |clock| clock.fit));
        memory.carry = bound.hand.map(|hand| {
            let shown = alive && !viewed;
            carried((posed.as_ref(), fitted), (actor.slot, entity, hand), (body, wish), shown)
        });
        let wanted = figure_play((table, &bound), memory, &shape, (look_yaw, hold));
        stats.edit(|stats| {
            stats.actors.push(actor_stats(
                (memory, table, &bound, &figure),
                (actor, body, clock),
                (&shape, hold.is_some(), look_yaw),
                (weapon, alive, reload, dip),
            ));
        });
        memory.moving = shape.moving;
        memory.bound = Some((bound_for, bound));
        written(&mut spawner, entity, play, wanted);
    }
    memories.held.retain(|entity, _| seen.contains(entity));
    refusals.keep_part(PUPPET_PART, |entity| seen.contains(&entity));
    if assets.is_changed() {
        crowns.keep(&assets);
    }
    let heads = (&*assets, table.head_check, &mut *crowns);
    stats.edit(|stats| {
        stats.actors.sort_unstable_by_key(|actor| actor.slot);
        let known = (&*memories, &*steps);
        watched_report(stats, known, (&actors, followed), shared.numbers, heads);
    });
}

fn watched_report(
    stats: &mut PuppetStats,
    (memories, steps): (&Memories, &StepJoints),
    (actors, followed): (&Actors, Option<Entity>),
    numbers: Option<&GameNumbers>,
    (assets, check, crowns): (&Assets, HeadCheck, &mut Crowns),
) {
    let watched = followed.and_then(|entity| Some((entity, actors.get(entity).ok()?)));
    let locomotion = watched.and_then(|(entity, ((_, actor, body, _), _, (_, _, figure, ..), ..))| {
        let shown = stats.actors.iter().find(|shown| shown.slot == actor.slot)?;
        Some(memories.held.get(&entity)?.locomotion((body, &*figure), shown))
    });
    stats.eyes = eye_fits(actors, numbers);
    stats.head = watched.and_then(
        |(entity, ((_, actor, body, _), _, (_, _, figure, drawn, ..), (_, clock), _))| {
            let posed = (steps.kept(entity), clock.map_or(1.0, |clock| clock.fit));
            head_check(numbers, (actor.slot, body), posed, |head, fit| {
                crowns.crown(assets, (&*figure, drawn?), (check, head, fit))
            })
        },
    );
    stats.aim = watched.and_then(|(entity, ((_, actor, _, wish), (_, offsets), ..))| {
        aim_check(
            memories.held.get(&entity)?,
            (actor.slot, wish, offsets),
            steps.kept(entity)?,
        )
    });
    stats.locomotion = locomotion;
    let mut missing: Vec<String> = actors
        .iter()
        .filter_map(|((entity, actor, ..), ..)| Some((actor.slot, memories.held.get(&entity)?)))
        .flat_map(|(slot, memory)| {
            memory
                .missing
                .iter()
                .map(move |clip| format!("slot {slot}: {clip}"))
        })
        .collect();
    missing.sort_unstable();
    stats.missing_count = missing.len() as u32;
    missing.truncate(8);
    stats.missing_clips = missing;
}
