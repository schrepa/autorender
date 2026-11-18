use engine::prelude::*;
use engine_core::math::{clamp, max, min, sin_cos};
use engine_core::{Mobility, Parent, SceneContext};

use super::{FirstPersonWeapon, ViewModelBrick, ViewModelStats};
use crate::names::CAMERA;
use crate::play::actors::{Actor, Alive, Body, Motion, Wish};
use crate::play::arms::Arms;
use crate::play::events::arms::{ArmsEvents, Weapon};
use crate::play::figures::{BaseLayer, FIGURE, Figure, FigurePlay, HandTarget};
use crate::play::flow::{Flow, Stage};
use crate::play::numbers::Numbers;
use crate::play::powers::Powers;
use crate::play::puppet::{
    PuppetBook, PuppetFile, PuppetTable, ViewModel, ViewPlace, ViewTable, WeaponLook, weapon_key,
};
use crate::play::view::{ViewOffsets, Watched};
use crate::weapons::{Loadout, Stance};

#[derive(Resource, Default)]
pub(super) struct Held {
    entity: Option<Entity>,
    subject: Option<Entity>,
    weapon: Option<&'static str>,
    equip_left: f32,
    equip_total: f32,
    scope: f32,
    lower: f32,
    sway: Vec3,
    phase: f32,
    kick_back: f32,
    kick_angle: f32,
    cue: u32,
    fired: bool,
    reloading: bool,
    look: Option<(f32, f32)>,
}

fn eased(share: f32) -> f32 {
    let share = clamp(share, 0.0, 1.0);
    share * share * (3.0 - 2.0 * share)
}

fn toward(value: f32, goal: f32, step: f32) -> f32 {
    if value < goal {
        min(value + step, goal)
    } else {
        max(value - step, goal)
    }
}

fn wrapped(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

fn about_x(angle: f32) -> Quat {
    let (sin, cos) = sin_cos(angle * 0.5);
    Quat::from_xyzw(sin, 0.0, 0.0, cos)
}

fn about_y(angle: f32) -> Quat {
    let (sin, cos) = sin_cos(angle * 0.5);
    Quat::from_xyzw(0.0, sin, 0.0, cos)
}

fn moving(visible: bool) -> Node {
    Node {
        visible,
        mobility: Some(Mobility::Moving),
        ..Node::default()
    }
}

fn seconds_of(numbers: &Numbers, loadout: &Loadout) -> (f32, Option<f32>) {
    let Some(numbers) = numbers.get() else {
        return (0.0, None);
    };
    match loadout.active_gun() {
        Some(gun) => {
            let gun = numbers.weapons.gun(gun.kind());
            (gun.equip_seconds, Some(gun.reload_seconds))
        }
        None => (numbers.weapons.melee.equip_seconds, None),
    }
}

struct Frame {
    root: Vec3,
    turn: Quat,
    sight: Vec3,
}

fn hands(
    book: &PuppetBook,
    weapon: &str,
    look: &WeaponLook,
    frame: &Frame,
) -> [Option<HandTarget>; 2] {
    let table = &book.table;
    let Some(pose) = table
        .weapons
        .get(weapon)
        .and_then(|hold| table.holds.get(hold))
    else {
        return [None, None];
    };
    let grip_r = look.markers.grip_r.unwrap_or(Vec3::ZERO);
    let at = |point: Vec3| frame.root + frame.turn * point;
    let right = HandTarget {
        position: at(pose.right.wrist(grip_r)).to_array(),
        rotation: (frame.turn * pose.right.hand(&table.hand_axes.right))
            .normalize()
            .to_array(),
        pole: (frame.turn * Vec3::from_array(pose.right.pole)).to_array(),
    };
    let left = pose
        .left
        .zip(look.markers.grip_l)
        .map(|(grip, held)| HandTarget {
            position: at(grip.wrist(held)).to_array(),
            rotation: (frame.turn * grip.hand(&table.hand_axes.left))
                .normalize()
                .to_array(),
            pole: (frame.turn * Vec3::from_array(grip.pole)).to_array(),
        });
    [Some(right), left]
}

fn sighted(place: &ViewPlace, held: &Held, raise: f32) -> Vec3 {
    let rest = Vec3::from_array(place.rest);
    let aimed = rest.lerp(Vec3::from_array(place.scoped), eased(held.scope));
    let down = max(1.0 - raise, eased(held.lower));
    aimed.lerp(Vec3::from_array(place.lowered), down)
}

type Reads<'w, 's> = (
    Res<'w, FixedTime>,
    Res<'w, Numbers>,
    Option<Res<'w, SceneContext>>,
    Query<'w, 's, &'static PuppetFile>,
    (
        Res<'w, ArmsEvents>,
        Option<Res<'w, Flow>>,
        Option<Res<'w, Powers>>,
    ),
);

type Humans<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Actor,
        &'static Body,
        &'static Wish,
        Option<&'static Arms>,
        Option<&'static ViewOffsets>,
        Has<Alive>,
    ),
>;

type Parts<'w, 's> = Query<
    'w,
    's,
    (
        &'static Drawn,
        &'static Figure,
        &'static Node,
        Option<&'static Parent>,
        Option<&'static mut FigurePlay>,
    ),
    With<FirstPersonWeapon>,
>;

type Writes<'w, 's> = (
    ResMut<'w, Held>,
    ResMut<'w, ViewModel>,
    ResMut<'w, ViewModelStats>,
    Moves<'w, 's, ViewModelBrick>,
    Spawner<'w, 's, ViewModelBrick>,
);

fn refused(
    stats: &mut ViewModelStats,
    (held, parts): (&Held, &Parts),
    spawner: &mut Spawner<ViewModelBrick>,
    why: Option<&str>,
) {
    if let Some(entity) = held.entity
        && parts.get(entity).is_ok_and(|(_, _, node, ..)| node.visible)
    {
        spawner.insert(entity, moving(false));
    }
    *stats = ViewModelStats {
        refusal: why.map(str::to_owned),
        ..ViewModelStats::default()
    };
}

fn equipped(held: &mut Held, weapon: Option<&'static str>, equip_seconds: f32, delta: f32) -> f32 {
    if weapon != held.weapon {
        held.weapon = weapon;
        held.equip_total = equip_seconds;
        held.equip_left = equip_seconds;
        held.fired = false;
        held.reloading = false;
        held.cue = held.cue.wrapping_add(1);
    }
    held.equip_left = max(held.equip_left - delta, 0.0);
    if held.equip_total > 0.0 {
        eased(1.0 - held.equip_left / held.equip_total)
    } else {
        1.0
    }
}

fn swayed(held: &mut Held, table: &PuppetTable, (yaw, pitch): (f32, f32), delta: f32) {
    let sway = &table.view.sway;
    let target = match held.look {
        Some((was_yaw, was_pitch)) if delta > 0.0 => {
            let reach = sway.max_m;
            Vec3::new(
                clamp(
                    wrapped(yaw - was_yaw) / delta * sway.metres_per_radian,
                    -reach,
                    reach,
                ),
                clamp(
                    -(pitch - was_pitch) / delta * sway.metres_per_radian,
                    -reach,
                    reach,
                ),
                0.0,
            )
        }
        Some(_) | None => Vec3::ZERO,
    };
    held.look = Some((yaw, pitch));
    let follow = min(sway.follow_per_second * delta, 1.0);
    held.sway = held.sway + (target - held.sway) * follow;
}

fn bobbing(held: &mut Held, table: &PuppetTable, body: &Body, delta: f32) -> Vec3 {
    let bob = &table.view.bob;
    let speed = Vec3::new(body.velocity.x, 0.0, body.velocity.z).length();
    if body.grounded && bob.stride_m > 0.0 {
        held.phase = (held.phase + speed * delta / bob.stride_m * std::f32::consts::TAU)
            .rem_euclid(std::f32::consts::TAU);
    }
    let amount = if body.grounded && bob.full_mps > 0.0 {
        min(speed / bob.full_mps, 1.0) * (1.0 - eased(held.scope))
    } else {
        0.0
    };
    let (side, _) = sin_cos(held.phase);
    let (_, twice) = sin_cos(held.phase * 2.0);
    Vec3::new(bob.side_m * side, bob.up_m * (twice - 1.0) * 0.5, 0.0) * amount
}

fn kicked(held: &mut Held, table: &PuppetTable, (events, slot): (&ArmsEvents, u32), delta: f32) {
    let kick = &table.view.kick;
    let decay = max(1.0 - kick.return_per_second * delta, 0.0);
    held.kick_back *= decay;
    held.kick_angle *= decay;
    for fired in &events.shots {
        if fired.shooter == slot && fired.weapon != Weapon::Melee {
            held.kick_back += kick.back_m;
            held.kick_angle += kick.up_radians;
            held.fired = true;
            held.reloading = false;
            held.cue = held.cue.wrapping_add(1);
        }
    }
}

fn reloading(loadout: Option<&Loadout>, numbers: &Numbers) -> bool {
    loadout
        .zip(numbers.get())
        .and_then(|(loadout, numbers)| {
            loadout
                .view(&numbers.weapons, Stance::Stand)
                .reload_progress
        })
        .is_some()
}

fn reloads(held: &mut Held, reloading: bool) {
    if reloading && !held.reloading {
        held.cue = held.cue.wrapping_add(1);
    }
    if !reloading && held.reloading {
        held.fired = false;
    }
    held.reloading = reloading;
}

fn played(
    held: &Held,
    view_table: &ViewTable,
    reload_seconds: Option<f32>,
) -> (String, FigurePlay) {
    let equipping = view_table
        .equip
        .as_ref()
        .filter(|_| held.equip_left > 0.0 && !held.fired);
    let (clip, base) = if let Some(equip) = equipping {
        (
            equip.clone(),
            BaseLayer {
                clip: equip.clone(),
                speed: 1.0,
                looping: false,
                fade_seconds: 0.0,
                cue: held.cue,
                fit_seconds: Some(held.equip_total),
            },
        )
    } else if held.reloading {
        (
            view_table.reload.clone(),
            BaseLayer {
                clip: view_table.reload.clone(),
                speed: 1.0,
                looping: false,
                fade_seconds: 0.0,
                cue: held.cue,
                fit_seconds: reload_seconds,
            },
        )
    } else {
        (
            view_table.fire.clone(),
            BaseLayer {
                clip: view_table.fire.clone(),
                speed: if held.fired { 1.0 } else { 0.0 },
                looping: false,
                fade_seconds: 0.0,
                cue: held.cue,
                fit_seconds: None,
            },
        )
    };
    let wanted = FigurePlay {
        base,
        upper: None,
        aim: None,
        hold: None,
        crouch: None,
        pin_feet: None,
    };
    (clip, wanted)
}

struct Part {
    drawn: Drawn,
    figure: Figure,
    wanted: FigurePlay,
    placed: Transform,
    visible: bool,
    camera: Entity,
}

fn kept(
    held: &mut Held,
    parts: &mut Parts,
    (moves, spawner): (&mut Moves<ViewModelBrick>, &mut Spawner<ViewModelBrick>),
    part: Part,
) {
    let Part {
        drawn,
        figure,
        wanted,
        placed,
        visible,
        camera,
    } = part;
    let existing = held
        .entity
        .and_then(|entity| parts.get_mut(entity).ok().map(|part| (entity, part)));
    match existing {
        None => {
            let entity = spawner.spawn((
                placed,
                moving(visible),
                drawn,
                figure,
                wanted,
                FirstPersonWeapon,
            ));
            spawner.set_parent(entity, Some(camera));
            held.entity = Some(entity);
        }
        Some((entity, (was_drawn, was_figure, node, parent, play))) => {
            if was_drawn.geometry != drawn.geometry || was_drawn.material != drawn.material {
                spawner.insert(entity, drawn);
            }
            if *was_figure != figure {
                spawner.insert(entity, figure);
            }
            if node.visible != visible {
                spawner.insert(entity, moving(visible));
            }
            if parent.map(Parent::parent) != Some(camera) {
                spawner.set_parent(entity, Some(camera));
            }
            match play {
                Some(mut play) => {
                    play.set_if_neq(wanted);
                }
                None => spawner.insert(entity, wanted),
            }
            if let Some(mut transform) = moves.get_mut(entity) {
                transform.set_if_neq(placed);
            }
        }
    }
}

fn stances(
    held: &mut Held,
    table: &PuppetTable,
    (place, loadout): (Option<&ViewPlace>, Option<&Loadout>),
    (body, aiming): (&Body, bool),
    delta: f32,
) {
    let scoped = loadout
        .and_then(Loadout::active_gun)
        .is_some_and(|gun| gun.scope_level() > 0);
    let running = place.is_some_and(|place| place.lower_running)
        && body.grounded
        && body.motion == Motion::Running;
    let running = running || aiming;
    held.scope = toward(
        held.scope,
        f32::from(u8::from(scoped)),
        delta / max(table.view.scope_seconds, delta),
    );
    held.lower = toward(
        held.lower,
        f32::from(u8::from(running)),
        delta / max(table.view.lower_seconds, delta),
    );
}

struct Shown<'a> {
    slot: u32,
    weapon: &'static str,
    look: &'a WeaponLook,
    clip: String,
    raise: f32,
    scale: f32,
}

fn published(
    book: &PuppetBook,
    (held, frame): (&Held, &Frame),
    (body, yaw, pitch): (&Body, f32, f32),
    shown: Shown<'_>,
    (tick, view, stats): (u64, &mut ViewModel, &mut ViewModelStats),
) {
    let Shown {
        slot,
        weapon,
        look,
        clip,
        raise,
        scale,
    } = shown;
    let (root, turn) = (frame.root, frame.turn);
    let eye_turn = about_y(yaw) * about_x(pitch);
    let muzzle = look
        .markers
        .muzzle
        .map(|at| body.eye + eye_turn * (root + turn * at));
    let direction = eye_turn * turn * Vec3::Z;
    let fresh = ViewModel {
        tick,
        slot: Some(slot),
        weapon: Some(weapon),
        clip: clip.clone(),
        muzzle,
        direction,
        sight: frame.sight,
        hands: hands(book, weapon, look, frame),
        scale,
    };
    if *view != fresh {
        *view = fresh;
    }
    let report = ViewModelStats {
        weapon: Some(weapon.to_owned()),
        clip,
        offset: frame.sight.to_array(),
        scale,
        raise,
        scope: held.scope,
        lower: held.lower,
        kick: held.kick_back,
        muzzle: muzzle.map(|at| at.to_array()),
        refusal: None,
    };
    *stats = report;
}

type Subject<'a> = (
    Entity,
    &'a Actor,
    &'a Body,
    &'a Wish,
    Option<&'a Arms>,
    Option<&'a ViewOffsets>,
    bool,
);

fn subject_of<'a>(watched: Option<&Watched>, humans: &'a Humans) -> Option<Subject<'a>> {
    let entity = watched?.entity?;
    humans.get(entity).ok()
}

fn followed(held: &mut Held, entity: Entity) {
    if held.subject != Some(entity) {
        held.subject = Some(entity);
        held.weapon = None;
        held.look = None;
    }
}

fn playing(flow: Option<&Flow>) -> bool {
    flow.is_none_or(|flow| flow.stage == Stage::Playing)
}

pub(super) fn veil(
    flow: Option<Res<Flow>>,
    held: Res<Held>,
    parts: Query<&Node, With<FirstPersonWeapon>>,
    mut spawner: Spawner<ViewModelBrick>,
) {
    if playing(flow.as_deref()) {
        return;
    }
    if let Some(entity) = held.entity
        && parts.get(entity).is_ok_and(|node| node.visible)
    {
        spawner.insert(entity, moving(false));
    }
}

fn camera_of(scene: &SceneContext) -> Option<Entity> {
    scene.entities.get(CAMERA).copied()
}

pub(super) fn place(
    (fixed, numbers, scene, files, (arms_events, flow, powers)): Reads,
    (humans, watched): (Humans, Option<Res<Watched>>),
    mut parts: Parts,
    (mut held, mut view, mut stats, mut moves, mut spawner): Writes,
) {
    let Some(book) = files.iter().find_map(PuppetFile::book).cloned() else {
        refused(
            &mut stats,
            (&held, &parts),
            &mut spawner,
            Some("no prepared `puppet` component"),
        );
        return;
    };
    let Some((entity, actor, body, wish, arms, offsets, alive)) =
        subject_of(watched.as_deref(), &humans)
    else {
        refused(
            &mut stats,
            (&held, &parts),
            &mut spawner,
            Some("no watched actor (corpse view, or `Watched` not written)"),
        );
        view.set_if_neq(ViewModel::default());
        return;
    };
    followed(&mut held, entity);
    let Some(camera) = scene.as_deref().and_then(camera_of) else {
        refused(
            &mut stats,
            (&held, &parts),
            &mut spawner,
            Some("no scene entity `camera`"),
        );
        return;
    };
    let table = &book.table;
    let delta = fixed.delta_seconds;
    let loadout = arms.filter(|_| alive).map(|arms| &arms.loadout);
    let weapon = loadout.map(weapon_key);
    let (equip_seconds, reload_seconds) =
        loadout.map_or((0.0, None), |loadout| seconds_of(&numbers, loadout));
    let raised = equipped(&mut held, weapon, equip_seconds, delta);
    let raise = table.view.equip.as_ref().map_or(raised, |_| 1.0);
    let look = weapon.and_then(|weapon| book.weapons.get(weapon).map(|look| (weapon, look)));
    let place = weapon.and_then(|weapon| table.view.weapons.get(weapon));
    let aiming = powers
        .as_deref()
        .and_then(|powers| powers.held(actor.slot))
        .is_some();
    stances(&mut held, table, (place, loadout), (body, aiming), delta);
    let yaw = wish.yaw_radians + offsets.map_or(0.0, ViewOffsets::yaw_radians);
    let pitch = wish.pitch_radians + offsets.map_or(0.0, ViewOffsets::pitch_radians);
    swayed(&mut held, table, (yaw, pitch), delta);
    let bobbed = bobbing(&mut held, table, body, delta);
    kicked(&mut held, table, (&arms_events, actor.slot), delta);
    reloads(&mut held, reloading(loadout, &numbers));
    let (clip, wanted) = played(&held, &table.view, reload_seconds);
    let scale = table.view.scale;
    let (Some((weapon, look)), Some(place)) = (look, place) else {
        refused(&mut stats, (&held, &parts), &mut spawner, None);
        let fresh = ViewModel {
            tick: fixed.tick,
            slot: Some(actor.slot),
            weapon,
            clip: String::new(),
            muzzle: None,
            direction: Vec3::ZERO,
            sight: Vec3::ZERO,
            hands: [None, None],
            scale,
        };
        view.set_if_neq(fresh);
        return;
    };
    let assets = scene.as_deref().map(|scene| &*scene.assets);
    let mesh = assets.and_then(|assets| assets.meshes.get(&look.files.mesh).copied());
    let material = assets.and_then(|assets| assets.materials.get(&look.files.material).copied());
    let (Some(mesh), Some(material)) = (mesh, material) else {
        refused(
            &mut stats,
            (&held, &parts),
            &mut spawner,
            Some("weapon mesh or material missing from the scene"),
        );
        return;
    };
    let sight =
        sighted(place, &held, raise) + held.sway + bobbed + Vec3::new(0.0, 0.0, held.kick_back);
    let turn = (about_x(held.kick_angle) * about_y(std::f32::consts::PI)).normalize();
    let marker = look.markers.sight.unwrap_or(Vec3::ZERO);
    let root = sight - turn * marker;
    let frame = Frame { root, turn, sight };
    let placed = Transform {
        translation: root * scale,
        rotation: turn,
        scale: Vec3::splat(scale),
    };
    let part = Part {
        drawn: Drawn::new(FIGURE, mesh, material),
        figure: look.files.figure.clone(),
        wanted,
        placed,
        visible: alive && playing(flow.as_deref()),
        camera,
    };
    kept(&mut held, &mut parts, (&mut moves, &mut spawner), part);
    let shown = Shown {
        slot: actor.slot,
        weapon,
        look,
        clip,
        raise,
        scale,
    };
    published(
        &book,
        (&held, &frame),
        (body, yaw, pitch),
        shown,
        (fixed.tick, &mut view, &mut stats),
    );
}
