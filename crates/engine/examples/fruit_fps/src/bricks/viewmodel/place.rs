use engine::prelude::*;
use engine_core::math::{clamp, max, min, sin_cos};
use engine_core::{Mobility, Parent, SceneContext};

use super::hands::Hands;
use super::{FirstPersonWeapon, ViewModelBrick, ViewModelStats};
use crate::names::CAMERA;
use crate::play::actors::{Actor, Alive, Body, Motion, Wish};
use crate::play::arms::Arms;
use crate::play::camera::TestCamera;
use crate::play::events::arms::{ArmsEvents, Weapon};
use crate::play::figures::{BaseLayer, FIELD_AT_CAMERA, FIGURE, Figure, FigurePlay, HandTarget};
use crate::play::flow::{Flow, Stage};
use crate::play::numbers::Numbers;
use crate::play::powers::Powers;
use crate::play::puppet::{
    ActorMuzzle, DEVICE, Muzzles, PuppetBook, PuppetFile, PuppetTable, Task, Tasks, ViewModel,
    ViewPlace, ViewTable, WeaponLook, weapon_key,
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
    bob: Vec3,
    phase: f32,
    kick_back: f32,
    kick_angle: f32,
    cue: u32,
    fired: bool,
    reloading: bool,
    swing: Option<bool>,
    look: Option<(f32, f32)>,
    hidden: Option<&'static str>,
    ability: bool,
}

impl Held {
    pub(super) fn offset(&self) -> Vec3 {
        self.sway + self.bob + Vec3::new(0.0, 0.0, self.kick_back)
    }
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

pub(super) fn moving(visible: bool) -> Node {
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
        Option<Res<'w, Tasks>>,
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

fn equipped(
    held: &mut Held,
    (weapon, (ability, returned)): (Option<&'static str>, (bool, bool)),
    equip_seconds: f32,
    delta: f32,
) -> f32 {
    held.ability = ability;
    if weapon != held.weapon || returned {
        held.weapon = weapon;
        held.equip_total = equip_seconds;
        held.equip_left = equip_seconds;
        held.fired = false;
        held.reloading = false;
        held.swing = None;
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
    held.bob = Vec3::new(bob.side_m * side, bob.up_m * (twice - 1.0) * 0.5, 0.0) * amount;
    held.bob
}

fn kicked(held: &mut Held, table: &PuppetTable, (events, slot): (&ArmsEvents, u32), delta: f32) {
    let kick = &table.view.kick;
    let decay = max(1.0 - kick.return_per_second * delta, 0.0);
    held.kick_back *= decay;
    held.kick_angle *= decay;
    for fired in events.shots.iter().filter(|fired| fired.shooter == slot) {
        if fired.weapon == Weapon::Melee {
            held.swing = Some(fired.heavy);
        } else {
            held.kick_back += kick.back_m;
            held.kick_angle += kick.up_radians;
        }
        held.fired = true;
        held.reloading = false;
        held.cue = held.cue.wrapping_add(1);
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

struct Timing {
    equip: f32,
    reload: Option<f32>,
    swing: (f32, f32),
    melee: bool,
    task: Option<(Task, f32)>,
}

fn chosen(
    numbers: &Numbers,
    loadout: Option<&Loadout>,
    (tasks, slot): (Option<&Tasks>, u32),
) -> (Option<&'static str>, Timing) {
    let task = tasks
        .and_then(|tasks| tasks.of(slot))
        .filter(|_| loadout.is_some());
    let game = numbers.get();
    let swing = game.map_or((0.0, 0.0), |game| {
        let melee = &game.weapons.melee;
        (melee.swing_seconds, melee.heavy_swing_seconds)
    });
    if let Some(task) = task {
        let seconds = game.map_or(0.0, |game| match task {
            Task::Planting => game.rules.round.plant_seconds,
            Task::Defusing => game.rules.round.defuse_seconds,
        });
        let timing = Timing {
            equip: 0.0,
            reload: None,
            swing,
            melee: false,
            task: Some((task, seconds as f32)),
        };
        return (Some(DEVICE), timing);
    }
    let weapon = loadout.map(weapon_key);
    let (equip, reload) = loadout.map_or((0.0, None), |loadout| seconds_of(numbers, loadout));
    let timing = Timing {
        equip,
        reload,
        swing,
        melee: weapon == Some(Weapon::Melee.key()),
        task: None,
    };
    (weapon, timing)
}

fn played(held: &Held, view: &ViewTable, timing: &Timing) -> (String, FigurePlay) {
    let equipping = view
        .equip
        .as_ref()
        .filter(|_| held.equip_left > 0.0 && !held.fired);
    let (clip, speed, looping, fit_seconds) = if let Some(equip) = equipping {
        (equip.clone(), 1.0, false, Some(held.equip_total))
    } else if let Some((task, seconds)) = timing.task {
        match task {
            Task::Planting => (view.device.plant.clone(), 1.0, false, Some(seconds)),
            Task::Defusing => (view.device.defuse.clone(), 1.0, true, None),
        }
    } else if held.reloading {
        (view.reload.clone(), 1.0, false, timing.reload)
    } else if timing.melee {
        match held.swing {
            Some(true) => (view.melee.heavy.clone(), 1.0, false, Some(timing.swing.1)),
            Some(false) => (view.melee.light.clone(), 1.0, false, Some(timing.swing.0)),
            None => match &view.equip {
                Some(equip) => (equip.clone(), 1.0, false, Some(held.equip_total)),
                None => (view.melee.light.clone(), 0.0, false, None),
            },
        }
    } else {
        let speed = f32::from(u8::from(held.fired));
        (view.fire.clone(), speed, false, None)
    };
    let base = BaseLayer {
        clip: clip.clone(),
        speed,
        looping,
        fade_seconds: 0.0,
        cue: held.cue,
        fit_seconds,
        blend: Vec::new(),
        stride: None,
        synced: false,
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
    hidden: Option<&'static str>,
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
        hidden,
        camera,
    } = part;
    let visible = hidden.is_none();
    held.hidden = hidden;
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

fn zoom_share(arms: &Arms, numbers: &Numbers) -> f32 {
    let first = numbers
        .get()
        .zip(arms.loadout.active_gun())
        .and_then(|(numbers, gun)| numbers.weapons.gun(gun.kind()).scope.as_ref())
        .and_then(|scope| scope.zoom.first().copied())
        .unwrap_or(1.0);
    if !arms.scoped {
        0.0
    } else if first <= 1.0 {
        1.0
    } else {
        clamp((arms.zoom - 1.0) / (first - 1.0), 0.0, 1.0)
    }
}

fn hidden(place: &ViewPlace, held: &Held) -> bool {
    place.hide_from_share.is_some_and(|from| held.scope >= from)
}

fn hidden_why(
    alive: bool,
    (flow, test): (Option<&Flow>, Option<&TestCamera>),
    (place, held): (&ViewPlace, &Held),
) -> Option<&'static str> {
    if !alive {
        Some("dead")
    } else if !playing(flow) {
        Some("not playing")
    } else if test.is_some_and(|test| test.outside) {
        Some("free camera")
    } else if held.ability {
        Some("ability")
    } else if hidden(place, held) {
        Some("scoped")
    } else {
        None
    }
}

fn stances(
    held: &mut Held,
    table: &PuppetTable,
    (place, arms, numbers): (Option<&ViewPlace>, Option<&Arms>, &Numbers),
    (body, powers, slot): (&Body, Option<&Powers>, u32),
    delta: f32,
) {
    let aiming = powers.and_then(|powers| powers.held(slot)).is_some();
    let loadout = arms.map(|arms| &arms.loadout);
    let share = arms.map_or(0.0, |arms| zoom_share(arms, numbers));
    let scoped = loadout
        .and_then(Loadout::active_gun)
        .is_some_and(|gun| gun.scope_level() > 0);
    let running = place.is_some_and(|place| place.lower_running)
        && body.grounded
        && body.motion == Motion::Running;
    let running = running || aiming;
    held.scope = if place.is_some_and(|place| place.hide_from_share.is_some()) {
        share
    } else {
        toward(
            held.scope,
            f32::from(u8::from(scoped)),
            delta / max(table.view.scope_seconds, delta),
        )
    };
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
    body_muzzle: Option<ActorMuzzle>,
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
        body_muzzle,
    } = shown;
    let (root, turn) = (frame.root, frame.turn);
    let eye_turn = about_y(yaw) * about_x(pitch);
    let (muzzle, direction) = body_muzzle.map_or_else(
        || {
            (
                look.markers
                    .muzzle
                    .map(|at| body.eye + eye_turn * (root + turn * at)),
                eye_turn * turn * Vec3::Z,
            )
        },
        |carried| (Some(carried.position), carried.direction),
    );
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
        bound: stats.bound.take(),
        hidden: stats.hidden.take(),
        refusal: None,
        hands: std::mem::take(&mut stats.hands),
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

pub(super) fn subject_of<'a>(watched: Option<&Watched>, humans: &'a Humans) -> Option<Subject<'a>> {
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

pub(super) fn playing(flow: Option<&Flow>) -> bool {
    flow.is_none_or(|flow| flow.stage == Stage::Playing)
}

type Viewer<'w> = (
    Option<Res<'w, Watched>>,
    Option<Res<'w, TestCamera>>,
    Option<Res<'w, Muzzles>>,
    Res<'w, Hands>,
);

fn body_muzzle(
    test: Option<&TestCamera>,
    muzzles: Option<&Muzzles>,
    slot: u32,
) -> Option<ActorMuzzle> {
    muzzles
        .filter(|_| test.is_some_and(|test| test.outside))
        .and_then(|muzzles| muzzles.of(slot))
        .copied()
}

type Veiled<'w> = (Option<Res<'w, Flow>>, Option<Res<'w, TestCamera>>);

pub(super) fn shown_now(flow: Option<&Flow>, test: Option<&TestCamera>) -> bool {
    playing(flow) && !test.is_some_and(|test| test.outside)
}

pub(super) fn veil(
    ((flow, test), scene): (Veiled, Option<Res<SceneContext>>),
    held: Res<Held>,
    parts: Query<(&Node, &Drawn), With<FirstPersonWeapon>>,
    (mut stats, mut spawner): (ResMut<ViewModelStats>, Spawner<ViewModelBrick>),
) {
    let part = held.entity.and_then(|entity| parts.get(entity).ok());
    let bound = part
        .and_then(|(_, drawn)| drawn.geometry)
        .zip(scene.as_deref())
        .and_then(|(geometry, scene)| {
            scene
                .assets
                .meshes
                .iter()
                .find(|(_, handle)| **handle == geometry)
                .map(|(name, _)| name.clone())
        });
    let shown = shown_now(flow.as_deref(), test.as_deref());
    let hidden = match (shown, test.as_deref().is_some_and(|test| test.outside)) {
        (true, _) => held.hidden,
        (false, true) => Some("free camera"),
        (false, false) => Some("not playing"),
    }
    .map(str::to_owned);
    if stats.bound != bound || stats.hidden != hidden {
        stats.bound = bound;
        stats.hidden = hidden;
    }
    if shown {
        return;
    }
    if let (Some(entity), Some((node, _))) = (held.entity, part)
        && node.visible
    {
        spawner.insert(entity, moving(false));
    }
}

pub(super) fn camera_lit(mut drawn: Drawn) -> Drawn {
    drawn.values.floats = FIELD_AT_CAMERA.to_vec();
    drawn
}

pub(super) fn camera_of(scene: &SceneContext) -> Option<Entity> {
    scene.entities.get(CAMERA).copied()
}

pub(super) fn place(
    (fixed, numbers, scene, files, (arms_events, flow, powers, tasks)): Reads,
    (humans, (watched, test, muzzles, hands)): (Humans, Viewer),
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
    let living = arms.filter(|_| alive);
    let loadout = living.map(|arms| &arms.loadout);
    let (weapon, timing) = chosen(&numbers, loadout, (tasks.as_deref(), actor.slot));
    let ability = (hands.shown, hands.returned);
    let raised = equipped(&mut held, (weapon, ability), timing.equip, delta);
    let raise = table.view.equip.as_ref().map_or(raised, |_| 1.0);
    let look = weapon.and_then(|weapon| book.weapons.get(weapon).map(|look| (weapon, look)));
    let place = weapon.and_then(|weapon| table.view.weapons.get(weapon));
    stances(
        &mut held,
        table,
        (place, living, &numbers),
        (body, powers.as_deref(), actor.slot),
        delta,
    );
    let yaw = wish.yaw_radians + offsets.map_or(0.0, ViewOffsets::yaw_radians);
    let pitch = wish.pitch_radians + offsets.map_or(0.0, ViewOffsets::pitch_radians);
    swayed(&mut held, table, (yaw, pitch), delta);
    let bobbed = bobbing(&mut held, table, body, delta);
    kicked(&mut held, table, (&arms_events, actor.slot), delta);
    reloads(&mut held, reloading(loadout, &numbers));
    let (clip, wanted) = played(&held, &table.view, &timing);
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
        drawn: camera_lit(Drawn::new(FIGURE, mesh, material)),
        figure: look.files.figure.clone(),
        wanted,
        placed,
        hidden: hidden_why(alive, (flow.as_deref(), test.as_deref()), (place, &held)),
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
        body_muzzle: body_muzzle(test.as_deref(), muzzles.as_deref(), actor.slot),
    };
    published(
        &book,
        (&held, &frame),
        (body, yaw, pitch),
        shown,
        (fixed.tick, &mut view, &mut stats),
    );
}
