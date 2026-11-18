use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, Quat, Vec3};
use engine_core::{
    AssetLibrary, Drawn, DrawnTable, Error, ErrorCode, FixedTime, FrameTime, GlobalTransforms,
    InstanceValues, Moves, Parent, Refusals, Spawner, Transform,
};
use engine_rig::{PaletteRow, Posed, Reach};

use super::clock::{Clocks, FigureClocks};
use super::files::{FigureFiles, Played, Rig, unreadable};
use super::{FIGURES_PART, FigureHead, FigureRefused, FiguresBrick, FiguresStats};
use crate::play::figures::{
    Blended, FIGURE, Figure, FigureJoints, FigurePlay, GROUND, HIP, Held, Hold, HoldSpace, JointAt,
    LEFT_ARM, LEFT_LEG, RIGHT_ARM, RIGHT_LEG, Stride,
};
use crate::play::light::Light;
use engine_core::math::max;

const DRAW_PART: &str = "figures_draw";

const HELD_PART: &str = "figures_held";

const ROW_BYTES: u64 = 48;

type Carried<'a> = BTreeMap<Entity, Vec<(Entity, &'a Held, Option<Entity>)>>;

#[derive(Default)]
pub struct Scratch {
    base: Vec<Posed>,
    other: Vec<Posed>,
    out: Vec<Posed>,
    weights: Vec<f32>,
    turns: Vec<(usize, Quat)>,
    model: Vec<Mat4>,
}

#[derive(Resource, Default)]
pub struct FigureDrawing {
    pub(super) rows: Vec<PaletteRow>,
    pub(super) written: Vec<(usize, usize)>,
    slots: BTreeMap<Entity, (usize, usize)>,
}

impl FigureDrawing {
    fn slot(&mut self, entity: Entity, joints: usize) -> usize {
        if let Some(&(at, held)) = self.slots.get(&entity)
            && held == joints
        {
            return at;
        }
        self.slots.remove(&entity);
        let mut taken: Vec<(usize, usize)> = self.slots.values().copied().collect();
        taken.sort_unstable();
        let mut at = 0;
        for (start, length) in taken {
            if start >= at + joints {
                break;
            }
            at = at.max(start + length);
        }
        self.slots.insert(entity, (at, joints));
        if self.rows.len() < at + joints {
            self.rows.resize(at + joints, [0.0; 12]);
        }
        at
    }

    fn keep(&mut self, drawn: &[Entity]) {
        self.slots.retain(|entity, _| drawn.contains(entity));
        let end = self
            .slots
            .values()
            .map(|(at, length)| at + length)
            .max()
            .unwrap_or(0);
        self.rows.truncate(end);
    }
}

fn unknown(path: &str, name: &str, rig: &Rig) -> Error {
    Error::new(ErrorCode::UnknownReference, path.to_owned())
        .got(name)
        .variants(rig.clips.keys().cloned())
}

fn named_clips(rig: &Rig, play: Option<&FigurePlay>) -> Result<(), Error> {
    let Some(play) = play else {
        return Ok(());
    };
    if !rig.clips.contains_key(&play.base.clip) {
        return Err(unknown("$.figure_play.base.clip", &play.base.clip, rig));
    }
    if let Some(missing) = play
        .base
        .blend
        .iter()
        .find(|blended| !rig.clips.contains_key(&blended.clip))
    {
        return Err(unknown("$.figure_play.base.blend", &missing.clip, rig));
    }
    if let Some(upper) = &play.upper
        && !rig.clips.contains_key(&upper.clip)
    {
        return Err(unknown("$.figure_play.upper.clip", &upper.clip, rig));
    }
    Ok(())
}

fn rigged(error: &Error) -> Error {
    unreadable("$.figure".to_owned(), error)
}

fn missing_joints(names: &[&str]) -> Error {
    Error::new(ErrorCode::UnknownReference, "$.figure.skeleton").got(names)
}

fn crouched(rig: &Rig, drop: f32, posed: &mut [Posed]) -> Result<(), Error> {
    let skeleton = &*rig.skeleton;
    let hip = rig.limbs.hip.ok_or_else(|| missing_joints(&[HIP]))?;
    let legs = rig
        .limbs
        .legs
        .ok_or_else(|| missing_joints(&[LEFT_LEG, RIGHT_LEG].concat()))?;
    let planted = legs.map(|leg| posed[leg[2]]);
    engine_rig::shift(skeleton, posed, hip, Vec3::new(0.0, -drop, 0.0))
        .map_err(|error| rigged(&error))?;
    for (leg, foot) in legs.into_iter().zip(planted) {
        let reach = Reach {
            target: foot.position,
            pole: posed[leg[1]].position - posed[leg[0]].position,
            end_rotation: Some(foot.rotation),
            weight: 1.0,
        };
        engine_rig::reach(skeleton, posed, leg, &reach).map_err(|error| rigged(&error))?;
    }
    Ok(())
}

fn mixed(
    rig: &Rig,
    (played, at): (&Played, f32),
    blend: &[Blended],
    scratch: &mut Scratch,
) -> Result<(), Error> {
    let skeleton = &*rig.skeleton;
    let share = if played.duration() > 0.0 {
        at / played.duration()
    } else {
        0.0
    };
    let mut held = 1.0 - blend.iter().map(|blended| blended.weight).sum::<f32>();
    for blended in blend.iter().filter(|blended| blended.weight > 0.0) {
        let Some(other) = rig.clips.get(&blended.clip) else {
            return Err(unknown("$.figure_play.base.blend", &blended.clip, rig));
        };
        other
            .bound
            .sample(
                &other.clip,
                share * other.duration(),
                skeleton,
                &mut scratch.other,
            )
            .map_err(|error| rigged(&error))?;
        let total = max(held, 0.0) + blended.weight;
        scratch.weights.fill(blended.weight / total);
        engine_rig::blend(
            skeleton,
            &scratch.base,
            &scratch.other,
            &scratch.weights,
            played.bound.root_joint(),
            &mut scratch.out,
        )
        .map_err(|error| rigged(&error))?;
        std::mem::swap(&mut scratch.base, &mut scratch.out);
        held = total;
    }
    Ok(())
}

fn strode(rig: &Rig, stride: Stride, posed: &mut [Posed]) -> Result<(), Error> {
    let skeleton = &*rig.skeleton;
    let (Some(hip), Some(legs)) = (rig.limbs.hip, rig.limbs.legs) else {
        return Ok(());
    };
    let axis = Vec3::new(stride.axis[0], 0.0, stride.axis[1]).normalize_or_zero();
    let centre = posed[hip].position;
    for leg in legs {
        let foot = posed[leg[2]];
        let along = (foot.position - centre).dot(axis);
        let reach = Reach {
            target: foot.position + axis * along * (stride.scale - 1.0),
            pole: posed[leg[1]].position - posed[leg[0]].position,
            end_rotation: Some(foot.rotation),
            weight: 1.0,
        };
        engine_rig::reach(skeleton, posed, leg, &reach).map_err(|error| rigged(&error))?;
    }
    Ok(())
}

fn joint_named(skeleton: &engine_rig::Skeleton, name: &str) -> Result<usize, Error> {
    skeleton
        .find(name)
        .ok_or_else(|| Error::new(ErrorCode::UnknownReference, "$.figure_play.aim").got(name))
}

fn looked(
    skeleton: &engine_rig::Skeleton,
    aim: &crate::play::figures::Aim,
    posed: &mut [Posed],
    turns: &mut Vec<(usize, Quat)>,
) -> Result<(), Error> {
    let Some(hips) = aim.hips.as_deref() else {
        return Ok(());
    };
    if aim.yaw == 0.0 || aim.chain.is_empty() {
        return Ok(());
    }
    let about = |angle: f32| {
        let (sin, cos) = engine_core::math::sin_cos(angle * 0.5);
        Quat::from_xyzw(0.0, sin, 0.0, cos)
    };
    turns.clear();
    turns.push((joint_named(skeleton, hips)?, about(-aim.yaw)));
    let share = aim.yaw / aim.chain.len() as f32;
    for name in &aim.chain {
        turns.push((joint_named(skeleton, name)?, about(share)));
    }
    engine_rig::turn(skeleton, posed, turns).map_err(|error| rigged(&error))
}

fn pinned(rig: &Rig, played: &Played, contact: f32, scratch: &mut Scratch) -> Result<(), Error> {
    let Some(legs) = rig.limbs.legs else {
        return Ok(());
    };
    let skeleton = &*rig.skeleton;
    played
        .bound
        .sample(&played.clip, 0.0, skeleton, &mut scratch.other)
        .map_err(|error| rigged(&error))?;
    for leg in legs {
        let start = scratch.other[leg[2]].position;
        let foot = scratch.base[leg[2]];
        if foot.position.y > start.y + contact {
            continue;
        }
        let reach = Reach {
            target: Vec3::new(start.x, foot.position.y, start.z),
            pole: scratch.base[leg[1]].position - scratch.base[leg[0]].position,
            end_rotation: Some(foot.rotation),
            weight: 1.0,
        };
        engine_rig::reach(skeleton, &mut scratch.base, leg, &reach)
            .map_err(|error| rigged(&error))?;
    }
    Ok(())
}

fn held_hands(
    rig: &Rig,
    hold: &Hold,
    world: Option<Mat4>,
    posed: &mut [Posed],
) -> Result<(), Error> {
    let to_model = match hold.space {
        HoldSpace::Figure => Mat4::IDENTITY,
        HoldSpace::World => world.map(|world| world.inverse()).ok_or_else(|| {
            Error::new(ErrorCode::InvalidWorldTransform, "$.figure_play.hold.space").got("world")
        })?,
    };
    let (_, turned, _) = to_model.to_scale_rotation_translation();
    let hands = [
        (hold.right, rig.limbs.right_arm, RIGHT_ARM),
        (hold.left, rig.limbs.left_arm, LEFT_ARM),
    ];
    for (hand, arm, names) in hands {
        let Some(hand) = hand else {
            continue;
        };
        let arm = arm.ok_or_else(|| missing_joints(&names))?;
        let reach = Reach {
            target: to_model.transform_point3(Vec3::from_array(hand.position)),
            pole: to_model.transform_vector3(Vec3::from_array(hand.pole)),
            end_rotation: Some((turned * Quat::from_array(hand.rotation)).normalize()),
            weight: hold.weight,
        };
        engine_rig::reach(&rig.skeleton, posed, arm, &reach).map_err(|error| rigged(&error))?;
    }
    Ok(())
}

fn posed(
    (rig, clocks, play): (&Rig, &Clocks, Option<&FigurePlay>),
    (files, world): (&mut FigureFiles, Option<Mat4>),
    back: f32,
    scratch: &mut Scratch,
) -> Result<(), Error> {
    let skeleton = &*rig.skeleton;
    let joints = skeleton.len();
    scratch.base.clear();
    scratch
        .base
        .extend((0..joints).map(|joint| skeleton.rest(joint)));
    scratch.other.clone_from(&scratch.base);
    scratch.out.clone_from(&scratch.base);
    scratch.weights.resize(joints, 0.0);
    let Some(base) = &clocks.base else {
        return Ok(());
    };
    let Some(played) = rig.clips.get(&base.clip) else {
        return Err(unknown("$.figure_play.base.clip", &base.clip, rig));
    };
    let at = base.at(played.duration(), back);
    played
        .bound
        .sample(&played.clip, at, skeleton, &mut scratch.base)
        .map_err(|error| rigged(&error))?;
    if let Some(fade) = &clocks.fade
        && let Some(from) = rig.clips.get(&fade.from.clip)
    {
        let at = fade.from.at(from.duration(), back);
        from.bound
            .sample(&from.clip, at, skeleton, &mut scratch.other)
            .map_err(|error| rigged(&error))?;
        scratch.weights.fill(fade.weight(back));
        engine_rig::blend(
            skeleton,
            &scratch.other,
            &scratch.base,
            &scratch.weights,
            played.bound.root_joint(),
            &mut scratch.out,
        )
        .map_err(|error| rigged(&error))?;
        std::mem::swap(&mut scratch.base, &mut scratch.out);
    }
    if let Some(play) = play
        && !play.base.blend.is_empty()
    {
        mixed(rig, (played, at), &play.base.blend, scratch)?;
    }
    let upper = clocks
        .upper
        .as_ref()
        .zip(play.and_then(|play| play.upper.as_ref()));
    if let Some((clock, layer)) = upper
        && layer.weight > 0.0
    {
        let Some(over) = rig.clips.get(&clock.clip) else {
            return Err(unknown("$.figure_play.upper.clip", &clock.clip, rig));
        };
        let at = clock.at(over.duration(), back);
        over.bound
            .sample(&over.clip, at, skeleton, &mut scratch.other)
            .map_err(|error| rigged(&error))?;
        let mask = files.mask(rig, &layer.from, layer.feather)?;
        for (weight, masked) in scratch.weights.iter_mut().zip(mask.iter()) {
            *weight = masked * layer.weight;
        }
        engine_rig::blend(
            skeleton,
            &scratch.base,
            &scratch.other,
            &scratch.weights,
            None,
            &mut scratch.out,
        )
        .map_err(|error| rigged(&error))?;
        std::mem::swap(&mut scratch.base, &mut scratch.out);
    }
    if let Some(aim) = play.and_then(|play| play.aim.as_ref()) {
        looked(skeleton, aim, &mut scratch.base, &mut scratch.turns)?;
    }
    if let Some(aim) = play.and_then(|play| play.aim.as_ref())
        && aim.pitch != 0.0
        && !aim.chain.is_empty()
    {
        let share = aim.pitch / aim.chain.len() as f32;
        let (sin, cos) = engine_core::math::sin_cos(-share * 0.5);
        let turn = Quat::from_xyzw(sin, 0.0, 0.0, cos);
        scratch.turns.clear();
        for name in &aim.chain {
            let joint = skeleton.find(name).ok_or_else(|| {
                Error::new(ErrorCode::UnknownReference, "$.figure_play.aim.chain").got(name)
            })?;
            scratch.turns.push((joint, turn));
        }
        engine_rig::turn(skeleton, &mut scratch.base, &scratch.turns)
            .map_err(|error| rigged(&error))?;
    }
    if let Some(drop) = play.and_then(|play| play.crouch)
        && drop > 0.0
    {
        crouched(rig, drop, &mut scratch.base)?;
    }
    if let Some(stride) = play.and_then(|play| play.base.stride)
        && (stride.scale - 1.0).abs() > 1.0e-3
    {
        strode(rig, stride, &mut scratch.base)?;
    }
    if let Some(contact) = play.and_then(|play| play.pin_feet) {
        pinned(rig, played, contact, scratch)?;
    }
    if let Some(hold) = play.and_then(|play| play.hold.as_ref())
        && hold.weight > 0.0
    {
        held_hands(rig, hold, world, &mut scratch.base)?;
    }
    Ok(())
}

fn world_of(local: Option<&Transform>, parent: Option<Mat4>) -> Mat4 {
    let local = local.map_or(Mat4::IDENTITY, |local| {
        Mat4::from_scale_rotation_translation(local.scale, local.rotation, local.translation)
    });
    parent.map_or(local, |parent| parent * local)
}

fn joint_at(world: Mat4, posed: Option<&Posed>) -> JointAt {
    let (_, turned, _) = world.to_scale_rotation_translation();
    match posed {
        Some(joint) => JointAt {
            position: world.transform_point3(joint.position),
            rotation: turned * joint.rotation,
            found: true,
        },
        None => JointAt {
            position: world.transform_point3(Vec3::ZERO),
            rotation: turned,
            found: false,
        },
    }
}

fn measured(joints: &mut FigureJoints, rig: &Rig) {
    if joints.strides.len() == rig.clips.len() {
        return;
    }
    joints.strides = rig
        .clips
        .iter()
        .map(|(name, played)| (name.clone(), played.stride_mps))
        .collect();
    joints.steps = rig
        .clips
        .iter()
        .map(|(name, played)| (name.clone(), played.stride_mps * played.duration() * 0.5))
        .collect();
}

fn head_top(rig: &Rig, pose: &[Posed], world: Mat4, top: f32) -> Option<Vec3> {
    let head = rig.head?;
    let inverse_bind = rig.skeleton.joints().get(head)?.inverse_bind;
    let bound = inverse_bind.inverse().w_axis.truncate();
    let local = inverse_bind.transform_point3(Vec3::new(bound.x, top, bound.z));
    let joint = pose.get(head)?;
    Some(world.transform_point3(joint.position + joint.rotation * local))
}

fn held_local(joint: &Posed, held: &Held) -> Transform {
    let offset = Quat::from_array(held.offset.rotation).normalize();
    Transform {
        translation: joint.position + joint.rotation * Vec3::from_array(held.offset.translation),
        rotation: (joint.rotation * offset).normalize(),
        scale: Vec3::ONE,
    }
}

fn follow_figures(
    files: &mut FigureFiles,
    clocks: &mut FigureClocks,
    figures: &[(Entity, Figure)],
) {
    clocks
        .held
        .retain(|entity, _| figures.iter().any(|(known, _)| known == entity));
    for (entity, figure) in figures {
        if clocks
            .held
            .get(entity)
            .is_some_and(|held| held.figure == *figure)
        {
            continue;
        }
        let rig = files.rig(figure).map(Arc::new);
        clocks
            .held
            .insert(*entity, Clocks::new(figure.clone(), rig));
    }
}

type TickReads<'w> = (
    Res<'w, FixedTime>,
    Res<'w, GlobalTransforms>,
    Option<Res<'w, AssetLibrary>>,
);

type TickState<'w> = (
    ResMut<'w, FigureFiles>,
    ResMut<'w, FigureClocks>,
    ResMut<'w, FiguresStats>,
);

type TickFigures<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Figure,
        Option<&'static FigurePlay>,
        Option<&'static mut FigureJoints>,
        Option<&'static Drawn>,
    ),
>;

type Holders<'w, 's> = (
    Query<'w, 's, (Entity, &'static Held, Option<&'static Parent>)>,
    Query<'w, 's, &'static Parent, With<Figure>>,
);

type Report<'w, 's> = (
    Local<'s, Scratch>,
    ResMut<'w, Refusals>,
    Spawner<'w, 's, FiguresBrick>,
);

pub fn figure_tick(
    (fixed, globals, library): TickReads,
    (mut files, mut clocks, mut stats): TickState,
    mut figures: TickFigures,
    (holders, parents): Holders,
    mut moves: Moves<FiguresBrick>,
    (mut scratch, mut refusals, mut spawner): Report,
) {
    let mut listed: Vec<(Entity, Figure)> = figures
        .iter()
        .map(|(entity, figure, ..)| (entity, figure.clone()))
        .collect();
    listed.sort_unstable_by_key(|(entity, _)| *entity);
    follow_figures(&mut files, &mut clocks, &listed);
    refusals.keep_part(FIGURES_PART, |entity| clocks.held.contains_key(&entity));
    refusals.keep_part(HELD_PART, |entity| holders.contains(entity));
    let mut carried: Carried = BTreeMap::new();
    for (entity, held, parent) in &holders {
        match held
            .target()
            .filter(|target| clocks.held.contains_key(target))
        {
            Some(target) => {
                carried
                    .entry(target)
                    .or_default()
                    .push((entity, held, parent.map(Parent::parent)))
            }
            None => refusals.refuse(
                entity,
                HELD_PART,
                Error::new(ErrorCode::UnknownReference, "$.held.figure")
                    .got(held.figure.clone().unwrap_or_default()),
            ),
        }
    }
    let delta = fixed.delta_seconds;
    let mut ticked = 0;
    for (entity, _) in &listed {
        let Ok((_, _, play, joints, drawn)) = figures.get_mut(*entity) else {
            continue;
        };
        let Some(clock) = clocks.held.get_mut(entity) else {
            continue;
        };
        clock.tick(play, delta);
        let rig = match &clock.rig {
            Ok(rig) => Arc::clone(rig),
            Err(error) => {
                refusals.refuse(*entity, FIGURES_PART, error.clone());
                continue;
            }
        };
        let named = named_clips(&rig, play);
        let wanted = carried.get(entity);
        if named.is_err() || (joints.is_none() && wanted.is_none()) {
            refusals.judge(*entity, FIGURES_PART, named);
            continue;
        }
        let parent = parents
            .get(*entity)
            .ok()
            .and_then(|parent| globals.get(parent.parent()))
            .and_then(|global| global.matrix());
        let world = world_of(moves.local(*entity), parent);
        let made = posed(
            (&*rig, clock, play),
            (&mut files, Some(world)),
            0.0,
            &mut scratch,
        );
        if let Err(error) = made {
            refusals.refuse(*entity, FIGURES_PART, error);
            continue;
        }
        refusals.accept(*entity, FIGURES_PART);
        ticked += 1;
        let pose = &scratch.base;
        if let Some(mut joints) = joints {
            let filled: Vec<JointAt> = joints
                .names
                .iter()
                .map(|name| joint_at(world, rig.skeleton.find(name).map(|at| &pose[at])))
                .collect();
            let readout = clock.readout();
            let top = drawn
                .and_then(|drawn| drawn.geometry)
                .and_then(|mesh| library.as_deref()?.mesh(mesh))
                .map(|mesh| mesh.bounds().max().y);
            joints.head_top = top.and_then(|top| head_top(&rig, pose, world, top));
            joints.joints = filled;
            joints.tick = fixed.tick;
            joints.base_seconds = readout.base_seconds;
            joints.base_done = readout.base_done;
            joints.base_cue = readout.base_cue;
            joints.upper_seconds = readout.upper_seconds;
            joints.upper_done = readout.upper_done;
            joints.upper_cue = readout.upper_cue;
            measured(&mut joints, &rig);
        }
        for &(held_entity, held, parent) in wanted.into_iter().flatten() {
            let Some(joint) = rig.skeleton.find(&held.joint) else {
                refusals.refuse(
                    held_entity,
                    HELD_PART,
                    Error::new(ErrorCode::UnknownReference, "$.held.joint").got(&held.joint),
                );
                continue;
            };
            refusals.accept(held_entity, HELD_PART);
            if parent != Some(*entity) {
                spawner.set_parent(held_entity, Some(*entity));
            }
            if let Some(mut transform) = moves.get_mut(held_entity) {
                *transform = held_local(&pose[joint], held);
            }
        }
    }
    stats.ticked = ticked;
}

fn placed(drawn: &Drawn, base: f32) -> Option<Drawn> {
    (drawn.values.floats.first() != Some(&base)).then(|| {
        let mut floats = drawn.values.floats.clone();
        match floats.first_mut() {
            Some(first) => *first = base,
            None => floats.push(base),
        }
        Drawn {
            values: InstanceValues {
                floats,
                ..drawn.values.clone()
            },
            ..drawn.clone()
        }
    })
}

type DrawReads<'w> = (
    Res<'w, FrameTime>,
    Res<'w, GlobalTransforms>,
    Res<'w, FigureClocks>,
);

type DrawFigures<'w, 's> =
    Query<'w, 's, (Entity, Option<&'static FigurePlay>, &'static Drawn), With<Figure>>;

pub(super) fn figure_draw(
    (time, globals, clocks): DrawReads,
    mut files: ResMut<FigureFiles>,
    figures: DrawFigures,
    (mut drawing, mut stats): (ResMut<FigureDrawing>, ResMut<FiguresStats>),
    (mut scratch, mut refusals, mut spawner): Report,
) {
    drawing.written.clear();
    refusals.keep_part(DRAW_PART, |entity| figures.contains(entity));
    let mut listed: Vec<(Entity, Option<&FigurePlay>, &Drawn)> = figures
        .iter()
        .filter(|(entity, ..)| globals.drawn(*entity))
        .collect();
    listed.sort_unstable_by_key(|(entity, ..)| *entity);
    let back = 1.0 - time.alpha;
    let mut heads = Vec::with_capacity(listed.len());
    let mut drawn_now = Vec::with_capacity(listed.len());
    for (entity, play, drawn) in listed {
        let Some(clock) = clocks.held.get(&entity) else {
            continue;
        };
        let Ok(rig) = &clock.rig else {
            continue;
        };
        if named_clips(rig, play).is_err() {
            continue;
        }
        let world = globals.get(entity).and_then(|global| global.matrix());
        let made = posed(
            (&**rig, clock, play),
            (&mut files, world),
            back,
            &mut scratch,
        )
        .and_then(|()| {
            let Scratch { base, model, .. } = &mut *scratch;
            model.clear();
            model.resize(base.len(), Mat4::IDENTITY);
            engine_rig::model_matrices(base, model).map_err(|error| rigged(&error))?;
            let at = drawing.slot(entity, base.len());
            engine_rig::palette(&rig.skeleton, model, &mut drawing.rows[at..at + base.len()])
                .map_err(|error| rigged(&error))?;
            Ok((at, base.len()))
        });
        match made {
            Ok((at, joints)) => {
                drawn_now.push(entity);
                drawing.written.push((at, joints));
                refusals.accept(entity, DRAW_PART);
                if let Some(moved) = placed(drawn, at as f32) {
                    spawner.insert(entity, moved);
                }
                let seconds = clock
                    .base
                    .as_ref()
                    .map_or(0.0, |base| base.at(rig.duration(&base.clip), back));
                heads.push(FigureHead {
                    entity: entity.to_string(),
                    time: seconds,
                    head: rig
                        .head
                        .map(|joint| scratch.base[joint].position.to_array()),
                });
            }
            Err(error) => refusals.refuse(entity, DRAW_PART, error),
        }
    }
    drawing.keep(&drawn_now);
    let joints: usize = drawing.written.iter().map(|(_, length)| length).sum();
    let (skeletons, clips) = files.counts();
    stats.sampled = drawn_now.len() as u32;
    stats.joints = joints as u32;
    stats.palette_bytes = joints as u64 * ROW_BYTES;
    stats.skeletons = skeletons;
    stats.clips = clips;
    stats.heads = heads;
}

type Refusing<'w, 's> = (
    Res<'w, Refusals>,
    Query<'w, 's, Entity, Or<(With<Figure>, With<Held>)>>,
);

pub(super) fn count(
    (table, library): (Res<DrawnTable>, Option<Res<AssetLibrary>>),
    lights: Query<(), With<Light>>,
    (refusals, owned): Refusing,
    mut stats: ResMut<FiguresStats>,
) {
    let mut refused = Vec::new();
    for entity in &owned {
        for held in refusals.of(entity) {
            if [FIGURES_PART, DRAW_PART, HELD_PART].contains(&held.part) {
                refused.push(FigureRefused {
                    entity: entity.to_string(),
                    part: held.part.to_owned(),
                    reason: serde_json::to_string(&held.reason).unwrap_or_default(),
                });
            }
        }
    }
    refused.sort_unstable_by(|a, b| (&a.entity, &a.part).cmp(&(&b.entity, &b.part)));
    if stats.refused != refused {
        stats.refused = refused;
    }
    let (mut figures, mut indices) = (0_u32, 0_u64);
    for kind in [FIGURE, GROUND] {
        for batch in table.batches(kind) {
            let instances = batch.instances.len() as u64;
            if kind == FIGURE {
                figures += instances as u32;
            }
            let mesh = batch
                .geometry
                .zip(library.as_deref())
                .and_then(|(geometry, library)| library.mesh(geometry));
            if let Some(mesh) = mesh {
                indices += mesh.indices().len() as u64 * instances;
            }
        }
    }
    stats.figures = figures;
    stats.indices = indices;
    stats.triangles = indices / 3;
    stats.lit = !lights.is_empty();
}
