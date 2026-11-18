use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, Quat, Vec3};
use engine_core::{
    Asset, AssetLibrary, AssetResidency, Drawn, DrawnTable, Error, ErrorCode, FixedTime, FrameTime,
    GeometryHandle, GlobalTransforms, InstanceValues, Interest, Moves, Parent, Refusals,
    SceneContext, Spawner, Transform,
};
use engine_rig::{PaletteRow, Posed, Reach};

use super::clock::{Clocks, FigureClocks, Readout, Upper};
use super::files::{Crown, FigureFiles, Played, Rig, unreadable};
use super::{
    FIGURES_PART, FigureForms, FigureHead, FigureRefused, FigureView, FiguresBrick, FiguresStats,
};
use crate::play::figures::{
    Blended, FIGURE, Figure, FigureJoints, FigurePlay, Form, GROUND, HIP, Held, Hold, HoldSpace,
    JointAt, LEFT_ARM, LEFT_LEG, RIGHT_ARM, RIGHT_LEG, Stride,
};
use crate::play::light::Light;
use crate::play::numbers::Numbers;
use crate::play::puppet::{PuppetFile, Seen};
use crate::play::screen::Covered;
use engine_core::math::{max, min};

const DRAW_PART: &str = "figures_draw";

const FORMS_PART: &str = "figures_forms";

const HELD_PART: &str = "figures_held";

const ROW_BYTES: u64 = 48;

type Carried<'a> = BTreeMap<Entity, Vec<(Entity, &'a Held, Option<Entity>)>>;

#[derive(Default)]
pub struct Scratch {
    base: Vec<Posed>,
    other: Vec<Posed>,
    out: Vec<Posed>,
    first: Vec<Posed>,
    held: Vec<Posed>,
    weights: Vec<f32>,
    turns: Vec<(usize, Quat)>,
    model: Vec<Mat4>,
    shift: Vec3,
}

#[derive(Resource, Default)]
pub struct FigureDrawing {
    pub(super) rows: Vec<PaletteRow>,
    pub(super) written: Vec<(usize, usize)>,
    slots: BTreeMap<Entity, (usize, usize)>,
    full: BTreeMap<Entity, GeometryHandle>,
    forms: BTreeMap<Entity, (Vec<GeometryHandle>, Vec<Interest<Asset>>)>,
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

fn chest_framed(
    (rig, reference): (&Rig, &Played),
    (joint, pitch, weight): (usize, f32, f32),
    scratch: &mut Scratch,
) -> Result<Vec3, Error> {
    let skeleton = &*rig.skeleton;
    reference
        .bound
        .sample(&reference.clip, 0.0, skeleton, &mut scratch.other)
        .map_err(|error| rigged(&error))?;
    let wanted = scratch.other[joint];
    let now = scratch.base[joint];
    let (sin, cos) = engine_core::math::sin_cos(-pitch * 0.5);
    let aimed = (Quat::from_xyzw(sin, 0.0, 0.0, cos) * wanted.rotation).normalize();
    let turn = Quat::IDENTITY.slerp((aimed * now.rotation.inverse()).normalize(), weight);
    engine_rig::turn(skeleton, &mut scratch.base, &[(joint, turn)])
        .map_err(|error| rigged(&error))?;
    Ok(now.position - wanted.position)
}

fn held_hands(
    rig: &Rig,
    (hold, pitch): (&Hold, f32),
    world: Option<Mat4>,
    scratch: &mut Scratch,
) -> Result<(), Error> {
    scratch.shift = Vec3::ZERO;
    if let Some(chest) = &hold.chest {
        let joint = joint_named(&rig.skeleton, &chest.joint)?;
        let reference = rig
            .clips
            .get(&chest.reference)
            .ok_or_else(|| unknown("$.figure_play.hold.chest.reference", &chest.reference, rig))?;
        scratch.shift = chest_framed((rig, reference), (joint, pitch, hold.weight), scratch)?;
    }
    let shift = scratch.shift;
    let posed = &mut scratch.base;
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
            target: to_model.transform_point3(Vec3::from_array(hand.position)) + shift,
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
    scratch.shift = Vec3::ZERO;
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
    if let Some(Upper { clock, layer }) = play.and(clocks.upper.as_ref())
        && layer.weight > 0.0
        && !layer.additive
    {
        let Some(over) = rig.clips.get(&clock.clip) else {
            return Err(unknown("$.figure_play.upper.clip", &clock.clip, rig));
        };
        let at = clock.at(over.duration(), back);
        over.bound
            .sample(&over.clip, at, skeleton, &mut scratch.other)
            .map_err(|error| rigged(&error))?;
        let mask = files.mask(rig, &layer.from, layer.feather, "$.figure_play.upper.from")?;
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
        let pitch = play
            .and_then(|play| play.aim.as_ref())
            .map_or(0.0, |aim| aim.pitch);
        held_hands(rig, (hold, pitch), world, scratch)?;
    }
    if play.is_some() {
        layered(rig, clocks, files, back, scratch)?;
    }
    Ok(())
}

fn additive(upper: Option<&Upper>) -> Option<&Upper> {
    upper.filter(|upper| upper.layer.additive && upper.layer.weight > 0.0)
}

fn layered(
    rig: &Rig,
    clocks: &Clocks,
    files: &mut FigureFiles,
    back: f32,
    scratch: &mut Scratch,
) -> Result<(), Error> {
    let now = additive(clocks.upper.as_ref());
    let fade = clocks.upper_fade.as_ref();
    let from = fade.and_then(|fade| additive(fade.from.as_ref()));
    if now.is_none() && from.is_none() {
        return Ok(());
    }
    let Scratch {
        base,
        other,
        out,
        weights,
        first,
        held,
        ..
    } = scratch;
    held.clone_from(base);
    first.clone_from(base);
    if let Some(upper) = now {
        added(
            rig,
            upper,
            files,
            back,
            (&held[..], &mut base[..]),
            (&mut other[..], &mut first[..]),
        )?;
    }
    let Some(fade) = fade else {
        return Ok(());
    };
    out.clone_from(held);
    if let Some(upper) = from {
        added(
            rig,
            upper,
            files,
            back,
            (&held[..], &mut out[..]),
            (&mut other[..], &mut first[..]),
        )?;
    }
    weights.fill(fade.weight(back));
    engine_rig::blend(&rig.skeleton, out, base, weights, None, other)
        .map_err(|error| rigged(&error))?;
    std::mem::swap(base, other);
    Ok(())
}

fn added(
    rig: &Rig,
    Upper { clock, layer }: &Upper,
    files: &mut FigureFiles,
    back: f32,
    (under, pose): (&[Posed], &mut [Posed]),
    (clip, first): (&mut [Posed], &mut [Posed]),
) -> Result<(), Error> {
    let skeleton = &*rig.skeleton;
    let over = rig
        .clips
        .get(&clock.clip)
        .ok_or_else(|| unknown("$.figure_play.upper.clip", &clock.clip, rig))?;
    over.bound
        .sample(&over.clip, clock.at(over.duration(), back), skeleton, clip)
        .map_err(|error| rigged(&error))?;
    over.bound
        .sample(&over.clip, 0.0, skeleton, first)
        .map_err(|error| rigged(&error))?;
    let (clip, first) = (&*clip, &*first);
    let mask = files.mask(rig, &layer.from, layer.feather, "$.figure_play.upper.from")?;
    let played = match &layer.played_from {
        Some(from) => Some(files.mask(rig, from, 0, "$.figure_play.upper.played_from")?),
        None => None,
    };
    for (joint, bone) in skeleton.joints().iter().enumerate() {
        let weight = mask[joint] * layer.weight;
        let Some(parent) = bone.parent.filter(|_| weight > 0.0) else {
            continue;
        };
        let local = |posed: &[Posed]| posed[parent].rotation.conjugate() * posed[joint].rotation;
        let reach =
            under[parent].rotation.conjugate() * (under[joint].position - under[parent].position);
        let rotation = if played.as_ref().is_some_and(|played| played[joint] > 0.0) {
            local(under).lerp(local(clip), weight)
        } else {
            let delta = (local(clip) * local(first).conjugate()).normalize();
            Quat::IDENTITY.lerp(delta, weight) * local(under)
        };
        let top = pose[parent];
        pose[joint] = Posed {
            rotation: (top.rotation * rotation).normalize(),
            position: top.position + top.rotation * reach,
        };
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

struct Found {
    crown: Option<Crown>,
    rest_eye: Option<f32>,
    fit: f32,
    shift: Vec3,
    tick: u64,
    readout: Readout,
}

fn published(joints: &mut FigureJoints, (rig, pose, world): (&Rig, &[Posed], Mat4), found: &Found) {
    joints.joints = joints
        .names
        .iter()
        .map(|name| joint_at(world, rig.skeleton.find(name).map(|at| &pose[at])))
        .collect();
    let crown = found.crown;
    joints.head_top = crown.and_then(|crown| head_top(rig, pose, world, crown.top));
    joints.crown = crown.and_then(|crown| head_top(rig, pose, world, crown.crown));
    joints.hold_shift = found.shift;
    joints.fit = found.fit;
    joints.rest_eye_m = found
        .rest_eye
        .map(|eye| eye * world_height(world) / found.fit);
    joints.tick = found.tick;
    let readout = &found.readout;
    joints.base_seconds = readout.base_seconds;
    joints.base_done = readout.base_done;
    joints.base_cue = readout.base_cue;
    joints.upper_seconds = readout.upper_seconds;
    joints.upper_done = readout.upper_done;
    joints.upper_cue = readout.upper_cue;
    joints.upper_rate = readout.upper_rate;
    measured(joints, rig);
}

fn measured(joints: &mut FigureJoints, rig: &Rig) {
    if joints.strides.len() == rig.clips.len() {
        return;
    }
    joints.strides = rig
        .clips
        .iter()
        .map(|(name, played)| (name.clone(), played.motion.stride_mps))
        .collect();
    joints.steps = rig
        .clips
        .iter()
        .map(|(name, played)| {
            (
                name.clone(),
                played.motion.stride_mps * played.duration() * 0.5,
            )
        })
        .collect();
    joints.drifts = rig
        .clips
        .iter()
        .map(|(name, played)| (name.clone(), played.motion.drift_m))
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

fn world_height(world: Mat4) -> f32 {
    world.y_axis.truncate().length()
}

fn fitted(rest_eye: Option<f32>, eye: Option<f32>, world: Mat4) -> f32 {
    match (rest_eye, eye) {
        (Some(rest), Some(eye)) if rest > 0.0 && world_height(world) > 0.0 => {
            eye / (rest * world_height(world))
        }
        (Some(_) | None, Some(_) | None) => 1.0,
    }
}

fn held_local(joint: &Posed, held: &Held, fit: f32) -> Transform {
    let offset = Quat::from_array(held.offset.rotation).normalize();
    Transform {
        translation: (joint.position + joint.rotation * Vec3::from_array(held.offset.translation))
            * fit,
        rotation: (joint.rotation * offset).normalize(),
        scale: Vec3::splat(held.offset.scale),
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

type TickReads<'w, 's> = (
    Res<'w, FixedTime>,
    Res<'w, GlobalTransforms>,
    (
        Option<Res<'w, AssetLibrary>>,
        Query<'w, 's, &'static PuppetFile>,
        Res<'w, Numbers>,
    ),
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
    (fixed, globals, (library, puppet, numbers)): TickReads,
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
        let book = puppet.iter().find_map(PuppetFile::book);
        let fit_to = book
            .and_then(|book| book.table.eye_fit.as_ref())
            .zip(numbers.get().map(|numbers| numbers.player.eye_height_m));
        let rest_eye = fit_to.and_then(|(fit, _)| files.rest_eye(&rig, &fit.reference));
        clock.fit = fitted(rest_eye, fit_to.map(|(_, eye)| eye), world);
        let world = world * Mat4::from_scale(Vec3::splat(clock.fit));
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
            let check = book.map(|book| book.table.head_check);
            let crown = drawn
                .and_then(|drawn| drawn.geometry)
                .zip(library.as_deref().zip(check))
                .and_then(|(mesh, (library, check))| files.crown(&rig, (library, mesh), check));
            let found = Found {
                crown,
                rest_eye,
                fit: clock.fit,
                shift: scratch.shift,
                tick: fixed.tick,
                readout: clock.readout(),
            };
            published(&mut joints, (&rig, pose, world), &found);
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
                *transform = held_local(&pose[joint], held, clock.fit);
            }
        }
    }
    stats.ticked = ticked;
}

fn submitted(
    library: Option<&AssetLibrary>,
    drawn: &Drawn,
    geometry: Option<GeometryHandle>,
) -> u64 {
    let count = geometry
        .and_then(|mesh| library?.mesh(mesh))
        .map_or(0, |mesh| mesh.vertices().len() as u64);
    let outlined = u64::from(drawn.values.emission != [0.0; 3]);
    count * (1 + outlined)
}

fn placed(drawn: &Drawn, base: f32, geometry: Option<GeometryHandle>) -> Option<Drawn> {
    let geometry = geometry.or(drawn.geometry);
    (drawn.values.floats.first() != Some(&base) || drawn.geometry != geometry).then(|| {
        let mut floats = drawn.values.floats.clone();
        match floats.first_mut() {
            Some(first) => *first = base,
            None => floats.push(base),
        }
        Drawn {
            geometry,
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
    (Option<Res<'w, Seen>>, Option<Res<'w, AssetLibrary>>),
);

type DrawFigures<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Option<&'static FigurePlay>,
        &'static Drawn,
        &'static Figure,
    ),
>;

type Placing<'w> = (
    Option<Res<'w, SceneContext>>,
    Option<Res<'w, AssetResidency>>,
    Option<Res<'w, Covered>>,
);

struct Shelf<'a> {
    scene: Option<&'a SceneContext>,
    residency: Option<&'a AssetResidency>,
    library: Option<&'a AssetLibrary>,
}

fn held_forms(
    shelf: &Shelf<'_>,
    figures: &DrawFigures,
    held: &mut BTreeMap<Entity, (Vec<GeometryHandle>, Vec<Interest<Asset>>)>,
) -> Vec<FigureForms> {
    held.retain(|entity, _| figures.contains(*entity));
    let (Some(scene), Some(residency)) = (shelf.scene, shelf.residency) else {
        return Vec::new();
    };
    let mut listed = Vec::new();
    for (entity, _, _, figure) in figures {
        let Some(forms) = &figure.forms else {
            held.remove(&entity);
            continue;
        };
        let handle = |name: &String| scene.assets.meshes.get(name).copied();
        let (mid, far) = (handle(&forms.mid), handle(&forms.far));
        let handles: Vec<GeometryHandle> = mid.into_iter().chain(far).collect();
        if held.get(&entity).is_none_or(|(known, _)| *known != handles) {
            let interests = handles.iter().map(|&mesh| residency.hold(mesh)).collect();
            held.insert(entity, (handles, interests));
        }
        let resident =
            |mesh: Option<GeometryHandle>| mesh.is_some_and(|mesh| residency.resident(mesh));
        listed.push(FigureForms {
            entity: entity.to_string(),
            named: mid.is_some() && far.is_some(),
            mid: resident(mid),
            far: resident(far),
        });
    }
    listed
}

fn viewed(
    (seen, current): (Option<&Seen>, Option<&Seen>),
    (drawn, listed): (&[Entity], usize),
    (forms, vertices): ([u32; 3], u64),
    posed: u32,
) -> FigureView {
    let distances = drawn
        .iter()
        .filter_map(|entity| current.and_then(|seen| seen.distance(*entity)));
    let (nearest_m, farthest_m) = distances.fold((0.0, 0.0), |(nearest, farthest), distance| {
        if farthest == 0.0 {
            (distance, distance)
        } else {
            (min(nearest, distance), max(farthest, distance))
        }
    });
    FigureView {
        posed,
        drawn: drawn.len() as u32,
        out_of_view: seen.map_or(0, |seen| seen.frustum),
        behind_map: seen.map_or(0, |seen| seen.map),
        held: seen.map_or(0, |seen| seen.held),
        full: forms[0],
        mid: forms[1],
        far: forms[2],
        vertices,
        nearest_m,
        farthest_m,
        unposed: (listed - drawn.len()) as u32,
    }
}

fn formed(
    shelf: &Shelf<'_>,
    (entity, figure, drawn, form): (Entity, &Figure, &Drawn, Form),
    (rig, full, files): (
        &Rig,
        &mut BTreeMap<Entity, GeometryHandle>,
        &mut FigureFiles,
    ),
) -> Result<Option<GeometryHandle>, Error> {
    let Some(current) = drawn.geometry else {
        return Ok(None);
    };
    let mesh = |name: &String| {
        shelf
            .scene
            .and_then(|scene| scene.assets.meshes.get(name).copied())
    };
    let reduced = figure.forms.as_ref().is_some_and(|forms| {
        [&forms.mid, &forms.far]
            .into_iter()
            .any(|name| mesh(name) == Some(current))
    });
    let whole = if reduced {
        full.get(&entity).copied().unwrap_or(current)
    } else {
        full.insert(entity, current);
        current
    };
    let (name, path) = match (form, &figure.forms) {
        (Form::Mid, Some(forms)) => (&forms.mid, "$.figure.forms.mid"),
        (Form::Far, Some(forms)) => (&forms.far, "$.figure.forms.far"),
        (Form::Full | Form::Mid | Form::Far, _) => return Ok(Some(whole)),
    };
    let handle =
        mesh(name).ok_or_else(|| Error::new(ErrorCode::UnknownReference, path).got(name))?;
    if !shelf
        .residency
        .is_some_and(|residency| residency.resident(handle))
    {
        return Ok(Some(whole));
    }
    let Some(library) = shelf.library else {
        return Ok(Some(whole));
    };
    files.form_fits(rig, (library, handle), path)?;
    Ok(Some(handle))
}

fn chosen(
    shelf: &Shelf<'_>,
    (seen, entity, figure, drawn): (Option<&Seen>, Entity, &Figure, &Drawn),
    (rig, full, files): (
        &Rig,
        &mut BTreeMap<Entity, GeometryHandle>,
        &mut FigureFiles,
    ),
    refusals: &mut Refusals,
) -> (Option<GeometryHandle>, Form) {
    let form = seen.and_then(|seen| seen.form(entity)).unwrap_or_default();
    let geometry = match formed(shelf, (entity, figure, drawn, form), (rig, full, files)) {
        Ok(geometry) => {
            refusals.accept(entity, FORMS_PART);
            geometry
        }
        Err(error) => {
            refusals.refuse(entity, FORMS_PART, error);
            full.get(&entity).copied()
        }
    };
    let shown = if geometry == full.get(&entity).copied() {
        Form::Full
    } else {
        form
    };
    (geometry, shown)
}

pub(super) fn figure_draw(
    (time, globals, clocks, (seen, library)): DrawReads,
    (mut files, (scene, residency, covered)): (ResMut<FigureFiles>, Placing),
    figures: DrawFigures,
    (mut drawing, mut stats): (ResMut<FigureDrawing>, ResMut<FiguresStats>),
    (mut scratch, mut refusals, mut spawner): Report,
) {
    drawing.written.clear();
    refusals.keep_part(DRAW_PART, |entity| figures.contains(entity));
    refusals.keep_part(FORMS_PART, |entity| figures.contains(entity));
    if covered.as_deref().is_some_and(|covered| covered.0) {
        stats.view = FigureView {
            posed: stats.ticked,
            ..FigureView::default()
        };
        return;
    }
    let shelf = Shelf {
        scene: scene.as_deref(),
        residency: residency.as_deref(),
        library: library.as_deref(),
    };
    let forms = held_forms(&shelf, &figures, &mut drawing.forms);
    if stats.forms != forms {
        stats.forms = forms;
    }
    let mut forms_drawn = [0_u32; 3];
    let current = seen.as_deref().filter(|seen| seen.frame == time.frame);
    let mut listed: Vec<(Entity, Option<&FigurePlay>, &Drawn, &Figure)> = figures
        .iter()
        .filter(|(entity, ..)| match current {
            Some(seen) if seen.judges(*entity) => seen.shows(*entity),
            Some(_) | None => globals.drawn(*entity),
        })
        .collect();
    listed.sort_unstable_by_key(|(entity, ..)| *entity);
    let listed_count = listed.len();
    let mut vertices = 0_u64;
    let back = 1.0 - time.alpha;
    let mut heads = Vec::with_capacity(listed.len());
    let mut drawn_now = Vec::with_capacity(listed.len());
    for (entity, play, drawn, figure) in listed {
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
            let fit = Mat4::from_scale(Vec3::splat(clock.fit));
            for matrix in model.iter_mut() {
                *matrix = fit * *matrix;
            }
            let at = drawing.slot(entity, base.len());
            engine_rig::palette(&rig.skeleton, model, &mut drawing.rows[at..at + base.len()])
                .map_err(|error| rigged(&error))?;
            Ok((at, base.len()))
        });
        match made {
            Ok((at, joints)) => {
                drawn_now.push(entity);
                let (geometry, shown) = chosen(
                    &shelf,
                    (seen.as_deref(), entity, figure, drawn),
                    (rig, &mut drawing.full, &mut files),
                    &mut refusals,
                );
                forms_drawn[shown as usize] += 1;
                vertices += submitted(library.as_deref(), drawn, geometry);
                drawing.written.push((at, joints));
                refusals.accept(entity, DRAW_PART);
                if let Some(moved) = placed(drawn, at as f32, geometry) {
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
    drawing.full.retain(|entity, _| figures.contains(*entity));
    let joints: usize = drawing.written.iter().map(|(_, length)| length).sum();
    let (skeletons, clips) = files.counts();
    stats.sampled = drawn_now.len() as u32;
    stats.view = viewed(
        (seen.as_deref(), current),
        (&drawn_now, listed_count),
        (forms_drawn, vertices),
        stats.ticked,
    );
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
    (lights, files): (Query<(), With<Light>>, Query<&PuppetFile>),
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
    stats.maps = files
        .iter()
        .find_map(PuppetFile::book)
        .map(|book| book.table.shade.maps);
}
