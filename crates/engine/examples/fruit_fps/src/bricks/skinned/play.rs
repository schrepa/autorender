use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, Quat, Vec3};
use engine_core::{
    Assets, Drawn, Error, ErrorCode, Expected, FixedTime, FrameTime, GlobalTransforms, Moves,
    Parent, Refusals, Spawner, Stats, Transform, ViewSkip,
};
use engine_rig::{Clip, PaletteRow, Posed, Skeleton};

use super::clock::{Clocks, FigureClocks, Upper, clips_in};
use super::pose::{self, Reach};
use super::rig::{FigureFiles, Played, Rig, unreadable};
use super::{FigureHead, FigureRefused, SkinnedBrick, SkinnedStats};
use crate::play::skinned::{
    Aim, Blended, Figure, FigureClock, FigureFit, FigureJoints, FigureLimbs, FigurePalette,
    FigurePlay, FigureSight, Held, Hold, HoldSpace, JointAt, JointNo, StepJoints, StepPose,
    StepPoser, Stride,
};
use engine_core::math::max;

const FIGURES_PART: &str = "figures";

const DRAW_PART: &str = "figures_draw";

const HELD_PART: &str = "figures_held";

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

fn rigged(error: &Error) -> Error {
    unreadable("$.figure".to_owned(), error)
}

fn missing_limb(limb: &str) -> Error {
    Error::new(ErrorCode::UnknownReference, format!("$.figure_limbs.{limb}"))
        .expected(Expected::Present)
}

fn clip_at<'a>(rig: &'a Rig, clip: usize, path: &str) -> Result<&'a Played, Error> {
    rig.clips
        .get(clip)
        .ok_or_else(|| Error::new(ErrorCode::UnknownReference, path.to_owned()).got(clip))
}

fn crouched(rig: &Rig, drop: f32, posed: &mut [Posed]) -> Result<(), Error> {
    let hip = rig.limbs.hip.ok_or_else(|| missing_limb("hip"))?;
    let legs = rig.limbs.legs.ok_or_else(|| missing_limb("legs"))?;
    let planted = legs.map(|leg| posed[leg[2]]);
    pose::shift(rig, posed, hip, Vec3::new(0.0, -drop, 0.0)).map_err(|error| rigged(&error))?;
    for (leg, foot) in legs.into_iter().zip(planted) {
        let reach = Reach {
            target: foot.position,
            pole: posed[leg[1]].position - posed[leg[0]].position,
            end_rotation: Some(foot.rotation),
            weight: 1.0,
        };
        pose::reach(rig, posed, leg, &reach).map_err(|error| rigged(&error))?;
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
        let other = clip_at(rig, blended.clip.0, "$.figure_play.base.blend")?;
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
        pose::reach(rig, posed, leg, &reach).map_err(|error| rigged(&error))?;
    }
    Ok(())
}

fn joint_in(rig: &Rig, joint: JointNo, path: &str) -> Result<usize, Error> {
    (joint.0 < rig.skeleton.len())
        .then_some(joint.0)
        .ok_or_else(|| Error::new(ErrorCode::UnknownReference, path.to_owned()).got(joint.0))
}

fn looked(
    rig: &Rig,
    aim: &Aim,
    posed: &mut [Posed],
    turns: &mut Vec<(usize, Quat)>,
) -> Result<(), Error> {
    let Some(hips) = aim.hips else {
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
    turns.push((hips.0, about(-aim.yaw)));
    let share = aim.yaw / aim.chain.len() as f32;
    for joint in &aim.chain {
        turns.push((joint.0, about(share)));
    }
    pose::turn(rig, posed, turns).map_err(|error| rigged(&error))
}

fn pitched(rig: &Rig, aim: &Aim, scratch: &mut Scratch) -> Result<(), Error> {
    let share = aim.pitch / aim.chain.len() as f32;
    let (sin, cos) = engine_core::math::sin_cos(-share * 0.5);
    let turn = Quat::from_xyzw(sin, 0.0, 0.0, cos);
    scratch.turns.clear();
    for joint in &aim.chain {
        scratch.turns.push((joint.0, turn));
    }
    pose::turn(rig, &mut scratch.base, &scratch.turns).map_err(|error| rigged(&error))
}

fn pinned(rig: &Rig, played: &Played, contact: f32, scratch: &mut Scratch) -> Result<(), Error> {
    let Some(legs) = rig.limbs.legs else {
        return Ok(());
    };
    played
        .bound
        .sample(&played.clip, 0.0, &rig.skeleton, &mut scratch.other)
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
        pose::reach(rig, &mut scratch.base, leg, &reach).map_err(|error| rigged(&error))?;
    }
    Ok(())
}

fn chest_framed(
    (rig, reference): (&Rig, &Played),
    (joint, pitch, weight): (usize, f32, f32),
    scratch: &mut Scratch,
) -> Result<Vec3, Error> {
    reference
        .bound
        .sample(&reference.clip, 0.0, &rig.skeleton, &mut scratch.other)
        .map_err(|error| rigged(&error))?;
    let wanted = scratch.other[joint];
    let now = scratch.base[joint];
    let (sin, cos) = engine_core::math::sin_cos(-pitch * 0.5);
    let aimed = (Quat::from_xyzw(sin, 0.0, 0.0, cos) * wanted.rotation).normalize();
    let turn = Quat::IDENTITY.slerp((aimed * now.rotation.inverse()).normalize(), weight);
    pose::turn(rig, &mut scratch.base, &[(joint, turn)]).map_err(|error| rigged(&error))?;
    Ok(now.position - wanted.position)
}

fn held_hands(
    rig: &Rig,
    (hold, pitch): (&Hold, f32),
    world: Option<Mat4>,
    scratch: &mut Scratch,
) -> Result<(), Error> {
    scratch.shift = Vec3::ZERO;
    if let Some(chest) = hold.chest {
        let joint = joint_in(rig, chest.joint, "$.figure_play.hold.chest.joint")?;
        let reference = clip_at(rig, chest.reference.0, "$.figure_play.hold.chest.reference")?;
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
        (hold.right, rig.limbs.right_arm, "right_arm"),
        (hold.left, rig.limbs.left_arm, "left_arm"),
    ];
    for (hand, arm, limb) in hands {
        let Some(hand) = hand else {
            continue;
        };
        let arm = arm.ok_or_else(|| missing_limb(limb))?;
        let reach = Reach {
            target: to_model.transform_point3(Vec3::from_array(hand.position)) + shift,
            pole: to_model.transform_vector3(Vec3::from_array(hand.pole)),
            end_rotation: Some((turned * Quat::from_array(hand.rotation)).normalize()),
            weight: hold.weight,
        };
        pose::reach(rig, posed, arm, &reach).map_err(|error| rigged(&error))?;
    }
    Ok(())
}

fn posed(
    (rig, clocks, play): (&Rig, &Clocks, Option<&FigurePlay>),
    world: Option<Mat4>,
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
    let played = clip_at(rig, base.clip, "$.figure_play.base.clip")?;
    let at = base.at(played.duration(), back);
    played
        .bound
        .sample(&played.clip, at, skeleton, &mut scratch.base)
        .map_err(|error| rigged(&error))?;
    if let Some(fade) = &clocks.fade
        && let Some(from) = rig.clips.get(fade.from.clip)
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
    let Some(play) = play else {
        return Ok(());
    };
    if !play.base.blend.is_empty() {
        mixed(rig, (played, at), &play.base.blend, scratch)?;
    }
    replaced(rig, clocks, back, scratch)?;
    if let Some(aim) = &play.aim {
        looked(rig, aim, &mut scratch.base, &mut scratch.turns)?;
        if aim.pitch != 0.0 && !aim.chain.is_empty() {
            pitched(rig, aim, scratch)?;
        }
    }
    if let Some(drop) = play.crouch
        && drop > 0.0
    {
        crouched(rig, drop, &mut scratch.base)?;
    }
    if let Some(stride) = play.base.stride
        && (stride.scale - 1.0).abs() > 1.0e-3
    {
        strode(rig, stride, &mut scratch.base)?;
    }
    if let Some(contact) = play.pin_feet {
        pinned(rig, played, contact, scratch)?;
    }
    if let Some(hold) = play.hold.as_ref().filter(|hold| hold.weight > 0.0) {
        let pitch = play.aim.as_ref().map_or(0.0, |aim| aim.pitch);
        held_hands(rig, (hold, pitch), world, scratch)?;
    }
    layered(rig, clocks, back, scratch)
}

fn masked(mask: &Result<Arc<[f32]>, Error>) -> Result<&[f32], Error> {
    mask.as_deref().map_err(Clone::clone)
}

fn replaced(rig: &Rig, clocks: &Clocks, back: f32, scratch: &mut Scratch) -> Result<(), Error> {
    let Some(Upper {
        clock, layer, mask, ..
    }) = clocks
        .upper
        .as_ref()
        .filter(|upper| upper.layer.weight > 0.0 && !upper.layer.additive)
    else {
        return Ok(());
    };
    let skeleton = &*rig.skeleton;
    let over = clip_at(rig, clock.clip, "$.figure_play.upper.clip")?;
    let at = clock.at(over.duration(), back);
    over.bound
        .sample(&over.clip, at, skeleton, &mut scratch.other)
        .map_err(|error| rigged(&error))?;
    let eased = clocks
        .upper_fade
        .as_ref()
        .filter(|_| layer.fade_seconds > 0.0)
        .map_or(1.0, |fade| fade.weight(back));
    let mask = masked(mask)?;
    for (weight, masked) in scratch.weights.iter_mut().zip(mask.iter()) {
        *weight = masked * layer.weight * eased;
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
    Ok(())
}

fn additive(upper: Option<&Upper>) -> Option<&Upper> {
    upper.filter(|upper| upper.layer.additive && upper.layer.weight > 0.0)
}

fn layered(rig: &Rig, clocks: &Clocks, back: f32, scratch: &mut Scratch) -> Result<(), Error> {
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
    upper: &Upper,
    back: f32,
    (under, pose): (&[Posed], &mut [Posed]),
    (clip, first): (&mut [Posed], &mut [Posed]),
) -> Result<(), Error> {
    let skeleton = &*rig.skeleton;
    let (clock, layer) = (&upper.clock, &upper.layer);
    let over = clip_at(rig, clock.clip, "$.figure_play.upper.clip")?;
    over.bound
        .sample(&over.clip, clock.at(over.duration(), back), skeleton, clip)
        .map_err(|error| rigged(&error))?;
    over.bound
        .sample(&over.clip, 0.0, skeleton, first)
        .map_err(|error| rigged(&error))?;
    let (clip, first) = (&*clip, &*first);
    let mask = masked(&upper.mask)?;
    let played = upper.played.as_ref().map(masked).transpose()?;
    for (joint, bone) in skeleton.joints().iter().enumerate() {
        let weight = mask[joint] * layer.weight;
        let Some(parent) = bone.parent.filter(|_| weight > 0.0) else {
            continue;
        };
        let local = |posed: &[Posed]| posed[parent].rotation.conjugate() * posed[joint].rotation;
        let reach =
            under[parent].rotation.conjugate() * (under[joint].position - under[parent].position);
        let rotation = if played.is_some_and(|played| played[joint] > 0.0) {
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

fn head_top(rig: &Rig, pose: &[Posed], world: Mat4, top: f32) -> Option<Vec3> {
    let head = rig.limbs.head?;
    let inverse_bind = rig.skeleton.joints().get(head)?.inverse_bind;
    let bound = inverse_bind.inverse().w_axis.truncate();
    let local = inverse_bind.transform_point3(Vec3::new(bound.x, top, bound.z));
    let joint = pose.get(head)?;
    Some(world.transform_point3(joint.position + joint.rotation * local))
}

fn world_height(world: Mat4) -> f32 {
    world.y_axis.truncate().length()
}

fn fitted(rest: Option<f32>, height: Option<f32>, world: Mat4) -> f32 {
    match (rest, height) {
        (Some(rest), Some(height)) if rest > 0.0 && world_height(world) > 0.0 => {
            height / (rest * world_height(world))
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

fn bounded(rig: &Rig, clock: &Clocks, stride: Option<Stride>, world: Mat4) -> Option<[Vec3; 2]> {
    let extent = rig.extent;
    let ceiling = clock.top.map_or(extent.ceiling, |top| max(extent.ceiling, top));
    if extent.floor > ceiling {
        return None;
    }
    let radius = extent.radius * stride.map_or(1.0, |stride| max(stride.scale, 1.0));
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    for corner in 0..8_u8 {
        let pick = |bit: u8, (off, on): (f32, f32)| if corner & bit == 0 { off } else { on };
        let local = Vec3::new(
            pick(1, (-radius, radius)),
            pick(2, (extent.floor, ceiling)),
            pick(4, (-radius, radius)),
        );
        let at = world.transform_point3(local);
        low = low.min(at);
        high = high.max(at);
    }
    Some([low, high])
}

fn clocked(facts: &mut FigureClock, (rig, clock): (&Rig, &Clocks), bounds: Option<[Vec3; 2]>) {
    clock.read(rig, facts);
    facts.fit_rest_m = clock
        .fit_rest
        .zip(clock.world)
        .map(|(rest, world)| rest * world_height(world) / clock.fit);
    facts.bounds = bounds;
    if facts.steps.len() == rig.clips.len() {
        return;
    }
    facts.steps = (rig.clips.iter())
        .map(|played| played.motion.stride_mps * played.duration() * 0.5)
        .collect();
    facts.drifts = rig.clips.iter().map(|played| played.motion.drift_m).collect();
}

fn followed(
    (files, assets): (&mut FigureFiles, &Assets),
    held: &mut BTreeMap<Entity, Clocks>,
    (entity, figure, changed): (Entity, &Figure, bool),
    limbs: Option<&Ref<FigureLimbs>>,
) {
    let limbed = limbs.is_some();
    let relimbed = limbs.is_some_and(|limbs| limbs.is_changed());
    let limbs = limbs.map(|limbs| &**limbs);
    match held.get_mut(&entity) {
        Some(clock) if !changed && !relimbed && clock.limbed == limbed => {}
        Some(clock) if !changed => {
            clock.rig = files.rig((figure, limbs), assets).map(Arc::new);
            clock.limbed = limbed;
            (clock.fitted, clock.fit_rest) = (false, None);
        }
        Some(_) | None => {
            let rig = files.rig((figure, limbs), assets).map(Arc::new);
            held.insert(entity, Clocks::new(limbed, rig));
        }
    }
}

fn holding(
    held: &BTreeMap<Entity, Clocks>,
    holders: &Query<(Entity, &Held, Option<&Parent>)>,
    (refusals, spawner): (&mut Refusals, &mut Spawner<SkinnedBrick>),
) {
    refusals.keep_part(HELD_PART, |entity| holders.contains(entity));
    for (entity, item, parent) in holders {
        let Some(clock) = held.get(&item.figure) else {
            refusals.refuse(
                entity,
                HELD_PART,
                Error::new(ErrorCode::UnknownReference, "$.held.figure")
                    .got(item.figure.to_string()),
            );
            continue;
        };
        let Ok(rig) = clock.rig.as_deref() else {
            continue;
        };
        if item.joint.0 >= rig.skeleton.len() {
            refusals.refuse(
                entity,
                HELD_PART,
                Error::new(ErrorCode::UnknownReference, "$.held.joint").got(item.joint.0),
            );
            continue;
        }
        refusals.accept(entity, HELD_PART);
        if parent.map(Parent::parent) != Some(item.figure) {
            spawner.set_parent(entity, Some(item.figure));
        }
    }
}

struct Stepped {
    held: Arc<BTreeMap<Entity, Clocks>>,
    scratch: Scratch,
}

impl StepPoser for Stepped {
    fn pose(
        &mut self,
        figure: Entity,
        play: Option<&FigurePlay>,
        asked: &[Option<JointNo>],
        out: &mut StepPose,
    ) -> bool {
        let Some(clock) = self.held.get(&figure) else {
            return false;
        };
        let (Ok(rig), Some(world)) = (clock.rig.as_deref(), clock.world) else {
            return false;
        };
        if play.is_some_and(|play| clips_in(rig, play).is_err())
            || posed((rig, clock, play), Some(world), 0.0, &mut self.scratch).is_err()
        {
            return false;
        }
        let pose = &self.scratch.base;
        out.joints.clear();
        out.joints
            .extend(asked.iter().map(|joint| joint_at(world, joint.and_then(|joint| pose.get(joint.0)))));
        out.top = clock.top.and_then(|top| head_top(rig, pose, world, top));
        out.hold_shift = self.scratch.shift;
        true
    }
}

type TickReads<'w, 's> = (
    Res<'w, FixedTime>,
    Res<'w, GlobalTransforms>,
    Res<'w, Assets>,
    Query<'w, 's, &'static Transform>,
);

type TickState<'w> = (
    ResMut<'w, FigureFiles>,
    ResMut<'w, FigureClocks>,
    ResMut<'w, StepJoints>,
    Stats<'w, SkinnedStats>,
);

type TickFigures<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Ref<'static, Figure>,
        Option<&'static FigurePlay>,
        Option<&'static mut FigureClock>,
        Option<&'static Drawn>,
        (
            Option<Ref<'static, FigureLimbs>>,
            Option<Ref<'static, FigureFit>>,
        ),
    ),
>;

type Holders<'w, 's> = (
    Query<'w, 's, (Entity, &'static Held, Option<&'static Parent>)>,
    Query<'w, 's, &'static Parent, With<Figure>>,
);

type Report<'w, 's> = (ResMut<'w, Refusals>, Spawner<'w, 's, SkinnedBrick>);

pub(super) fn tick(
    (fixed, globals, assets, transforms): TickReads,
    (mut files, mut clocks, mut steps, mut stats): TickState,
    mut figures: TickFigures,
    (holders, parents): Holders,
    (mut refusals, mut spawner): Report,
) {
    steps.renew(None);
    if assets.is_changed() {
        files.keep(&assets);
    }
    let held = Arc::make_mut(&mut clocks.held);
    held.retain(|entity, _| figures.contains(*entity));
    let delta = fixed.delta_seconds;
    let mut ticked = 0;
    for (entity, figure, play, facts, drawn, (limbs, fit)) in &mut figures {
        let changed = figure.is_changed();
        followed((&mut files, &assets), held, (entity, &figure, changed), limbs.as_ref());
        let Some(clock) = held.get_mut(&entity) else {
            continue;
        };
        let rig = match &clock.rig {
            Ok(rig) => Arc::clone(rig),
            Err(error) => {
                refusals.refuse(entity, FIGURES_PART, error.clone());
                continue;
            }
        };
        if let Err(error) = clock.tick((&rig, play), &mut files, delta) {
            refusals.refuse(entity, FIGURES_PART, error);
            continue;
        }
        refusals.accept(entity, FIGURES_PART);
        ticked += 1;
        let parent = parents
            .get(entity)
            .ok()
            .and_then(|parent| globals.get(parent.parent()))
            .and_then(|global| global.matrix());
        let world = world_of(transforms.get(entity).ok(), parent);
        if fit.as_ref().is_some_and(|fit| fit.is_changed()) || clock.fitted != fit.is_some() {
            clock.fit_rest = fit.as_deref().and_then(|fit| files.fit_rest(&rig, fit));
            clock.fitted = fit.is_some();
        }
        clock.fit = fitted(clock.fit_rest, fit.as_ref().map(|fit| fit.height_m), world);
        let world = world * Mat4::from_scale(Vec3::splat(clock.fit));
        clock.world = Some(world);
        clock.top = (drawn.and_then(|drawn| drawn.geometry))
            .and_then(|mesh| files.top(&rig, (&assets, mesh)));
        let bounds = bounded(&rig, clock, play.and_then(|play| play.base.stride), world);
        match facts {
            Some(mut facts) => clocked(&mut facts, (&rig, clock), bounds),
            None => {
                let mut made = FigureClock::default();
                clocked(&mut made, (&rig, clock), bounds);
                spawner.insert(entity, made);
            }
        }
    }
    refusals.keep_part(FIGURES_PART, |entity| held.contains_key(&entity));
    holding(held, &holders, (&mut refusals, &mut spawner));
    steps.renew(Some(Box::new(Stepped {
        held: Arc::clone(&clocks.held),
        scratch: Scratch::default(),
    })));
    stats.edit(|stats| stats.ticked = ticked);
}

type PoseReads<'w> = (
    Res<'w, FrameTime>,
    Res<'w, GlobalTransforms>,
    Res<'w, FigureClocks>,
    (Res<'w, FigureSight>, Res<'w, Assets>),
);

type PoseFigures<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Option<&'static FigurePlay>,
        Option<&'static mut FigureJoints>,
    ),
    (With<Figure>, With<Drawn>),
>;

type Carrying<'w, 's> = (
    Query<'w, 's, (Entity, &'static Held)>,
    Moves<'w, 's, SkinnedBrick>,
);

fn paletted(
    (rig, fit): (&Rig, f32),
    scratch: &mut Scratch,
    (palette, entity): (&mut FigurePalette, Entity),
) -> Result<(usize, usize), Error> {
    let Scratch { base, model, .. } = scratch;
    model.clear();
    model.resize(base.len(), Mat4::IDENTITY);
    engine_rig::model_matrices(base, model).map_err(|error| rigged(&error))?;
    let fit = Mat4::from_scale(Vec3::splat(fit));
    for matrix in model.iter_mut() {
        *matrix = fit * *matrix;
    }
    let at = palette.slot(entity, base.len());
    engine_rig::palette(&rig.skeleton, model, &mut palette.rows[at..at + base.len()])
        .map_err(|error| rigged(&error))?;
    Ok((at, base.len()))
}

fn published(
    joints: &mut FigureJoints,
    (rig, pose, world): (&Rig, &[Posed], Mat4),
    (top, shift): (Option<f32>, Vec3),
) {
    let FigureJoints {
        asked,
        joints: at,
        top: topped,
        hold_shift,
    } = joints;
    at.clear();
    at.extend(asked.iter().map(|joint| joint_at(world, joint.and_then(|joint| pose.get(joint.0)))));
    *topped = top.and_then(|top| head_top(rig, pose, world, top));
    *hold_shift = shift;
}

fn carried(
    figure: Entity,
    (fit, pose): (f32, &[Posed]),
    (holders, moves): (&Query<(Entity, &Held)>, &mut Moves<SkinnedBrick>),
) {
    for (entity, held) in holders {
        if held.figure != figure {
            continue;
        }
        let Some(joint) = pose.get(held.joint.0) else {
            continue;
        };
        if let Some(mut transform) = moves.get_mut(entity) {
            transform.set_if_neq(held_local(joint, held, fit));
        }
    }
}

pub(super) fn pose(
    (time, globals, clocks, (sight, assets)): PoseReads,
    skipped: Query<(), With<ViewSkip>>,
    mut figures: PoseFigures,
    (holders, mut moves): Carrying,
    (mut palette, mut stats): (ResMut<FigurePalette>, Stats<SkinnedStats>),
    (mut scratch, mut refusals): (Local<Scratch>, ResMut<Refusals>),
) {
    palette.written.clear();
    refusals.keep_part(DRAW_PART, |entity| figures.contains(entity));
    if !skipped.is_empty() {
        stats.edit(|stats| stats.unposed = 0);
        return;
    }
    stats.edit(|stats| stats.heads.clear());
    let sighted = Some(&*sight).filter(|sight| sight.frame == time.frame);
    let mut listed: Vec<Entity> = figures
        .iter()
        .map(|(entity, ..)| entity)
        .filter(|entity| match sighted {
            Some(sight) if sight.judged.binary_search(entity).is_ok() => {
                sight.shown.binary_search(entity).is_ok()
            }
            Some(_) | None => globals.drawn(*entity),
        })
        .collect();
    listed.sort_unstable();
    let listed_count = listed.len();
    let back = 1.0 - time.alpha;
    let mut drawn_now = Vec::with_capacity(listed.len());
    for entity in listed {
        let Some(clock) = clocks.held.get(&entity) else {
            continue;
        };
        let Ok(rig) = clock.rig.as_deref() else {
            continue;
        };
        let Ok((_, play, joints)) = figures.get_mut(entity) else {
            continue;
        };
        if play.is_some_and(|play| clips_in(rig, play).is_err()) {
            continue;
        }
        let world = globals.get(entity).and_then(|global| global.matrix());
        let made = posed((rig, clock, play), world, back, &mut scratch)
            .and_then(|()| paletted((rig, clock.fit), &mut scratch, (&mut palette, entity)));
        let slot = match made {
            Ok(slot) => slot,
            Err(error) => {
                refusals.refuse(entity, DRAW_PART, error);
                continue;
            }
        };
        drawn_now.push(entity);
        palette.written.push(slot);
        refusals.accept(entity, DRAW_PART);
        let fitted = world.map(|world| world * Mat4::from_scale(Vec3::splat(clock.fit)));
        if let (Some(mut joints), Some(world)) = (joints, fitted) {
            let found = (clock.top, scratch.shift);
            published(&mut joints, (rig, &scratch.base, world), found);
        }
        carried(entity, (clock.fit, &scratch.base), (&holders, &mut moves));
        stats.edit(|stats| {
            stats.heads.push(FigureHead {
                entity: entity.to_string(),
                time: clock
                    .base
                    .as_ref()
                    .map_or(0.0, |base| base.at(rig.duration(base.clip), back)),
                head: rig
                    .limbs
                    .head
                    .map(|joint| scratch.base[joint].position.to_array()),
            });
        });
    }
    palette.keep(&drawn_now);
    let joints: usize = palette.written.iter().map(|(_, length)| length).sum();
    let skeletons = assets.each::<Arc<Skeleton>>().count() as u32;
    let clips = assets.each::<Arc<Clip>>().count() as u32;
    stats.edit(|stats| {
        stats.sampled = drawn_now.len() as u32;
        stats.unposed = (listed_count - drawn_now.len()) as u32;
        stats.joints = joints as u32;
        stats.palette_bytes = joints as u64 * size_of::<PaletteRow>() as u64;
        stats.skeletons = skeletons;
        stats.clips = clips;
    });
}

type Refusing<'w, 's> = (
    Res<'w, Refusals>,
    Query<'w, 's, Entity, Or<(With<Figure>, With<Held>)>>,
);

pub(super) fn count((refusals, owned): Refusing, mut stats: Stats<SkinnedStats>) {
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
    stats.edit(|stats| stats.refused = refused);
}
