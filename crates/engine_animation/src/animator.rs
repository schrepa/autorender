use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{
    Applies, Clock, Drawn, EntityIds, Event, Feed, FeedSettings, FixedTime, FrameTime,
    GlobalTransforms, Mover, Moves, Pace, Parent, Pause, Raises, Spawner, Transform, mover_of,
};
use engine_places::meshes::SKIN;
use engine_places_rendering::animation::{
    Animate, Animator, Attachment, Clip, Joint, Keys, Rig, Track,
};
use glam::{Quat, Vec3};

use crate::AnimationBrick;

#[derive(Component, Clone, Copy)]
#[component(immutable)]
pub(crate) struct JointNode {
    animator: Entity,
}

struct Skeleton {
    rig: Arc<Rig>,
    joints: Vec<Entity>,
    parts: Vec<Entity>,
    pace: Pace,
}

#[derive(Default)]
struct Scratch {
    locals: Vec<Transform>,
    below: Vec<Transform>,
    target: Vec<Transform>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnimationEnded {
    pub animator: Entity,
    pub layer: usize,
    pub clip: Option<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, schemars::JsonSchema)]
pub struct AnimationEndedRow {
    pub entity: Option<String>,
    pub layer: usize,
    pub clip: Option<String>,
}

impl Event for AnimationEnded {
    const NAME: &'static str = "animation_ended";
    type Row = AnimationEndedRow;

    fn row(&self, ids: &dyn EntityIds) -> AnimationEndedRow {
        AnimationEndedRow {
            entity: ids.id(self.animator),
            layer: self.layer,
            clip: self.clip.clone(),
        }
    }
}

#[derive(Resource)]
pub struct AnimationEnds {
    raised: Feed<AnimationEnded>,
}

impl Default for AnimationEnds {
    fn default() -> Self {
        Self {
            raised: Feed::new(FeedSettings::default().events as usize),
        }
    }
}

impl Raises<AnimationEnded> for AnimationEnds {
    fn raised(&self) -> &Feed<AnimationEnded> {
        &self.raised
    }
}

fn finished_layers(animator: &Animator) -> Vec<bool> {
    (0..animator.layers().len())
        .map(|layer| animator.finished(layer))
        .collect()
}

fn playing_clip(animator: &Animator, layer: usize) -> Option<String> {
    let playing = animator.layers().get(layer)?.playing.last()?;
    match playing.track {
        Track::Clip { clip, .. } => animator
            .rig()
            .clips
            .get(clip as usize)
            .map(|clip| clip.name.clone()),
        Track::Rest => None,
    }
}

fn step_ending(
    entity: Entity,
    animator: &mut Animator,
    seconds: f32,
    (ends, feeds): (&mut AnimationEnds, &FeedSettings),
) {
    let before = finished_layers(animator);
    animator.step(seconds);
    ends.raised.resize(feeds.events as usize);
    for (layer, was) in before.into_iter().enumerate() {
        if !was && animator.finished(layer) {
            ends.raised.push(AnimationEnded {
                animator: entity,
                layer,
                clip: playing_clip(animator, layer),
            });
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct Books {
    skeletons: HashMap<Entity, Skeleton>,
    scratch: Scratch,
}

fn rest(joint: &Joint) -> Transform {
    Transform {
        translation: Vec3::from_array(joint.translation),
        rotation: Quat::from_array(joint.rotation),
        scale: Vec3::from_array(joint.scale),
    }
}

fn around<T: Copy>(keys: &Keys<T>, time: f32) -> Option<(T, T, f32)> {
    let after = keys.times.partition_point(|&at| at <= time);
    let Some(last) = after.checked_sub(1) else {
        let first = *keys.values.first()?;
        return Some((first, first, 0.0));
    };
    let from = *keys.values.get(last)?;
    let (Some(&to), Some(&end), false) = (keys.values.get(after), keys.times.get(after), keys.step)
    else {
        return Some((from, from, 0.0));
    };
    let start = *keys.times.get(last)?;
    Some((from, to, (time - start) / (end - start)))
}

fn sample(clip: &Clip, time: f32, pose: &mut [Transform]) {
    for track in &clip.tracks {
        let Some(held) = pose.get_mut(track.joint as usize) else {
            continue;
        };
        if let Some((from, to, mix)) = track.translation.as_ref().and_then(|k| around(k, time)) {
            held.translation = Vec3::from_array(from).lerp(Vec3::from_array(to), mix);
        }
        if let Some((from, to, mix)) = track.rotation.as_ref().and_then(|k| around(k, time)) {
            held.rotation = Quat::from_array(from).lerp(Quat::from_array(to), mix);
        }
        if let Some((from, to, mix)) = track.scale.as_ref().and_then(|k| around(k, time)) {
            held.scale = Vec3::from_array(from).lerp(Vec3::from_array(to), mix);
        }
    }
}

fn mix(from: &Transform, to: &Transform, by: f32) -> Transform {
    if by >= 1.0 {
        return *to;
    }
    Transform {
        translation: from.translation.lerp(to.translation, by),
        rotation: from.rotation.lerp(to.rotation, by),
        scale: from.scale.lerp(to.scale, by),
    }
}

fn evaluate(animator: &Animator, scratch: &mut Scratch) {
    let rig = &**animator.rig();
    let Scratch {
        locals,
        below,
        target,
    } = scratch;
    locals.clear();
    locals.extend(rig.joints.iter().map(rest));
    for layer in animator.layers() {
        below.clone_from(locals);
        for playing in &layer.playing {
            target.clone_from(below);
            if let Track::Clip { clip, time, .. } = playing.track
                && let Some(clip) = rig.clips.get(clip as usize)
            {
                sample(clip, time, target);
            }
            let weight = playing.fade.weight();
            for (joint, (local, wanted)) in locals.iter_mut().zip(target.iter()).enumerate() {
                if layer.covers(joint) {
                    *local = mix(local, wanted, weight);
                }
            }
        }
    }
    for &(joint, rotation) in animator.overrides() {
        if let Some(local) = locals.get_mut(joint as usize) {
            local.rotation = rotation;
        }
    }
}

fn clock_of(globals: &GlobalTransforms, node: Entity) -> Clock {
    globals.word(node).map_or(Clock::Game, |word| word.clock)
}

fn pace_of(clock: Clock) -> Pace {
    match clock {
        Clock::Game => Pace::Tick,
        Clock::Real => Pace::Frame,
    }
}

const fn mover(pace: Pace) -> Mover {
    Mover {
        by: mover_of::<AnimationBrick>(),
        pace,
    }
}

fn pose(
    animator: &Animator,
    skeleton: Option<&Skeleton>,
    scratch: &mut Scratch,
    moves: &mut Moves<'_, '_, AnimationBrick>,
) {
    let Some(skeleton) = skeleton.filter(|skeleton| Arc::ptr_eq(&skeleton.rig, animator.rig()))
    else {
        return;
    };
    evaluate(animator, scratch);
    for (&joint, local) in skeleton.joints.iter().zip(&scratch.locals) {
        if let Some(mut transform) = moves.get_mut(joint)
            && *transform != *local
        {
            *transform = *local;
        }
    }
}

fn obey(asked: Vec<Animate>, animators: &mut Query<(Entity, &mut Animator)>) {
    for Animate { animator, command } in asked {
        if let Ok((_, mut held)) = animators.get_mut(animator) {
            held.apply(&command);
        }
    }
}

pub(crate) fn command(
    mut applies: Applies<Animate>,
    mut animators: Query<(Entity, &mut Animator)>,
) {
    obey(applies.take(), &mut animators);
}

pub(crate) fn advance(
    (time, feeds): (Res<FixedTime>, Res<FeedSettings>),
    globals: Res<GlobalTransforms>,
    (mut books, mut ends): (ResMut<Books>, ResMut<AnimationEnds>),
    mut animators: Query<(Entity, &mut Animator)>,
    mut moves: Moves<AnimationBrick>,
) {
    let Books { skeletons, scratch } = &mut *books;
    for (entity, mut animator) in &mut animators {
        if !globals.active(entity) || clock_of(&globals, entity) != Clock::Game {
            continue;
        }
        step_ending(entity, &mut *animator, time.delta_seconds, (&mut *ends, &*feeds));
        pose(&animator, skeletons.get(&entity), scratch, &mut moves);
    }
}

pub(crate) fn advance_real(
    (frame, feeds): (Res<FrameTime>, Res<FeedSettings>),
    pause: Res<Pause>,
    globals: Res<GlobalTransforms>,
    mut applies: Applies<Animate>,
    (mut books, mut ends): (ResMut<Books>, ResMut<AnimationEnds>),
    mut animators: Query<(Entity, &mut Animator)>,
    mut moves: Moves<AnimationBrick>,
) {
    if pause.paused() {
        obey(applies.take(), &mut animators);
    }
    let Books { skeletons, scratch } = &mut *books;
    for (entity, mut animator) in &mut animators {
        if !globals.active(entity) || clock_of(&globals, entity) != Clock::Real {
            continue;
        }
        step_ending(entity, &mut *animator, frame.real_seconds as f32, (&mut *ends, &*feeds));
        pose(&animator, skeletons.get(&entity), scratch, &mut moves);
    }
}

fn spawn(
    animator_node: Entity,
    animator: &Animator,
    pace: Pace,
    scratch: &mut Scratch,
    spawner: &mut Spawner<'_, '_, AnimationBrick>,
) -> Skeleton {
    evaluate(animator, scratch);
    let rig = animator.rig();
    let joints: Vec<Entity> = scratch
        .locals
        .iter()
        .map(|&local| {
            spawner.spawn((
                local,
                JointNode {
                    animator: animator_node,
                },
                mover(pace),
            ))
        })
        .collect();
    let parents: Vec<(Entity, Option<Entity>)> = rig
        .joints
        .iter()
        .zip(&joints)
        .map(|(joint, &node)| {
            let parent = joint
                .parent
                .and_then(|parent| joints.get(parent as usize).copied())
                .unwrap_or(animator_node);
            (node, Some(parent))
        })
        .collect();
    let parts: Vec<Entity> = rig
        .parts
        .iter()
        .map(|part| {
            spawner.spawn((
                Transform::IDENTITY,
                Drawn::new(SKIN, part.mesh, part.material),
            ))
        })
        .collect();
    spawner.set_parents(
        parents
            .into_iter()
            .chain(parts.iter().map(|&part| (part, Some(animator_node))))
            .collect(),
    );
    Skeleton {
        rig: Arc::clone(rig),
        joints,
        parts,
        pace,
    }
}

fn hold(
    skeletons: &HashMap<Entity, Skeleton>,
    nodes: &Query<&JointNode>,
    held: &Query<(Entity, &Attachment, &Parent)>,
    spawner: &mut Spawner<'_, '_, AnimationBrick>,
) {
    let mut moved = Vec::new();
    for (item, attachment, parent) in held {
        let holder = parent.parent();
        let animator = nodes.get(holder).map_or(holder, |node| node.animator);
        let Some(skeleton) = skeletons.get(&animator) else {
            continue;
        };
        let wanted = skeleton
            .rig
            .joint(&attachment.joint)
            .and_then(|joint| skeleton.joints.get(joint).copied())
            .unwrap_or(animator);
        if wanted != holder {
            moved.push((item, Some(wanted)));
        }
    }
    if !moved.is_empty() {
        spawner.set_parents(moved);
    }
}

pub(crate) fn sync(
    globals: Res<GlobalTransforms>,
    mut books: ResMut<Books>,
    mut animators: Query<(Entity, &mut Animator)>,
    nodes: Query<&JointNode>,
    held: Query<(Entity, &Attachment, &Parent)>,
    mut spawner: Spawner<AnimationBrick>,
) {
    let Books { skeletons, scratch } = &mut *books;
    let mut gone = Vec::new();
    for (entity, mut animator) in &mut animators {
        let pace = pace_of(clock_of(&globals, entity));
        let kept = skeletons
            .get_mut(&entity)
            .filter(|skeleton| Arc::ptr_eq(&skeleton.rig, animator.rig()));
        match kept {
            Some(skeleton) => {
                if skeleton.pace != pace {
                    skeleton.pace = pace;
                    for &joint in &skeleton.joints {
                        spawner.insert(joint, mover(pace));
                    }
                }
            }
            None => {
                let skeleton = spawn(entity, &animator, pace, scratch, &mut spawner);
                if let Some(old) = skeletons.insert(entity, skeleton) {
                    gone.extend(old.joints);
                    gone.extend(old.parts);
                }
            }
        }
        if let Some(skeleton) = skeletons.get(&entity) {
            if animator.joint_nodes() != skeleton.joints.as_slice() {
                animator.publish_joints(&skeleton.joints);
            }
            if animator.part_nodes() != skeleton.parts.as_slice() {
                animator.publish_parts(&skeleton.parts);
            }
        }
    }
    hold(skeletons, &nodes, &held, &mut spawner);
    for joint in gone {
        spawner.despawn(joint);
    }
}

pub(crate) fn release(
    In(animator): In<Entity>,
    mut books: ResMut<Books>,
    held: Query<(Entity, &Parent), With<Attachment>>,
    nodes: Query<(), With<Transform>>,
    mut spawner: Spawner<AnimationBrick>,
) {
    let Some(skeleton) = books.skeletons.remove(&animator) else {
        return;
    };
    let back = nodes.contains(animator).then_some(animator);
    let dropped: Vec<(Entity, Option<Entity>)> = held
        .iter()
        .filter(|(_, parent)| skeleton.joints.contains(&parent.parent()))
        .map(|(item, _)| (item, back))
        .collect();
    if !dropped.is_empty() {
        spawner.set_parents(dropped);
    }
    for node in skeleton.joints.into_iter().chain(skeleton.parts) {
        spawner.despawn(node);
    }
}
