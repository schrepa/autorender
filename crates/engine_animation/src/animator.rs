use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::clamp;
use engine_core::{
    AssetLibrary, AssetUses, FixedTime, GlobalTransforms, Parent, Spawner, Transform,
};
use engine_places::animation::{Animator, Attachment, Clip, ClipLayer, Joint, Keys, Rig};
use glam::{Mat4, Quat, Vec3};

use crate::AnimationBrick;
use crate::part::Books;

pub(crate) struct Ticked {
    rig: Arc<Rig>,
    locals: Vec<Transform>,
    previous: Vec<Transform>,
}

impl Ticked {
    fn new(rig: &Arc<Rig>) -> Self {
        let held: Vec<Transform> = rig.joints.iter().map(rest).collect();
        Self {
            rig: Arc::clone(rig),
            locals: held.clone(),
            previous: held,
        }
    }
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

fn compose(joints: &[Joint], locals: impl Iterator<Item = Transform>, model: &mut Vec<Mat4>) {
    model.clear();
    for (joint, local) in joints.iter().zip(locals) {
        let parent = joint
            .parent
            .and_then(|parent| model.get(parent as usize))
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        model.push(parent * local.matrix());
    }
}

fn wrapped(time: f32, duration: f32, looped: bool) -> f32 {
    if duration <= 0.0 {
        0.0
    } else if looped {
        time.rem_euclid(duration)
    } else {
        clamp(time, 0.0, duration)
    }
}

fn settle(layer: &mut ClipLayer) {
    let Some(full) = layer
        .playing
        .iter()
        .rposition(|playing| playing.weight() >= 1.0)
    else {
        return;
    };
    layer.playing.drain(..full);
    if layer
        .playing
        .first()
        .is_some_and(|playing| playing.clip.is_none())
    {
        layer.playing.remove(0);
    }
}

fn evaluate(animator: &mut Animator, ticked: &mut Ticked) {
    std::mem::swap(&mut ticked.previous, &mut ticked.locals);
    let rig = &*ticked.rig;
    ticked.locals.clear();
    ticked.locals.extend(rig.joints.iter().map(rest));
    let mut target = Vec::new();
    for layer in &animator.layers {
        let below = ticked.locals.clone();
        for playing in &layer.playing {
            target.clone_from(&below);
            if let Some(clip) = playing.clip.and_then(|clip| rig.clips.get(clip as usize)) {
                sample(clip, playing.time, &mut target);
            }
            let weight = playing.weight();
            for (joint, (local, wanted)) in ticked.locals.iter_mut().zip(&target).enumerate() {
                if layer.covers(joint) {
                    *local = mix(local, wanted, weight);
                }
            }
        }
    }
    for &(joint, rotation) in &animator.overrides {
        if let Some(local) = ticked.locals.get_mut(joint as usize) {
            local.rotation = rotation;
        }
    }
    compose(
        &rig.joints,
        ticked.locals.iter().copied(),
        &mut animator.model,
    );
}

fn step(animator: &mut Animator, ticked: &mut Ticked, seconds: f32, root: Mat4) {
    animator.root = root;
    let rig = &*ticked.rig;
    for layer in &mut animator.layers {
        for playing in &mut layer.playing {
            playing.fade += seconds;
            let duration = playing
                .clip
                .and_then(|clip| rig.clips.get(clip as usize))
                .map_or(0.0, |clip| clip.duration);
            let time = playing.time + playing.speed * seconds;
            playing.time = wrapped(time, duration, playing.looped);
        }
        settle(layer);
    }
    evaluate(animator, ticked);
}

pub(crate) fn presented_pose(
    animator: &Animator,
    ticked: Option<&Ticked>,
    alpha: f32,
    model: &mut Vec<Mat4>,
) {
    let current = ticked.filter(|ticked| Arc::ptr_eq(&ticked.rig, animator.rig()));
    let Some(ticked) = current else {
        model.clone_from(&animator.model);
        return;
    };
    let locals = ticked
        .previous
        .iter()
        .zip(&ticked.locals)
        .map(|(from, to)| mix(from, to, alpha));
    compose(&ticked.rig.joints, locals, model);
}

pub(crate) fn advance(
    time: Res<FixedTime>,
    globals: Res<GlobalTransforms>,
    mut books: ResMut<Books>,
    mut animators: Query<(Entity, &mut Animator)>,
) {
    books.ticked.retain(|entity, _| animators.contains(*entity));
    for (entity, mut animator) in &mut animators {
        let root = globals.pose(entity).map_or(Mat4::IDENTITY, Mat4::from);
        let ticked = books
            .ticked
            .entry(entity)
            .or_insert_with(|| Ticked::new(animator.rig()));
        if !Arc::ptr_eq(&ticked.rig, animator.rig()) {
            *ticked = Ticked::new(animator.rig());
        }
        step(&mut animator, ticked, time.delta_seconds, root);
    }
}

pub(crate) fn follow_joints(
    animators: Query<&Animator>,
    mut attached: Query<(&Attachment, &Parent, &mut Transform)>,
) {
    for (attachment, parent, mut transform) in &mut attached {
        let Ok(animator) = animators.get(parent.parent()) else {
            continue;
        };
        let Some(joint) = animator.joint_model(&attachment.joint) else {
            continue;
        };
        let (scale, rotation, translation) =
            (joint * attachment.offset()).to_scale_rotation_translation();
        let followed = Transform {
            translation,
            rotation,
            scale,
        };
        if *transform != followed {
            *transform = followed;
        }
    }
}

#[derive(Component)]
pub(crate) struct SkinAssets;

pub(crate) fn publish_assets(
    animators: Query<&Animator>,
    assets: Res<AssetLibrary>,
    own: Query<(Entity, &AssetUses), With<SkinAssets>>,
    mut spawner: Spawner<AnimationBrick>,
) {
    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    for animator in &animators {
        for part in &animator.rig().parts {
            meshes.push(part.mesh);
            materials.push(part.material);
        }
    }
    meshes.sort_unstable();
    meshes.dedup();
    materials.sort_unstable();
    materials.dedup();
    let mut textures: Vec<_> = materials
        .iter()
        .filter_map(|&material| assets.material(material))
        .flat_map(|material| material.textures())
        .collect();
    textures.sort_unstable();
    textures.dedup();
    let used = AssetUses {
        meshes,
        materials,
        textures,
    };
    match own.single() {
        Ok((entity, held)) => {
            if *held != used {
                spawner.insert(entity, used);
            }
        }
        Err(_) => {
            spawner.spawn((SkinAssets, used));
        }
    }
}

pub(crate) fn forget(
    own: Query<Entity, With<SkinAssets>>,
    mut books: ResMut<Books>,
    mut spawner: Spawner<AnimationBrick>,
) {
    for entity in &own {
        spawner.despawn(entity);
    }
    books.ticked.clear();
    books.epoch += 1;
}
