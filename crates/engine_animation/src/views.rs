use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::{
    Asked, Asks, AssetResidency, DrawnPoses, Error, ErrorCode, Expected, GlobalTransforms, Lands,
    Operation, SceneIds,
};
use engine_places_rendering::animation::{
    Animate, Animator, AnimatorCommand, Rig, Track, full_speed, looping,
};
use glam::Quat;
use serde::{Deserialize, Serialize};

use crate::skins::Skins;

fn unknown_entity(id: &str) -> Error {
    Error::new(ErrorCode::UnknownEntity, "$.entity")
        .expected(Expected::Entity)
        .got(id)
}

fn unknown_joint(rig: &Rig, path: &str, joint: &str) -> Error {
    Error::new(ErrorCode::InvalidValue, path)
        .variants(rig.joints.iter().map(|joint| joint.name.clone()))
        .got(joint)
}

type Ordered = (
    SceneIds<'static, 'static>,
    Query<'static, 'static, &'static Animator>,
    Res<'static, Asked<Animate>>,
    Asks<'static, 'static, Animate>,
);

fn animator_of<'a>(
    ids: &SceneIds<'_, '_>,
    animators: &'a Query<'_, '_, &'static Animator>,
    id: &str,
) -> Result<(Entity, &'a Animator), Error> {
    ids.entity(id)
        .and_then(|entity| animators.get(entity).ok().map(|held| (entity, held)))
        .ok_or_else(|| unknown_entity(id))
}

fn layers(animator: &Animator, asked: &Asked<Animate>, entity: Entity) -> usize {
    let pending = asked
        .pending()
        .filter(|ask| {
            ask.animator == entity && matches!(ask.command, AnimatorCommand::Layer { .. })
        })
        .count();
    animator.layers().len() + pending
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct AnimationAsked {
    pub temporary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnimationPlay {
    pub entity: String,
    #[serde(default)]
    pub clip: Option<String>,
    #[serde(default)]
    pub layer: usize,
    #[serde(default)]
    pub seconds: f32,
    #[serde(default = "looping")]
    pub looped: bool,
    #[serde(default = "full_speed")]
    pub speed: f32,
}

impl Operation for AnimationPlay {
    const NAME: &'static str = "animation_play";
    const LANDS: Lands = Lands::NextStep;
    type Params = Ordered;
    type Request = Self;
    type Reply = AnimationAsked;

    fn run(
        (ids, animators, asked, mut asks): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<AnimationAsked, Error> {
        let (entity, animator) = animator_of(&ids, &animators, &request.entity)?;
        let layers = layers(animator, &asked, entity);
        if request.layer >= layers {
            return Err(Error::new(ErrorCode::OutOfRange, "$.layer")
                .at_most(layers as u64 - 1)
                .got(request.layer));
        }
        let command = match &request.clip {
            Some(clip) if animator.rig().clip(clip).is_none() => {
                let names = animator.rig().clips.iter().map(|clip| clip.name.clone());
                return Err(Error::new(ErrorCode::InvalidValue, "$.clip")
                    .variants(names)
                    .got(clip));
            }
            Some(clip) => AnimatorCommand::Blend {
                layer: request.layer,
                clip: clip.clone(),
                seconds: request.seconds,
                looped: request.looped,
                speed: request.speed,
            },
            None => AnimatorCommand::Stop {
                layer: request.layer,
                seconds: request.seconds,
            },
        };
        asks.send(Animate {
            animator: entity,
            command,
        });
        Ok(AnimationAsked {
            temporary: true,
            layer: Some(request.layer),
        })
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnimationLayer {
    pub entity: String,
    pub root: String,
}

impl Operation for AnimationLayer {
    const NAME: &'static str = "animation_layer";
    const LANDS: Lands = Lands::NextStep;
    type Params = Ordered;
    type Request = Self;
    type Reply = AnimationAsked;

    fn run(
        (ids, animators, asked, mut asks): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<AnimationAsked, Error> {
        let (entity, animator) = animator_of(&ids, &animators, &request.entity)?;
        if animator.rig().joint(&request.root).is_none() {
            return Err(unknown_joint(animator.rig(), "$.root", &request.root));
        }
        let layer = layers(animator, &asked, entity);
        asks.send(Animate {
            animator: entity,
            command: AnimatorCommand::Layer { root: request.root },
        });
        Ok(AnimationAsked {
            temporary: true,
            layer: Some(layer),
        })
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnimationAim {
    pub entity: String,
    pub joint: String,
    #[serde(default)]
    pub rotation: Option<[f32; 4]>,
}

impl Operation for AnimationAim {
    const NAME: &'static str = "animation_aim";
    const LANDS: Lands = Lands::NextStep;
    type Params = Ordered;
    type Request = Self;
    type Reply = AnimationAsked;

    fn run(
        (ids, animators, _, mut asks): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<AnimationAsked, Error> {
        let (entity, animator) = animator_of(&ids, &animators, &request.entity)?;
        if animator.rig().joint(&request.joint).is_none() {
            return Err(unknown_joint(animator.rig(), "$.joint", &request.joint));
        }
        let rotation = match request.rotation {
            Some(held) => {
                let rotation = Quat::from_array(held);
                if !rotation.is_finite() || rotation.length_squared() == 0.0 {
                    return Err(Error::new(ErrorCode::InvalidValue, "$.rotation")
                        .expected(Expected::Nonzero)
                        .got(request.rotation));
                }
                Some(rotation.normalize())
            }
            None => None,
        };
        asks.send(Animate {
            animator: entity,
            command: AnimatorCommand::Aim {
                joint: request.joint,
                rotation,
            },
        });
        Ok(AnimationAsked {
            temporary: true,
            layer: None,
        })
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct PlayingView {
    pub clip: Option<String>,
    pub time: f32,
    pub weight: f32,
    pub looped: bool,
    pub speed: f32,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct LayerView {
    pub root: Option<String>,
    pub playing: Vec<PlayingView>,
    pub finished: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct ClipView {
    pub name: String,
    pub duration: f32,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct AnimationStateView {
    pub layers: Vec<LayerView>,
    pub clips: Vec<ClipView>,
    pub joints: BTreeMap<String, [f32; 3]>,
    pub drawn: BTreeMap<String, [f32; 3]>,
    pub parts: usize,
    pub resident: usize,
    pub posed: usize,
}

fn playing_view(rig: &Rig, track: Track, weight: f32) -> PlayingView {
    match track {
        Track::Clip {
            clip,
            time,
            looped,
            speed,
        } => PlayingView {
            clip: rig.clips.get(clip as usize).map(|clip| clip.name.clone()),
            time,
            weight,
            looped,
            speed,
        },
        Track::Rest => PlayingView {
            clip: None,
            time: 0.0,
            weight,
            looped: false,
            speed: 0.0,
        },
    }
}

fn state_of(
    animator: &Animator,
    (globals, poses): (&GlobalTransforms, &DrawnPoses),
    (residency, skins): (Option<&AssetResidency>, &Skins),
) -> AnimationStateView {
    let rig = animator.rig();
    let layers = animator
        .layers()
        .iter()
        .enumerate()
        .map(|(index, layer)| LayerView {
            root: layer
                .root
                .and_then(|joint| rig.joints.get(joint as usize))
                .map(|joint| joint.name.clone()),
            playing: layer
                .playing
                .iter()
                .map(|playing| playing_view(rig, playing.track, playing.fade.weight()))
                .collect(),
            finished: animator.finished(index),
        })
        .collect();
    let clips = rig
        .clips
        .iter()
        .map(|clip| ClipView {
            name: clip.name.clone(),
            duration: clip.duration,
        })
        .collect();
    let named = || rig.joints.iter().zip(animator.joint_nodes());
    let joints = named()
        .filter_map(|(joint, &node)| {
            let position = globals.get(node)?.translation()?;
            Some((joint.name.clone(), position.to_array()))
        })
        .collect();
    let drawn = named()
        .filter_map(|(joint, &node)| {
            let position = poses.drawn_pose(globals, node)?.w_axis.truncate();
            Some((joint.name.clone(), position.to_array()))
        })
        .collect();
    let resident = residency.map_or(0, |residency| {
        rig.parts
            .iter()
            .filter(|part| {
                residency.resident(part.mesh) && residency.material_resident(part.material)
            })
            .count()
    });
    AnimationStateView {
        layers,
        clips,
        joints,
        drawn,
        parts: rig.parts.len(),
        resident,
        posed: animator
            .part_nodes()
            .iter()
            .filter(|&&part| skins.posed(part))
            .count(),
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnimationState {
    pub entity: String,
}

impl Operation for AnimationState {
    const NAME: &'static str = "animation_state";
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, &'static Animator>,
        Res<'static, GlobalTransforms>,
        Res<'static, DrawnPoses>,
        (Option<Res<'static, AssetResidency>>, Res<'static, Skins>),
    );
    type Request = Self;
    type Reply = AnimationStateView;

    fn run(
        (ids, animators, globals, poses, (residency, skins)): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<AnimationStateView, Error> {
        let (_, animator) = animator_of(&ids, &animators, &request.entity)?;
        Ok(state_of(
            animator,
            (&globals, &poses),
            (residency.as_deref(), &skins),
        ))
    }
}
