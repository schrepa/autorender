use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::{
    AssetResidency, Error, ErrorCode, Expected, GlobalTransforms, Operation, SceneIds,
};
use engine_places::animation::{Animator, Rig, full_speed, looping};
use glam::{Mat4, Quat};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn schema_of<T: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null)
}

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
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, &'static mut Animator>,
    );
    type Request = Self;

    fn run(
        (ids, mut animators): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let mut animator = ids
            .entity(&request.entity)
            .and_then(|entity| animators.get_mut(entity).ok())
            .ok_or_else(|| unknown_entity(&request.entity))?;
        if request.layer >= animator.layers.len() {
            return Err(Error::new(ErrorCode::OutOfRange, "$.layer")
                .at_most(animator.layers.len() as u64 - 1)
                .got(request.layer));
        }
        match &request.clip {
            Some(clip) => {
                let started = animator.crossfade_on(
                    request.layer,
                    clip,
                    request.seconds,
                    request.looped,
                    request.speed,
                );
                if !started {
                    let names = animator.rig().clips.iter().map(|clip| clip.name.clone());
                    return Err(Error::new(ErrorCode::InvalidValue, "$.clip")
                        .variants(names)
                        .got(clip));
                }
            }
            None => {
                animator.stop_on(request.layer, request.seconds);
            }
        }
        Ok(json!({ "temporary": true, "layer": request.layer }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
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
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, &'static mut Animator>,
    );
    type Request = Self;

    fn run(
        (ids, mut animators): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let mut animator = ids
            .entity(&request.entity)
            .and_then(|entity| animators.get_mut(entity).ok())
            .ok_or_else(|| unknown_entity(&request.entity))?;
        let Some(layer) = animator.add_layer(&request.root) else {
            return Err(unknown_joint(animator.rig(), "$.root", &request.root));
        };
        Ok(json!({ "temporary": true, "layer": layer }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
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
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, &'static mut Animator>,
    );
    type Request = Self;

    fn run(
        (ids, mut animators): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let mut animator = ids
            .entity(&request.entity)
            .and_then(|entity| animators.get_mut(entity).ok())
            .ok_or_else(|| unknown_entity(&request.entity))?;
        if animator.rig().joint(&request.joint).is_none() {
            return Err(unknown_joint(animator.rig(), "$.joint", &request.joint));
        }
        match request.rotation {
            Some(rotation) => {
                let rotation = Quat::from_array(rotation);
                if !rotation.is_finite() || rotation.length_squared() == 0.0 {
                    return Err(Error::new(ErrorCode::InvalidValue, "$.rotation")
                        .expected(Expected::Nonzero)
                        .got(request.rotation));
                }
                animator.set_rotation(&request.joint, rotation.normalize());
            }
            None => animator.clear_rotation(&request.joint),
        }
        Ok(json!({ "temporary": true }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
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
    pub parts: usize,
    pub resident: usize,
}

fn state_of(
    animator: &Animator,
    root: Mat4,
    residency: Option<&AssetResidency>,
) -> AnimationStateView {
    let rig = animator.rig();
    let layers = animator
        .layers
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
                .map(|playing| PlayingView {
                    clip: playing
                        .clip
                        .and_then(|clip| rig.clips.get(clip as usize))
                        .map(|clip| clip.name.clone()),
                    time: playing.time,
                    weight: playing.weight(),
                    looped: playing.looped,
                    speed: playing.speed,
                })
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
    let joints = rig
        .joints
        .iter()
        .zip(&animator.model)
        .map(|(joint, model)| {
            let position = (root * *model).w_axis.truncate();
            (joint.name.clone(), position.to_array())
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
        parts: rig.parts.len(),
        resident,
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
        Option<Res<'static, AssetResidency>>,
    );
    type Request = Self;

    fn run(
        (ids, animators, globals, residency): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let (entity, animator) = ids
            .entity(&request.entity)
            .and_then(|entity| animators.get(entity).ok().map(|held| (entity, held)))
            .ok_or_else(|| unknown_entity(&request.entity))?;
        let root = globals.pose(entity).map_or(Mat4::IDENTITY, Mat4::from);
        Ok(json!(state_of(animator, root, residency.as_deref())))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }

    fn result_schema() -> Option<Value> {
        Some(schema_of::<AnimationStateView>())
    }
}
