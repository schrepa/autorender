use std::sync::Arc;

use bevy_ecs::prelude::Component;
use engine_codec::Expected;
use engine_math::clamp;
use engine_place::{Error, ErrorCode, InterpolateTransform, MaterialHandle, MeshHandle};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

const UNIT_SLACK: f32 = 1.0e-2;

fn no_rotation() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn unit_scale() -> [f32; 3] {
    [1.0; 3]
}

pub fn looping() -> bool {
    true
}

pub fn full_speed() -> f32 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Joint {
    pub name: String,
    #[serde(default)]
    pub parent: Option<u32>,
    #[serde(default)]
    pub translation: [f32; 3],
    #[serde(default = "no_rotation")]
    pub rotation: [f32; 4],
    #[serde(default = "unit_scale")]
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    pub joint: u32,
    pub inverse_bind: [[f32; 4]; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Keys<T> {
    pub times: Vec<f32>,
    pub values: Vec<T>,
    #[serde(default)]
    pub step: bool,
}

impl<T> Keys<T> {
    fn check(&self, path: &str, fine: fn(&T) -> bool) -> Result<(), Error> {
        if self.times.is_empty() || self.times.len() != self.values.len() {
            return Err(invalid(format!("{path}.values"))
                .expected(Expected::Matching)
                .got(self.values.len()));
        }
        let sorted = self.times.is_sorted_by(|a, b| a <= b);
        if !sorted || !self.times.iter().all(|time| time.is_finite()) {
            return Err(invalid(format!("{path}.times")).expected(Expected::Sorted));
        }
        if !self.values.iter().all(fine) {
            return Err(invalid(format!("{path}.values")).expected(Expected::Finite));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Track {
    pub joint: u32,
    #[serde(default)]
    pub translation: Option<Keys<[f32; 3]>>,
    #[serde(default)]
    pub rotation: Option<Keys<[f32; 4]>>,
    #[serde(default)]
    pub scale: Option<Keys<[f32; 3]>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Clip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<Track>,
}

impl Clip {
    fn check(&self, path: &str, joints: usize) -> Result<(), Error> {
        if !self.duration.is_finite() || self.duration < 0.0 {
            return Err(invalid(format!("{path}.duration"))
                .expected(Expected::Finite)
                .got(self.duration));
        }
        for (index, track) in self.tracks.iter().enumerate() {
            let path = format!("{path}.tracks[{index}]");
            if track.joint as usize >= joints {
                return Err(invalid(format!("{path}.joint"))
                    .at_most(joints as u64)
                    .got(track.joint));
            }
            if let Some(keys) = &track.translation {
                keys.check(&format!("{path}.translation"), finite3)?;
            }
            if let Some(keys) = &track.rotation {
                keys.check(&format!("{path}.rotation"), unit_rotation)?;
            }
            if let Some(keys) = &track.scale {
                keys.check(&format!("{path}.scale"), finite3)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VertexSkin {
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RigPart {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub skin: Vec<VertexSkin>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Rig {
    pub joints: Vec<Joint>,
    #[serde(default)]
    pub palette: Vec<Bound>,
    #[serde(default)]
    pub clips: Vec<Clip>,
    #[serde(default)]
    pub parts: Vec<RigPart>,
}

impl Rig {
    pub fn joint(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|joint| joint.name == name)
    }

    pub fn clip(&self, name: &str) -> Option<usize> {
        self.clips.iter().position(|clip| clip.name == name)
    }

    pub fn check(&self) -> Result<(), Error> {
        if self.joints.is_empty() {
            return Err(invalid("$.joints").expected(Expected::NonEmpty));
        }
        for (index, joint) in self.joints.iter().enumerate() {
            check_joint(joint, index)?;
        }
        for (index, bound) in self.palette.iter().enumerate() {
            if bound.joint as usize >= self.joints.len() {
                return Err(invalid(format!("$.palette[{index}].joint"))
                    .at_most(self.joints.len() as u64)
                    .got(bound.joint));
            }
            if !bound
                .inverse_bind
                .as_flattened()
                .iter()
                .all(|value| value.is_finite())
            {
                return Err(
                    invalid(format!("$.palette[{index}].inverse_bind")).expected(Expected::Finite)
                );
            }
        }
        for (index, clip) in self.clips.iter().enumerate() {
            clip.check(&format!("$.clips[{index}]"), self.joints.len())?;
        }
        for (index, part) in self.parts.iter().enumerate() {
            check_part(part, index, self.palette.len())?;
        }
        Ok(())
    }

    fn rest_model(&self) -> Vec<Mat4> {
        let mut model: Vec<Mat4> = Vec::with_capacity(self.joints.len());
        for joint in &self.joints {
            let parent = joint
                .parent
                .and_then(|parent| model.get(parent as usize))
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            model.push(parent * offset(joint.translation, joint.rotation, joint.scale));
        }
        model
    }
}

fn offset(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        Vec3::from_array(scale),
        Quat::from_array(rotation),
        Vec3::from_array(translation),
    )
}

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidValue, path)
}

fn finite3(value: &[f32; 3]) -> bool {
    value.iter().all(|part| part.is_finite())
}

fn unit_rotation(value: &[f32; 4]) -> bool {
    let rotation = Quat::from_array(*value);
    rotation.is_finite() && (rotation.length_squared() - 1.0).abs() < UNIT_SLACK
}

fn check_joint(joint: &Joint, index: usize) -> Result<(), Error> {
    let path = format!("$.joints[{index}]");
    if joint.parent.is_some_and(|parent| parent as usize >= index) {
        return Err(invalid(format!("{path}.parent"))
            .expected(Expected::Sorted)
            .got(joint.parent));
    }
    if !finite3(&joint.translation) {
        return Err(invalid(format!("{path}.translation")).expected(Expected::Finite));
    }
    if !finite3(&joint.scale) {
        return Err(invalid(format!("{path}.scale")).expected(Expected::Finite));
    }
    if !unit_rotation(&joint.rotation) {
        return Err(invalid(format!("{path}.rotation")).expected(Expected::Normalized));
    }
    Ok(())
}

fn check_part(part: &RigPart, index: usize, palette: usize) -> Result<(), Error> {
    for (at, skin) in part.skin.iter().enumerate() {
        let path = format!("$.parts[{index}].skin[{at}]");
        if skin.joints.iter().any(|&joint| joint as usize >= palette) {
            return Err(invalid(format!("{path}.joints"))
                .at_most(palette as u64)
                .got(skin.joints));
        }
        let sum: f32 = skin.weights.iter().sum();
        let fine = skin
            .weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0);
        if !fine || sum <= 0.0 {
            return Err(invalid(format!("{path}.weights")).expected(Expected::Nonzero));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Start {
    pub clip: String,
    #[serde(default = "looping")]
    pub looped: bool,
    #[serde(default = "full_speed")]
    pub speed: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnimatorScene {
    pub rig: Rig,
    #[serde(default)]
    pub play: Option<Start>,
}

impl From<AnimatorScene> for Animator {
    fn from(scene: AnimatorScene) -> Self {
        let mut animator = Self::new(Arc::new(scene.rig));
        if let Some(start) = &scene.play {
            animator.play(&start.clip, start.looped, start.speed);
        }
        animator.start = scene.play;
        animator
    }
}

impl From<Animator> for AnimatorScene {
    fn from(animator: Animator) -> Self {
        Self {
            rig: Rig::clone(animator.rig()),
            play: animator.start,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Playing {
    pub clip: Option<u32>,
    pub time: f32,
    pub looped: bool,
    pub speed: f32,
    pub fade: f32,
    pub seconds: f32,
}

impl Playing {
    pub fn weight(&self) -> f32 {
        if self.seconds <= 0.0 {
            1.0
        } else {
            clamp(self.fade / self.seconds, 0.0, 1.0)
        }
    }

    pub fn done(&self, rig: &Rig) -> bool {
        let Some(clip) = self.clip.and_then(|clip| rig.clips.get(clip as usize)) else {
            return true;
        };
        !self.looped
            && if self.speed < 0.0 {
                self.time <= 0.0
            } else {
                self.time >= clip.duration
            }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ClipLayer {
    pub root: Option<u32>,
    pub mask: Option<Vec<bool>>,
    pub playing: Vec<Playing>,
}

impl ClipLayer {
    pub fn covers(&self, joint: usize) -> bool {
        self.mask
            .as_ref()
            .is_none_or(|mask| mask.get(joint).copied().unwrap_or(false))
    }
}

#[derive(Component, Clone, Debug, Serialize, Deserialize)]
#[serde(from = "AnimatorScene", into = "AnimatorScene")]
#[require(InterpolateTransform)]
pub struct Animator {
    rig: Arc<Rig>,
    start: Option<Start>,
    pub layers: Vec<ClipLayer>,
    pub overrides: Vec<(u32, Quat)>,
    pub model: Vec<Mat4>,
    pub root: Mat4,
}

impl Animator {
    #[must_use]
    pub fn new(rig: Arc<Rig>) -> Self {
        let model = rig.rest_model();
        Self {
            rig,
            start: None,
            layers: vec![ClipLayer::default()],
            overrides: Vec::new(),
            model,
            root: Mat4::IDENTITY,
        }
    }

    pub fn start(&self) -> Option<&Start> {
        self.start.as_ref()
    }

    pub fn rig(&self) -> &Arc<Rig> {
        &self.rig
    }

    pub fn play(&mut self, clip: &str, looped: bool, speed: f32) -> bool {
        self.play_on(0, clip, looped, speed)
    }

    pub fn crossfade(&mut self, clip: &str, seconds: f32, looped: bool, speed: f32) -> bool {
        self.crossfade_on(0, clip, seconds, looped, speed)
    }

    pub fn add_layer(&mut self, root: &str) -> Option<usize> {
        let root = self.rig.joint(root)?;
        let mut mask = vec![false; self.rig.joints.len()];
        mask[root] = true;
        for (index, joint) in self.rig.joints.iter().enumerate() {
            if joint
                .parent
                .is_some_and(|parent| mask.get(parent as usize).copied().unwrap_or(false))
            {
                mask[index] = true;
            }
        }
        self.layers.push(ClipLayer {
            root: Some(root as u32),
            mask: Some(mask),
            playing: Vec::new(),
        });
        Some(self.layers.len() - 1)
    }

    pub fn play_on(&mut self, layer: usize, clip: &str, looped: bool, speed: f32) -> bool {
        self.start_on(layer, clip, 0.0, looped, speed)
    }

    pub fn crossfade_on(
        &mut self,
        layer: usize,
        clip: &str,
        seconds: f32,
        looped: bool,
        speed: f32,
    ) -> bool {
        let Some(index) = self.rig.clip(clip) else {
            return false;
        };
        let rig = &*self.rig;
        let newest = self.layers.get(layer).and_then(|held| held.playing.last());
        if newest.is_some_and(|newest| newest.clip == Some(index as u32) && !newest.done(rig)) {
            return true;
        }
        self.start_on(layer, clip, seconds, looped, speed)
    }

    fn start_on(
        &mut self,
        layer: usize,
        clip: &str,
        seconds: f32,
        looped: bool,
        speed: f32,
    ) -> bool {
        let Some(index) = self.rig.clip(clip) else {
            return false;
        };
        let duration = self.rig.clips[index].duration;
        let Some(layer) = self.layers.get_mut(layer) else {
            return false;
        };
        layer.playing.push(Playing {
            clip: Some(index as u32),
            time: if speed < 0.0 { duration } else { 0.0 },
            looped,
            speed,
            fade: 0.0,
            seconds,
        });
        true
    }

    pub fn stop_on(&mut self, layer: usize, seconds: f32) -> bool {
        let Some(layer) = self.layers.get_mut(layer) else {
            return false;
        };
        if !layer.playing.is_empty() {
            layer.playing.push(Playing {
                clip: None,
                time: 0.0,
                looped: false,
                speed: 0.0,
                fade: 0.0,
                seconds,
            });
        }
        true
    }

    pub fn set_speed(&mut self, layer: usize, speed: f32) -> bool {
        let newest = self
            .layers
            .get_mut(layer)
            .and_then(|held| held.playing.last_mut());
        let Some(newest) = newest else {
            return false;
        };
        newest.speed = speed;
        true
    }

    pub fn set_rotation(&mut self, joint: &str, rotation: Quat) -> bool {
        let Some(joint) = self.rig.joint(joint) else {
            return false;
        };
        let joint = joint as u32;
        match self.overrides.iter_mut().find(|(held, _)| *held == joint) {
            Some((_, held)) => *held = rotation,
            None => self.overrides.push((joint, rotation)),
        }
        true
    }

    pub fn clear_rotation(&mut self, joint: &str) {
        if let Some(joint) = self.rig.joint(joint) {
            self.overrides.retain(|(held, _)| *held as usize != joint);
        }
    }

    pub fn joint_world(&self, joint: &str) -> Option<Mat4> {
        Some(self.root * self.joint_model(joint)?)
    }

    pub fn joint_model(&self, joint: &str) -> Option<Mat4> {
        self.model.get(self.rig.joint(joint)?).copied()
    }

    pub fn clip_duration(&self, clip: &str) -> Option<f32> {
        Some(self.rig.clips.get(self.rig.clip(clip)?)?.duration)
    }

    pub fn finished(&self, layer: usize) -> bool {
        self.layers
            .get(layer)
            .and_then(|held| held.playing.last())
            .is_none_or(|newest| newest.done(&self.rig))
    }
}

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[require(InterpolateTransform)]
pub struct Attachment {
    pub joint: String,
    #[serde(default)]
    pub translation: [f32; 3],
    #[serde(default = "no_rotation")]
    pub rotation: [f32; 4],
    #[serde(default = "unit_scale")]
    pub scale: [f32; 3],
}

impl Attachment {
    #[must_use]
    pub fn new(joint: impl Into<String>) -> Self {
        Self {
            joint: joint.into(),
            translation: [0.0; 3],
            rotation: no_rotation(),
            scale: unit_scale(),
        }
    }

    pub fn offset(&self) -> Mat4 {
        offset(self.translation, self.rotation, self.scale)
    }
}
