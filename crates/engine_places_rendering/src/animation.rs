use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::{Component, Entity};
use engine_codec::Expected;
use engine_math::clamp;
use engine_place::{ApplyAt, Ask, Error, ErrorCode, MaterialHandle, MeshHandle};
use glam::Quat;
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
pub struct JointKeys {
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
    pub tracks: Vec<JointKeys>,
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
        let animator = Self::new(Arc::new(scene.rig));
        match scene.play {
            Some(start) => animator.starting(start),
            None => animator,
        }
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

fn wrapped(time: f32, duration: f32, looped: bool) -> f32 {
    if duration <= 0.0 {
        0.0
    } else if looped {
        time.rem_euclid(duration)
    } else {
        clamp(time, 0.0, duration)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Track {
    Clip {
        clip: u32,
        time: f32,
        looped: bool,
        speed: f32,
    },
    Rest,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fade {
    In { elapsed: f32, seconds: f32 },
    Full,
}

impl Fade {
    fn over(seconds: f32) -> Self {
        if seconds > 0.0 {
            Self::In {
                elapsed: 0.0,
                seconds,
            }
        } else {
            Self::Full
        }
    }

    pub fn weight(self) -> f32 {
        match self {
            Self::In { elapsed, seconds } => clamp(elapsed / seconds, 0.0, 1.0),
            Self::Full => 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Playing {
    pub track: Track,
    pub fade: Fade,
}

impl Playing {
    pub fn clip(&self) -> Option<u32> {
        match self.track {
            Track::Clip { clip, .. } => Some(clip),
            Track::Rest => None,
        }
    }

    pub fn done(&self, rig: &Rig) -> bool {
        match self.track {
            Track::Clip {
                clip,
                time,
                looped,
                speed,
            } => {
                let Some(clip) = rig.clips.get(clip as usize) else {
                    return true;
                };
                !looped
                    && if speed < 0.0 {
                        time <= 0.0
                    } else {
                        time >= clip.duration
                    }
            }
            Track::Rest => true,
        }
    }

    fn step(&mut self, rig: &Rig, seconds: f32) {
        self.fade = match self.fade {
            Fade::In {
                elapsed,
                seconds: over,
            } if elapsed + seconds < over => Fade::In {
                elapsed: elapsed + seconds,
                seconds: over,
            },
            Fade::In { .. } | Fade::Full => Fade::Full,
        };
        if let Track::Clip {
            clip,
            time,
            looped,
            speed,
        } = &mut self.track
        {
            let duration = rig.clips.get(*clip as usize).map_or(0.0, |clip| clip.duration);
            *time = wrapped(*time + *speed * seconds, duration, *looped);
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

    fn step(&mut self, rig: &Rig, seconds: f32) {
        for playing in &mut self.playing {
            playing.step(rig, seconds);
        }
        let Some(full) = self
            .playing
            .iter()
            .rposition(|playing| playing.fade == Fade::Full)
        else {
            return;
        };
        self.playing.drain(..full);
        if self
            .playing
            .first()
            .is_some_and(|playing| playing.track == Track::Rest)
        {
            self.playing.remove(0);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AnimatorCommand {
    Play {
        layer: usize,
        clip: String,
        looped: bool,
        speed: f32,
    },
    Blend {
        layer: usize,
        clip: String,
        seconds: f32,
        looped: bool,
        speed: f32,
    },
    Stop {
        layer: usize,
        seconds: f32,
    },
    Speed {
        layer: usize,
        speed: f32,
    },
    Layer {
        root: String,
    },
    Aim {
        joint: String,
        rotation: Option<Quat>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Animate {
    pub animator: Entity,
    pub command: AnimatorCommand,
}

impl Ask for Animate {
    const NAME: &'static str = "animate";
    const AT: ApplyAt = ApplyAt::TickStart;

    fn node(&self) -> Option<Entity> {
        Some(self.animator)
    }
}

#[derive(Component, Clone, Debug, Serialize, Deserialize)]
#[serde(from = "AnimatorScene", into = "AnimatorScene")]
pub struct Animator {
    rig: Arc<Rig>,
    revision: u64,
    start: Option<Start>,
    layers: Vec<ClipLayer>,
    overrides: Vec<(u32, Quat)>,
    joints: Vec<Entity>,
}

impl Animator {
    #[must_use]
    pub fn new(rig: Arc<Rig>) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            rig,
            revision: NEXT.fetch_add(1, Ordering::Relaxed),
            start: None,
            layers: vec![ClipLayer::default()],
            overrides: Vec::new(),
            joints: Vec::new(),
        }
    }

    #[must_use]
    pub fn starting(mut self, start: Start) -> Self {
        self.apply(&AnimatorCommand::Play {
            layer: 0,
            clip: start.clip.clone(),
            looped: start.looped,
            speed: start.speed,
        });
        self.start = Some(start);
        self
    }

    pub fn start(&self) -> Option<&Start> {
        self.start.as_ref()
    }

    pub fn rig(&self) -> &Arc<Rig> {
        &self.rig
    }

    pub fn rig_revision(&self) -> u64 {
        self.revision
    }

    pub fn layers(&self) -> &[ClipLayer] {
        &self.layers
    }

    pub fn overrides(&self) -> &[(u32, Quat)] {
        &self.overrides
    }

    pub fn joint_nodes(&self) -> &[Entity] {
        &self.joints
    }

    pub fn joint_node(&self, joint: &str) -> Option<Entity> {
        self.joints.get(self.rig.joint(joint)?).copied()
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

    #[doc(hidden)]
    pub fn apply(&mut self, command: &AnimatorCommand) -> bool {
        match command {
            AnimatorCommand::Play {
                layer,
                clip,
                looped,
                speed,
            } => self.start_on(*layer, clip, 0.0, *looped, *speed),
            AnimatorCommand::Blend {
                layer,
                clip,
                seconds,
                looped,
                speed,
            } => {
                let Some(index) = self.rig.clip(clip) else {
                    return false;
                };
                let rig = &*self.rig;
                let newest = self.layers.get(*layer).and_then(|held| held.playing.last());
                if newest.is_some_and(|newest| {
                    newest.clip() == Some(index as u32) && !newest.done(rig)
                }) {
                    return true;
                }
                self.start_on(*layer, clip, *seconds, *looped, *speed)
            }
            AnimatorCommand::Stop { layer, seconds } => {
                let Some(held) = self.layers.get_mut(*layer) else {
                    return false;
                };
                if !held.playing.is_empty() {
                    held.playing.push(Playing {
                        track: Track::Rest,
                        fade: Fade::over(*seconds),
                    });
                }
                true
            }
            AnimatorCommand::Speed { layer, speed } => {
                let newest = self
                    .layers
                    .get_mut(*layer)
                    .and_then(|held| held.playing.last_mut());
                match newest.map(|newest| &mut newest.track) {
                    Some(Track::Clip { speed: held, .. }) => {
                        *held = *speed;
                        true
                    }
                    Some(Track::Rest) | None => false,
                }
            }
            AnimatorCommand::Layer { root } => self.add_layer(root),
            AnimatorCommand::Aim {
                joint,
                rotation: Some(rotation),
            } => self.set_rotation(joint, *rotation),
            AnimatorCommand::Aim {
                joint,
                rotation: None,
            } => self.clear_rotation(joint),
        }
    }

    #[doc(hidden)]
    pub fn step(&mut self, seconds: f32) {
        let rig = &*self.rig;
        for layer in &mut self.layers {
            layer.step(rig, seconds);
        }
    }

    #[doc(hidden)]
    pub fn publish_joints(&mut self, joints: &[Entity]) {
        self.joints.clear();
        self.joints.extend_from_slice(joints);
    }

    fn add_layer(&mut self, root: &str) -> bool {
        let Some(root) = self.rig.joint(root) else {
            return false;
        };
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
        true
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
            track: Track::Clip {
                clip: index as u32,
                time: if speed < 0.0 { duration } else { 0.0 },
                looped,
                speed,
            },
            fade: Fade::over(seconds),
        });
        true
    }

    fn set_rotation(&mut self, joint: &str, rotation: Quat) -> bool {
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

    fn clear_rotation(&mut self, joint: &str) -> bool {
        let Some(joint) = self.rig.joint(joint) else {
            return false;
        };
        self.overrides.retain(|(held, _)| *held as usize != joint);
        true
    }
}

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Attachment {
    pub joint: String,
}

impl Attachment {
    #[must_use]
    pub fn new(joint: impl Into<String>) -> Self {
        Self {
            joint: joint.into(),
        }
    }
}
