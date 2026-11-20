use std::collections::BTreeMap;
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::Arc;

use engine::prelude::{Component, Entity, Resource};
use engine_core::Asset;
use engine_rig::{Clip, PaletteRow, Skeleton};
use glam::{Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::names::{HEAD, HIP, LEFT_HAND, RIGHT_HAND, SPINE};
use crate::play::figures::{LEFT_EYE, LEFT_LEG, RIGHT_EYE, RIGHT_LEG};

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Figure {
    pub skeleton: Asset<Arc<Skeleton>>,
    pub clips: BTreeMap<String, Asset<Arc<Clip>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClipNo(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct JointNo(pub usize);

impl Figure {
    pub fn clip(&self, name: &str) -> Option<ClipNo> {
        (self.clips.contains_key(name))
            .then(|| ClipNo(self.clips.range::<str, _>((Unbounded, Excluded(name))).count()))
    }

    pub fn clip_name(&self, clip: ClipNo) -> Option<&str> {
        self.clips.keys().nth(clip.0).map(String::as_str)
    }
}

pub const POSED_JOINTS: [&str; 9] = [
    HEAD,
    SPINE,
    HIP,
    LEFT_LEG[2],
    RIGHT_LEG[2],
    RIGHT_HAND,
    LEFT_HAND,
    LEFT_EYE,
    RIGHT_EYE,
];

pub mod posed {
    pub const HEAD: usize = 0;
    pub const SPINE: usize = 1;
    pub const HIP: usize = 2;
    pub const LEFT_FOOT: usize = 3;
    pub const RIGHT_FOOT: usize = 4;
    pub const RIGHT_HAND: usize = 5;
    pub const LEFT_EYE: usize = 7;
    pub const RIGHT_EYE: usize = 8;
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct FigureLimbs {
    pub head: String,
    pub hip: String,
    pub legs: [[String; 3]; 2],
    pub right_arm: [String; 3],
    pub left_arm: [String; 3],
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct FigureFit {
    pub joints: Vec<String>,
    pub clip: String,
    pub height_m: f32,
}

#[derive(Resource, Default)]
pub struct FigureSight {
    pub frame: u64,
    pub judged: Vec<Entity>,
    pub shown: Vec<Entity>,
}

#[derive(Resource, Default)]
pub struct FigurePalette {
    pub rows: Vec<PaletteRow>,
    pub written: Vec<(usize, usize)>,
    slots: BTreeMap<Entity, (usize, usize)>,
}

impl FigurePalette {
    pub fn posed(&self) -> impl Iterator<Item = (Entity, (usize, usize))> + '_ {
        self.slots.iter().map(|(entity, slot)| (*entity, *slot))
    }

    pub fn slot(&mut self, entity: Entity, joints: usize) -> usize {
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

    pub fn keep(&mut self, drawn: &[Entity]) {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blended {
    pub clip: ClipNo,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stride {
    pub scale: f32,
    pub axis: [f32; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct BaseLayer {
    pub clip: ClipNo,
    pub speed: f32,
    pub looping: bool,
    pub fade_seconds: f32,
    pub cue: u32,
    pub fit_seconds: Option<f32>,
    pub blend: Vec<Blended>,
    pub stride: Option<Stride>,
    pub synced: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpperLayer {
    pub clip: ClipNo,
    pub weight: f32,
    pub looping: bool,
    pub from: JointNo,
    pub feather: u32,
    pub speed: f32,
    pub cue: u32,
    pub window: Option<[f32; 2]>,
    pub fit_seconds: Option<f32>,
    pub additive: bool,
    pub played_from: Option<JointNo>,
    pub fade_seconds: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Aim {
    pub pitch: f32,
    pub chain: Vec<JointNo>,
    pub yaw: f32,
    pub hips: Option<JointNo>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldSpace {
    World,
    Figure,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandTarget {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub pole: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChestFrame {
    pub joint: String,
    pub reference: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chest {
    pub joint: JointNo,
    pub reference: ClipNo,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hold {
    pub space: HoldSpace,
    pub weight: f32,
    pub chest: Option<Chest>,
    pub right: Option<HandTarget>,
    pub left: Option<HandTarget>,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct FigurePlay {
    pub base: BaseLayer,
    pub upper: Option<UpperLayer>,
    pub aim: Option<Aim>,
    pub hold: Option<Hold>,
    pub crouch: Option<f32>,
    pub pin_feet: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointAt {
    pub position: Vec3,
    pub rotation: Quat,
    pub found: bool,
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct FigureJoints {
    pub asked: Vec<Option<JointNo>>,
    pub joints: Vec<JointAt>,
    pub top: Option<Vec3>,
    pub hold_shift: Vec3,
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct FigureClock {
    pub base_seconds: f32,
    pub base_done: bool,
    pub base_cue: u32,
    pub upper_seconds: f32,
    pub upper_done: bool,
    pub upper_cue: Option<u32>,
    pub upper_rate: f32,
    pub fit: f32,
    pub fit_rest_m: Option<f32>,
    pub steps: Vec<f32>,
    pub drifts: Vec<f32>,
    pub bounds: Option<[Vec3; 2]>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepPose {
    pub joints: Vec<JointAt>,
    pub top: Option<Vec3>,
    pub hold_shift: Vec3,
}

pub trait StepPoser: Send + Sync {
    fn pose(
        &mut self,
        figure: Entity,
        play: Option<&FigurePlay>,
        asked: &[Option<JointNo>],
        out: &mut StepPose,
    ) -> bool;
}

#[derive(Resource, Default)]
pub struct StepJoints {
    poser: Option<Box<dyn StepPoser>>,
    kept: Vec<(Entity, Option<StepPose>)>,
}

impl StepJoints {
    pub fn renew(&mut self, poser: Option<Box<dyn StepPoser>>) {
        self.poser = poser;
        self.kept.clear();
    }

    pub fn posed(
        &mut self,
        figure: Entity,
        play: Option<&FigurePlay>,
        asked: &[Option<JointNo>],
    ) -> Option<&StepPose> {
        let at = match self.kept.iter().position(|(kept, _)| *kept == figure) {
            Some(at) => at,
            None => {
                let mut made = StepPose::default();
                let posed = (self.poser.as_mut())
                    .is_some_and(|poser| poser.pose(figure, play, asked, &mut made));
                self.kept.push((figure, posed.then_some(made)));
                self.kept.len() - 1
            }
        };
        self.kept[at].1.as_ref()
    }

    pub fn kept(&self, figure: Entity) -> Option<&StepPose> {
        let (_, kept) = self.kept.iter().find(|(kept, _)| *kept == figure)?;
        kept.as_ref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Offset {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: f32,
}

impl Offset {
    pub const STILL: Offset = Offset {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: 1.0,
    };
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Held {
    pub figure: Entity,
    pub joint: JointNo,
    pub offset: Offset,
}

impl Held {
    pub fn on(figure: Entity, joint: JointNo, offset: Offset) -> Self {
        Self {
            figure,
            joint,
            offset,
        }
    }
}
