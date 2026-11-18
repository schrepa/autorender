use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::{Component, Entity};
use engine_core::{DrawKind, Error, ErrorCode, Expected, SceneCx};
use glam::{Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const FIGURE: DrawKind = DrawKind("figure");
pub const GROUND: DrawKind = DrawKind("ground");
pub const REVEALED: DrawKind = DrawKind("revealed");
pub const FIELD_AT_CAMERA: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

pub const HIP: &str = "Hip";
pub const RIGHT_ARM: [&str; 3] = ["R_Upperarm", "R_Forearm", "R_Hand"];
pub const LEFT_ARM: [&str; 3] = ["L_Upperarm", "L_Forearm", "L_Hand"];
pub const RIGHT_LEG: [&str; 3] = ["R_Thigh", "R_Calf", "R_Foot"];
pub const LEFT_LEG: [&str; 3] = ["L_Thigh", "L_Calf", "L_Foot"];

#[derive(Clone, Default)]
pub struct FigureBytes(Arc<BTreeMap<String, Arc<[u8]>>>);

impl FigureBytes {
    pub fn new(files: BTreeMap<String, Arc<[u8]>>) -> Self {
        Self(Arc::new(files))
    }

    pub fn get(&self, uri: &str) -> Option<&Arc<[u8]>> {
        self.0.get(uri)
    }
}

impl PartialEq for FigureBytes {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|((uri, bytes), (known, held))| uri == known && Arc::ptr_eq(bytes, held))
    }
}

impl std::fmt::Debug for FigureBytes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.0.keys()).finish()
    }
}

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Figure {
    pub skeleton: String,
    pub clips: BTreeMap<String, String>,
    #[serde(skip)]
    pub bytes: FigureBytes,
}

impl Figure {
    pub fn files(&self) -> Vec<String> {
        std::iter::once(self.skeleton.clone())
            .chain(self.clips.values().cloned())
            .collect()
    }

    pub fn present(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let mut held = BTreeMap::new();
        if let Some(bytes) = cx.buffer(&self.skeleton, &format!("{path}.skeleton"))? {
            held.insert(self.skeleton.clone(), bytes);
        }
        for (name, uri) in &self.clips {
            if let Some(bytes) = cx.buffer(uri, &format!("{path}.clips.{name}"))? {
                held.insert(uri.clone(), bytes);
            }
        }
        self.bytes = FigureBytes(Arc::new(held));
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Blended {
    pub clip: String,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Stride {
    pub scale: f32,
    pub axis: [f32; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BaseLayer {
    pub clip: String,
    pub speed: f32,
    pub looping: bool,
    pub fade_seconds: f32,
    pub cue: u32,
    #[serde(default)]
    pub fit_seconds: Option<f32>,
    #[serde(default)]
    pub blend: Vec<Blended>,
    #[serde(default)]
    pub stride: Option<Stride>,
    #[serde(default)]
    pub synced: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UpperLayer {
    pub clip: String,
    pub weight: f32,
    pub looping: bool,
    pub from: String,
    pub feather: u32,
    pub speed: f32,
    pub cue: u32,
    #[serde(default)]
    pub window: Option<[f32; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Aim {
    pub pitch: f32,
    pub chain: Vec<String>,
    #[serde(default)]
    pub yaw: f32,
    #[serde(default)]
    pub hips: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HoldSpace {
    World,
    Figure,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandTarget {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub pole: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Hold {
    pub space: HoldSpace,
    pub weight: f32,
    #[serde(default)]
    pub right: Option<HandTarget>,
    #[serde(default)]
    pub left: Option<HandTarget>,
}

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FigurePlay {
    pub base: BaseLayer,
    #[serde(default)]
    pub upper: Option<UpperLayer>,
    #[serde(default)]
    pub aim: Option<Aim>,
    #[serde(default)]
    pub hold: Option<Hold>,
    #[serde(default)]
    pub crouch: Option<f32>,
    #[serde(default)]
    pub pin_feet: Option<f32>,
}

fn finite_at_least(value: f32, least: f32, path: String) -> Result<(), Error> {
    if value.is_finite() && value >= least {
        return Ok(());
    }
    Err(Error::new(ErrorCode::OutOfRange, path)
        .expected(Expected::Finite)
        .got(value))
}

fn unit_weight(value: f32, path: String) -> Result<(), Error> {
    finite_at_least(value, 0.0, path.clone())?;
    if value > 1.0 {
        return Err(Error::new(ErrorCode::OutOfRange, path).got(value));
    }
    Ok(())
}

fn finite_all(values: &[f32], path: String) -> Result<(), Error> {
    match values.iter().copied().find(|value| !value.is_finite()) {
        Some(value) => Err(Error::new(ErrorCode::InvalidValue, path)
            .expected(Expected::Finite)
            .got(value)),
        None => Ok(()),
    }
}

impl FigurePlay {
    pub fn checked(&self, path: &str) -> Result<(), Error> {
        finite_at_least(self.base.speed, 0.0, format!("{path}.base.speed"))?;
        finite_at_least(
            self.base.fade_seconds,
            0.0,
            format!("{path}.base.fade_seconds"),
        )?;
        if let Some(fit) = self.base.fit_seconds {
            finite_at_least(fit, 0.0, format!("{path}.base.fit_seconds"))?;
        }
        if let Some(upper) = &self.upper {
            finite_at_least(upper.speed, 0.0, format!("{path}.upper.speed"))?;
            unit_weight(upper.weight, format!("{path}.upper.weight"))?;
        }
        if let Some(aim) = &self.aim {
            finite_all(&[aim.pitch, aim.yaw], format!("{path}.aim"))?;
        }
        if let Some(hold) = &self.hold {
            unit_weight(hold.weight, format!("{path}.hold.weight"))?;
            for (side, hand) in [("right", hold.right), ("left", hold.left)] {
                if let Some(hand) = hand {
                    let values: Vec<f32> = hand
                        .position
                        .iter()
                        .chain(&hand.rotation)
                        .chain(&hand.pole)
                        .copied()
                        .collect();
                    finite_all(&values, format!("{path}.hold.{side}"))?;
                }
            }
        }
        if let Some(crouch) = self.crouch {
            finite_at_least(crouch, 0.0, format!("{path}.crouch"))?;
        }
        if let Some(contact) = self.pin_feet {
            finite_at_least(contact, 0.0, format!("{path}.pin_feet"))?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointAt {
    pub position: Vec3,
    pub rotation: Quat,
    pub found: bool,
}

#[derive(Component, Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FigureJoints {
    pub names: Vec<String>,
    #[serde(skip)]
    pub joints: Vec<JointAt>,
    #[serde(skip)]
    pub tick: u64,
    #[serde(skip)]
    pub base_seconds: f32,
    #[serde(skip)]
    pub base_done: bool,
    #[serde(skip)]
    pub base_cue: u32,
    #[serde(skip)]
    pub upper_seconds: f32,
    #[serde(skip)]
    pub upper_done: bool,
    #[serde(skip)]
    pub upper_cue: Option<u32>,
    #[serde(skip)]
    pub strides: BTreeMap<String, f32>,
    #[serde(skip)]
    pub steps: BTreeMap<String, f32>,
    #[serde(skip)]
    pub head_top: Option<Vec3>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Offset {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Held {
    #[serde(default)]
    pub figure: Option<String>,
    pub joint: String,
    pub offset: Offset,
    #[serde(skip)]
    target: Option<Entity>,
}

impl Held {
    pub fn on(figure: Entity, joint: impl Into<String>, offset: Offset) -> Self {
        Self {
            figure: None,
            joint: joint.into(),
            offset,
            target: Some(figure),
        }
    }

    pub fn target(&self) -> Option<Entity> {
        self.target.filter(|found| *found != Entity::PLACEHOLDER)
    }

    pub fn checked(&self, path: &str) -> Result<(), Error> {
        let values: Vec<f32> = self
            .offset
            .translation
            .iter()
            .chain(&self.offset.rotation)
            .copied()
            .collect();
        finite_all(&values, format!("{path}.offset"))
    }

    pub fn resolved(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        if let (Some(id), Some(scene)) = (self.figure.as_deref(), cx.scene()) {
            let found = scene.entity(id).ok_or_else(|| {
                Error::new(ErrorCode::UnknownReference, format!("{path}.figure")).got(id)
            })?;
            self.target = Some(found);
        }
        Ok(self)
    }
}
