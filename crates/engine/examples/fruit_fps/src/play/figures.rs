use engine::prelude::Component;
use engine_core::{Asset, DrawKind, MeshFile};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::names::{HEAD, HIP, LEFT_HAND, RIGHT_HAND};
use crate::play::skinned::FigureLimbs;

pub const FIGURE: DrawKind = DrawKind("figure");
pub const GROUND: DrawKind = DrawKind("ground");
pub const REVEALED: DrawKind = DrawKind("revealed");
pub const FIELD_AT_CAMERA: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

pub const LEFT_EYE: &str = "L_Eye";
pub const RIGHT_EYE: &str = "R_Eye";
pub const RIGHT_ARM: [&str; 3] = ["R_Upperarm", "R_Forearm", RIGHT_HAND];
pub const LEFT_ARM: [&str; 3] = ["L_Upperarm", "L_Forearm", LEFT_HAND];
pub const RIGHT_LEG: [&str; 3] = ["R_Thigh", "R_Calf", "R_Foot"];
pub const LEFT_LEG: [&str; 3] = ["L_Thigh", "L_Calf", "L_Foot"];

#[derive(Component, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Forms {
    pub mid: Asset<MeshFile>,
    pub far: Asset<MeshFile>,
}

pub fn biped() -> FigureLimbs {
    let named = |names: [&str; 3]| names.map(str::to_owned);
    FigureLimbs {
        head: HEAD.to_owned(),
        hip: HIP.to_owned(),
        legs: [named(LEFT_LEG), named(RIGHT_LEG)],
        right_arm: named(RIGHT_ARM),
        left_arm: named(LEFT_ARM),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Form {
    #[default]
    Full,
    Mid,
    Far,
}
