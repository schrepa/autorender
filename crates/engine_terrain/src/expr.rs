use serde::{Deserialize, Serialize};

use crate::surface::{Facing, NoiseMask};

pub use crate::expr_layer::{EXPRESSION, ExpressionLayer, ExpressionSettings, expression_kind};

pub const MAX_EXPR_DEPTH: usize = 8;
pub const MAX_EXPR_LIST: usize = 16;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Expr {
    Constant(f32),
    Map(String),
    Ramp(Ramp),
    Remap(Remap),
    Facing(Facing),
    Noise(NoiseMask),
    Mul(Vec<Expr>),
    Add(Vec<Expr>),
    Min(Vec<Expr>),
    Max(Vec<Expr>),
    Invert(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Ramp {
    pub map: String,
    pub from: f32,
    pub to: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Remap {
    pub of: Box<Expr>,
    pub from: f32,
    pub to: f32,
}

impl Expr {
    pub fn one() -> Self {
        Self::Constant(1.0)
    }
}

impl engine_core::Setting for Expr {}

crate::codec::tagged!(Expr {
    0 => Constant(value),
    1 => Map(name),
    2 => Ramp(ramp),
    3 => Remap(remap),
    4 => Facing(facing),
    5 => Noise(noise),
    6 => Mul(list),
    7 => Add(list),
    8 => Min(list),
    9 => Max(list),
    10 => Invert(expr),
});

crate::codec::record!(Ramp { map, from, to });
crate::codec::record!(Remap { of, from, to });
