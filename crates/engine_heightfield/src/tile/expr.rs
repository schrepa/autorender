use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::surface::{Facing, NoiseMask};
use crate::Layers;
use crate::layer::{Below, Row, TerrainLayer};

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

impl engine_codec::Setting for Expr {}

engine_codec::tagged!(Expr {
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

engine_codec::record!(Ramp { map, from, to });
engine_codec::record!(Remap { of, from, to });

pub fn producer(layers: &[Arc<dyn TerrainLayer>], name: &str) -> Option<usize> {
    (layers.iter()).rposition(|layer| layer.outputs().iter().any(|output| output == name))
}

pub fn read_map(
    stack: &Layers,
    producer: Option<usize>,
    name: &str,
    row: &Row<'_>,
    heights: &[f32],
    out: &mut [f32],
) {
    let read = match producer {
        Some(index) => stack
            .layers()
            .get(index)
            .is_some_and(|layer| layer.map_row(name, Below::new(stack, index), row, heights, out)),
        None => false,
    };
    if !read {
        out.fill(0.0);
    }
}
