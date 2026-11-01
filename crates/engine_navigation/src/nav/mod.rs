use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Arc;

use engine_codec::{self as codec, Reader, Writer, record, tagged};
use engine_core::{Error, ErrorCode};
use engine_math::collections::HashMap;
use engine_math::{clamp, max, min};
use engine_places::ground::{
    GroundBridge, GroundGrid, GroundObstacle, MAX_LEVEL, Pyramid, TILE_QUADS, TILE_SAMPLES,
    TileKey,
};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

mod corridor;
mod flood;
mod graph;
mod ground;
mod portals;
mod search;

use corridor::{Corridor, Pull};
use flood::{Buckets, Flood, Stepper, octile};
use graph::Graph;
pub use ground::WalkGround;
pub use portals::{Portal, Side, TilePortals};
pub use search::{WalkData, WalkSearch, Want};

pub(crate) const LEVELS: (f64, f64) = (0.0, MAX_LEVEL as f64);
pub(crate) const MAX_SLOPE: (f64, f64) = (0.01, 10.0);
pub(crate) const RADIUS: (f64, f64) = (0.0, 100.0);
pub(crate) const MAX_WATER: (f64, f64) = (0.0, 100.0);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NavOptions {
    pub level: u32,
    pub max_slope: f32,
    pub radius: f32,
    pub max_water: f32,
}

impl Default for NavOptions {
    fn default() -> Self {
        Self {
            level: 0,
            max_slope: 0.8,
            radius: 0.5,
            max_water: 0.6,
        }
    }
}

impl NavOptions {
    pub fn validate(&self) -> Result<(), Error> {
        whole(self.level, "level", LEVELS)?;
        within(self.max_slope, "max_slope", MAX_SLOPE)?;
        within(self.radius, "radius", RADIUS)?;
        within(self.max_water, "max_water", MAX_WATER)
    }

    pub fn gait(&self, pyramid: &Pyramid) -> Gait {
        Gait {
            level: self.level.min(pyramid.levels() - 1),
            max_slope: self.max_slope,
            max_water: self.max_water,
        }
    }
}

pub(crate) fn kept(value: f32, (low, high): (f64, f64), default: f32) -> f32 {
    if !value.is_finite() {
        return default;
    }
    clamp(value, low as f32, high as f32)
}

fn within(value: f32, field: &str, (low, high): (f64, f64)) -> Result<(), Error> {
    if (low as f32..=high as f32).contains(&value) {
        return Ok(());
    }
    let shown: f64 = value.to_string().parse().unwrap_or(f64::from(value));
    Err(Error::new(ErrorCode::OutOfRange, format!("$.{field}"))
        .range(low, high)
        .got(shown))
}

fn whole(value: u32, field: &str, (low, high): (f64, f64)) -> Result<(), Error> {
    let (low_whole, high_whole) = (low as u64, high as u64);
    if (low_whole..=high_whole).contains(&u64::from(value)) {
        return Ok(());
    }
    Err(Error::new(ErrorCode::OutOfRange, format!("$.{field}"))
        .range(low, high)
        .got(value)
        .nearest(u64::from(value).clamp(low_whole, high_whole)))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gait {
    pub level: u32,
    pub max_slope: f32,
    pub max_water: f32,
}

record!(Gait {
    level,
    max_slope,
    max_water,
});

impl Gait {
    pub fn validate(&self, pyramid: &Pyramid) -> Result<(), Error> {
        if self.level >= pyramid.levels() {
            return Err(Error::new(ErrorCode::OutOfRange, "$.level")
                .at_most(u64::from(pyramid.levels().saturating_sub(1)))
                .got(self.level));
        }
        NavOptions {
            level: self.level,
            max_slope: self.max_slope,
            max_water: self.max_water,
            ..NavOptions::default()
        }
        .validate()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NavPath {
    pub points: Vec<[f32; 3]>,
    pub length: f32,
    pub level: u32,
    pub searched: usize,
}

record!(NavPath {
    points,
    length,
    level,
    searched,
});

const FOLLOW: f32 = 4.0;
const SNAP: i64 = 8;
const SIDE: usize = TILE_SAMPLES as usize;
const QUADS: i64 = TILE_QUADS as i64;
pub const PORTAL_RUN: usize = 32;
const STRAIGHT: u32 = 256;
const DIAGONAL: u32 = 362;
const UNREACHED: u32 = u32::MAX;
const EXPAND_COST: u64 = 16;
const NODE_COST: u64 = 64;
const PULL_REACH: usize = 1024;
const NEIGHBORS: [(i64, i64); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

type Sample = [i64; 2];

fn position(sample: Sample, spacing: f32) -> Vec2 {
    Vec2::new(sample[0] as f32, sample[1] as f32) * spacing
}
