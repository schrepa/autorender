use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Arc;

use engine_core::{Error, ErrorCode, Pyramid, TILE_QUADS, TILE_SAMPLES, TileKey};
use engine_math::collections::HashMap;
use engine_math::{max, min};
use engine_places::{GroundBridge, GroundGrid, GroundObstacle};
use engine_settings::settings;
use engine_terrain::codec::{self, Reader, Writer};
use engine_terrain::{MAX_LEVEL, record, tagged};
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

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct NavOptions {
        pub level: u32 = 0 => range(0, MAX_LEVEL),
        pub max_slope: f32 = 0.8 => range(0.01, 10.0),
        pub radius: f32 = 0.5 => range(0.0, 100.0),
        pub max_water: f32 = 0.6 => range(0.0, 100.0),
    }
}

impl NavOptions {
    pub fn gait(&self, pyramid: &Pyramid) -> Gait {
        Gait {
            level: self.level.min(pyramid.levels() - 1),
            max_slope: self.max_slope,
            max_water: self.max_water,
        }
    }
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
        .map_err(Error::from)
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
