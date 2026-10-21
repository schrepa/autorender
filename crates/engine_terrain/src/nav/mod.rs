//! Navigation over a terrain: the samples of a pyramid level an
//! agent can stand on (not holes, not under water deeper than it wades, not
//! within its radius of a placed obstacle) and the cheapest walk between two
//! points across them, each step between neighboring samples never cutting
//! a corner, no steeper than it climbs and costing its run and its rise.
//! Roads' bridges carry walks at their decks' height over the water and
//! ground below them.
//!
//! A level's walk graph is cooked tile by tile ([`TilePortals`], a record per
//! tile and [`Gait`]): a tile's portals, the runs of ground an agent stands
//! on along its edges with its neighbors, with what walking inside the tile
//! between each two costs. A [`WalkSearch`] costs its ends to their tiles'
//! portals, finds the cheapest way over the graph between them, then walks
//! the samples of the tiles that way crosses, its corridor, among the
//! obstacles there; the walk is pulled taut wherever a straight stretch
//! stays walkable and follows the ground in steps of a few meters. Searches
//! read tiles' grounds ([`WalkGround`], from the tiles' records) and portals
//! as hosts hand them back, waiting for those they lack, and do their work
//! in budgeted steps, so agents' walks spread over frames: what one finds
//! depends on that data and its ends alone.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Arc;

use engine_math::collections::HashMap;
use engine_math::{max, min};
use engine_settings::settings;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::codec::{self, CodecError, Reader, Writer, record, tagged};
use crate::{
    Generator, Layers, MAX_LEVEL, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, Tile, TileKey,
    TileWater,
};

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
pub use portals::{Portal, Side, TilePortals, tile_portals};
pub use search::{WalkData, WalkSearch, Want};

settings! {
    /// How an agent walks.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct NavOptions {
        /// The pyramid level it walks: 0 the finest.
        pub level: u32 = 0 => range(0, MAX_LEVEL),
        /// The steepest step it climbs: rise per meter.
        pub max_slope: f32 = 0.8 => range(0.01, 10.0),
        /// Its radius, by which obstacles widen, meters.
        pub radius: f32 = 0.5 => range(0.0, 100.0),
        /// The deepest water it wades, meters.
        pub max_water: f32 = 0.6 => range(0.0, 100.0),
    }
}

impl NavOptions {
    /// The gait walks with these options take over `pyramid`: at its
    /// coarsest level at most.
    pub fn gait(&self, pyramid: &Pyramid) -> Gait {
        Gait {
            level: self.level.min(pyramid.levels() - 1),
            max_slope: self.max_slope,
            max_water: self.max_water,
        }
    }
}

/// What a walk graph depends on of how agents walk: the level they walk,
/// the steepest step they climb (rise per meter) and the deepest water they
/// wade (meters). Walks of one gait read one graph, whatever their radius.
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
    /// Ok where walks over `pyramid` take this gait: its level is one of
    /// the pyramid's and the rest keep [`NavOptions`]' rules.
    pub fn validate(&self, pyramid: &Pyramid) -> Result<(), String> {
        if self.level >= pyramid.levels() {
            return Err(format!(
                "gait level {} is past the pyramid's {} levels",
                self.level,
                pyramid.levels()
            ));
        }
        NavOptions {
            level: self.level,
            max_slope: self.max_slope,
            max_water: self.max_water,
            ..NavOptions::default()
        }
        .validate()
        .map_err(|invalid| invalid.to_string())
    }
}

/// A placed obstacle's footprint: its terrain-local center and radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstacle {
    pub center: Vec2,
    pub radius: f32,
}

/// A walk: its points (terrain-local, following the ground), its length
/// (meters), the level it walked and the samples and portals it searched.
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

/// Meters between the points a walk follows the ground by.
const FOLLOW: f32 = 4.0;
/// Samples around an end searched for ground to stand on.
const SNAP: i64 = 8;
/// Samples along each side of a tile's ground: its samples, edges included.
const SIDE: usize = TILE_SAMPLES as usize;
/// Samples between tiles' first samples along each axis.
const QUADS: i64 = TILE_QUADS as i64;
/// The most samples a portal spans along its tile's edge.
pub const PORTAL_RUN: usize = 32;
/// What steps cost, in 256ths of their level's sample spacing: a straight
/// step's run, a diagonal one's (the square root of two of it), and each
/// spacing of rise as much as a straight run.
const STRAIGHT: u32 = 256;
const DIAGONAL: u32 = 362;
/// The cost no walk reaches.
const UNREACHED: u32 = u32::MAX;
/// The work, in samples, of expanding a searched sample and of a portal of
/// the walk graph.
const EXPAND_COST: u64 = 16;
const NODE_COST: u64 = 64;
/// Cells of a walk pulled taut over from one anchor at most.
const PULL_REACH: usize = 1024;
/// A sample's eight neighbors, the four straight first.
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

/// A sample of a level: its column and row from the terrain's first corner.
type Sample = [i64; 2];

/// The terrain-local position of `sample`, `spacing` apart.
fn position(sample: Sample, spacing: f32) -> Vec2 {
    Vec2::new(sample[0] as f32, sample[1] as f32) * spacing
}
