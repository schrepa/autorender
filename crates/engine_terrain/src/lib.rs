//! Heightfield terrain data: tile pyramids, the layer stack
//! that generates heights, tiles with their samples, height queries, the
//! patches a view draws, the shading coarser tiles take from level 0 and
//! the jobs that make tiles and bakes.
//!
//! A terrain is a square grid of level-0 tiles, each [`TILE_QUADS`] quads of
//! its sample spacing across; every coarser level halves the tiles along
//! each axis and doubles the spacing, up to a level of one tile. Tiles store
//! their samples with a border of one sample, so normals at their edges
//! match their neighbors'. Heights are `f32` meters above the terrain's
//! origin, identical bits on native targets and in browsers.
//!
//! Heights come from a [`Layers`] stack (procedural noise with erosion,
//! imported heightmaps and stamps, later layers winning) evaluated tile by
//! tile. Each tile is a pure function of its key, so tiles stream in any
//! order and neighbors agree on shared samples. A terrain's tiles of level 1
//! and coarser shade with level 0's materials and normals averaged
//! ([`TileShading`]), records of their own made from the level finer.
//!
//! Cells split into two triangles along the diagonal from their (x, z+1)
//! corner to their (x+1, z) corner, as `parry3d` heightfields split them;
//! renderers, collision and [`Layers::height_at`] use the same split.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

// `SOURCE`, which keys the records terrain jobs make: hosts keep them, and
// the cook ships them, while this code stays the same.
include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod bake;
mod bed;
pub mod bounce;
pub mod codec;
mod erosion;
mod flow;
pub mod gpu;
pub mod hydraulics;
mod hydrology;
pub mod jobs;
mod layers;
pub mod lighting;
pub mod macro_colors;
mod nav;
mod noise;
mod path;
mod placement;
mod pyramid;
mod rivers;
mod select;
mod shading;
pub mod shallow_water;
mod sights;
mod steady_water;
mod surface;
mod tile;
mod tile_water;

pub use bake::{Bake, BakeKind, Baked, ErosionBake, HydrologyBake, PathBake, Stage, TerrainBake};
pub use bed::{Bed, Reach};
pub use erosion::{Erosion, ErosionLayer, ErosionStats};
pub use flow::{Block, Pool, RiverFlow, WEIR};
pub use hydrology::{
    FLOW_RANGE, Fall, Hydrology, HydrologyLayer, HydrologySample, HydrologyStats, LAKE_RANGE,
    RIVER_FAR, WETNESS_RANGE,
};
pub use layers::{
    Blend, ErosionStage, HeightImage, HeightmapLayer, HydrologyStage, Layer, LayerError, Layers,
    NoiseLayer, PathStage, Stamp, StampOp,
};
pub use nav::{
    Gait, NavOptions, NavPath, Obstacle, PORTAL_RUN, Portal, Side, TilePortals, WalkData,
    WalkGround, WalkSearch, Want, tile_portals,
};
pub use path::{
    Bridge, HeightGrid, MASK_MARGIN, MAX_PATH_POINTS, PATH_FAR, PORTAL_HEIGHT, PROFILE_STEP,
    PathLayer, PathProfile, PathSample, PathStats, Route, RouteOptions, route,
};
pub use placement::{
    AUTHORED_LAYER, AuthoredPlacement, FarPlacement, Placed, PlacementChoice, PlacementLayer,
    Placing,
};
pub use pyramid::{Pyramid, PyramidError, TileKey};
pub use rivers::{RIVER_STEP, Rivers};
pub use select::{
    LodRanges, Patch, Selection, WantedTile, WantedView, select_patches, wanted_tiles,
};
pub use shading::TileShading;
pub use sights::{FlowSummary, LakeSight, RiverCourse, RiverSight};
pub use steady_water::{
    BLOCK_CELLS, BLOCK_GROUND, BLOCK_SIDE, MARGIN, NO_BLOCK, REGION_CELLS, SUMMARY_SIDE,
    SettleRegion, SettleStart, SettledKey, SteadyCell, SteadySource, SteadySquare, SteadyStats,
    SteadySummary, SteadyWater, StoredBlocks, WaterPart, settle_regions, watered_tiles,
};
pub use surface::{
    Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Control, MAX_BIOMES, MAX_SURFACE_MATERIALS,
    Mask, NoiseMask, Remap, SampleInputs, SampleShape, Steep, SteepLayer,
};
pub use tile::{Generator, RIVER_EDGE_FAR, RIVER_EDGE_STEP, RIVER_EDGE_ZERO, Tile, TileSample};
pub use tile_water::{
    BREAK_FULL, BREAK_ONSET, BREAK_THINNEST, TURBULENT_SPEED, TURBULENT_SWIRL, TileWater,
    WATER_FLOW_RANGE, WATER_OBSTACLE_STEP, WATER_RADIUS_STEP, WATER_SHEAR_RANGE, WATER_WIDTH_RANGE,
    WaterPoint, WaterState, pack as pack_water, unpack as unpack_water,
};

/// The finest-to-coarsest pyramid level any layer, bake, walk graph, route
/// or placement names at most.
pub const MAX_LEVEL: u32 = 15;

/// Quads along each side of a tile.
pub const TILE_QUADS: u32 = 256;
/// Samples along each side of a tile, its edges included.
pub const TILE_SAMPLES: u32 = TILE_QUADS + 1;
/// Samples along each side of a stored tile: its samples and a border of
/// one sample around them.
pub const TILE_STRIDE: u32 = TILE_SAMPLES + 2;
/// Quads along each side of a patch, the unit views draw; a tile holds
/// [`PATCHES_PER_TILE`] of them along each side.
pub const PATCH_QUADS: u32 = 16;
pub const PATCHES_PER_TILE: u32 = TILE_QUADS / PATCH_QUADS;
