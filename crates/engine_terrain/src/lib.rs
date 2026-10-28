#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod bake;
mod bed;
pub mod bounce;
pub mod codec;
mod erosion;
mod flow;
mod given;
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
mod relief;
mod rivers;
mod select;
mod shading;
pub mod shallow_water;
mod sights;
mod steady_water;
mod surface;
mod tile;
mod tile_errors;
mod tile_water;

pub use bake::{Bake, BakeKind, Baked, ErosionBake, HydrologyBake, PathBake, Stage, TerrainBake};
pub use bed::{Bed, NARROWEST, Reach};
pub use erosion::{Erosion, ErosionLayer, ErosionStats};
pub use flow::{Block, Pool, RiverFlow, WEIR};
pub use given::{
    GivenRiver, MAX_CREDIT, MAX_DEPTH, MAX_DISCHARGE, MAX_GIVEN_RIVERS, MAX_PRECISION,
    MAX_RIVER_POINTS, MAX_WIDTH, RiversLayer,
};
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
    AUTHORED_LAYER, AuthoredPlacement, FarPlacement, Footprint, Placed, PlacementChoice,
    PlacementLayer, Placing, Solid,
};
pub use pyramid::{Pyramid, PyramidError, TileKey};
pub use relief::{HummocksLayer, StepsLayer};
pub use rivers::{RIVER_STEP, Rivers};
pub use select::{LodRanges, WantedTile, WantedView, wanted_tiles};
pub use shading::TileShading;
pub use sights::{FlowSummary, LakeSight, RiverCourse, RiverListing, RiverSight};
pub use steady_water::{
    BLOCK_CELLS, BLOCK_GROUND, BLOCK_SIDE, COVER_POINTS, LiveWater, MARGIN, NO_BLOCK, REGION_CELLS,
    SUMMARY_SIDE, SettleRegion, SettleStart, SettledKey, SteadyCell, SteadySource, SteadySquare,
    SteadyStats, SteadySummary, SteadyWater, StoredBlocks, WaterPart, cover_cells, settle_regions,
    watered_tiles,
};
pub use surface::{
    Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Control, Facing, MAX_BIOMES,
    MAX_SURFACE_MATERIALS, MacroCauses, Mask, NoiseMask, Remap, SampleInputs, SampleShape, Steep,
    SteepLayer,
};
pub use tile::{Generator, RIVER_EDGE_FAR, RIVER_EDGE_STEP, RIVER_EDGE_ZERO, Tile, TileSample};
pub use tile_errors::{DRY_CELL, ERROR_CELLS, ERROR_MIPS, ERROR_TEXELS, ErrorPyramid};
pub use tile_water::{
    BREAK_FULL, BREAK_ONSET, BREAK_THINNEST, TURBULENT_SPEED, TURBULENT_SWIRL, TileWater,
    WATER_FLOW_RANGE, WATER_REACH, WATER_SHEAR_RANGE, WATER_SPAN_LEAST, WATER_SPAN_MOST, WaterMesh,
    WaterPoint, WaterState, pack as pack_water, unpack as unpack_water,
};

pub const MAX_LEVEL: u32 = 15;

pub const TILE_QUADS: u32 = 256;
pub const TILE_SAMPLES: u32 = TILE_QUADS + 1;
pub const TILE_STRIDE: u32 = TILE_SAMPLES + 2;
pub const PATCH_QUADS: u32 = 16;
pub const PATCHES_PER_TILE: u32 = TILE_QUADS / PATCH_QUADS;
