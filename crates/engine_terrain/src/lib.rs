#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

pub mod bake;
pub mod expr;
pub mod expr_layer;
pub mod gpu;
pub mod jobs;
pub mod path;
mod placement;
mod runtime;
pub mod scene;
mod select;
mod surface;
mod tile;
mod tile_errors;

use engine_codec as codec;
use engine_heightfield::given::MAX_GIVEN_RIVERS;
use engine_heightfield::layer::TerrainLayer;
use engine_heightfield::path::PATH_FAR;
use engine_heightfield::{
    Bed, Blend, Bridge, GivenRiver, HeightImage, HydrologyBake, HydrologySample, HydrologyStats,
    Layers, PathSample, Reach, Rivers,
};
use engine_heightfield::{hydraulics, hydrology, layer, noise, rivers};
use engine_places::ground::Placed;

pub use bake::TerrainBake;
pub use engine_heightfield::grid::{
    MAX_LEVEL, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey,
};
pub use engine_heightfield::{Stamp, StampOp};
pub use expr::Expr;
pub use path::{HeightGrid, Route, RouteOptions, height_grid, route};
pub use placement::{
    AUTHORED_LAYER, AuthoredPlacement, FarPlacement, Footprint, Obstacle, PlacementChoice,
    PlacementLayer, Placing, Solid, footprint,
};
pub use runtime::LayerStats;
pub use runtime::{
    BRIDGE_SOLIDS, DECK_DEPTH, DECK_MARGIN, DECK_RISE, MAX_COLLIDER_EXTENT, MIN_COLLIDER_EXTENT,
    MotionDetail, PlacedCollider, PlacedShape, PlacementAsset, Surroundings, SurroundingsStatus,
    Terrain, TerrainBakes, TerrainBrick, TerrainEdit, TerrainKey, TerrainPrefetch, TerrainRiver,
    TerrainSample, TerrainScene, TerrainSettings, TerrainStats, TerrainSurface, Terrains,
};
pub use runtime::{
    HeightSample, RouteFound, StandingWater, TerrainEditControl, TerrainEdited, TerrainHeightQuery,
    TerrainHeightResult, TerrainRedoControl, TerrainRiverControl, TerrainRouteQuery,
    TerrainRouteResult, TerrainUndoControl,
};
pub use select::{LodRanges, WantedTile, WantedView, wanted_tiles};
pub use surface::{
    Biome, BiomeLayer, BiomeRules, Biomes, Control, Facing, MAX_BIOMES, MAX_SURFACE_MATERIALS,
    MacroCauses, NoiseMask, SampleInputs, SampleShape, SteepLayer,
};
pub use tile::{Generator, RIVER_EDGE_FAR, RIVER_EDGE_STEP, RIVER_EDGE_ZERO, Tile, TileSample};
pub use tile_errors::{ERROR_CELLS, ERROR_MIPS, ERROR_TEXELS, ErrorPyramid};

pub const PATCH_QUADS: u32 = 16;
pub const PATCHES_PER_TILE: u32 = TILE_QUADS / PATCH_QUADS;
