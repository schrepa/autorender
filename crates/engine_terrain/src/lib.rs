#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

pub mod bake;
pub mod expr_layer;
pub mod gpu;
pub mod jobs;
pub mod path;
mod runtime;
pub mod scene;
mod select;

use engine_codec as codec;
use engine_heightfield::given::MAX_GIVEN_RIVERS;
use engine_heightfield::layer;
use engine_heightfield::layer::TerrainLayer;
use engine_heightfield::path::PATH_FAR;
use engine_heightfield::tile::{Generator, Tile};
use engine_heightfield::{
    Blend, Bridge, GivenRiver, HydrologyBake, HydrologySample, HydrologyStats, Layers, PathSample,
};

pub use bake::TerrainBake;
pub use engine_heightfield::grid::{
    MAX_LEVEL, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey,
};
pub use engine_heightfield::{Stamp, StampOp};
pub use path::{HeightGrid, Route, RouteOptions, height_grid, route};
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
