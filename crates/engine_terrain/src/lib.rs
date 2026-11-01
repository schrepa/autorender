#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

pub mod bake;
pub mod expr;
pub mod expr_layer;
pub mod gpu;
pub mod jobs;
pub mod layers;
pub mod path;
mod placement;
mod runtime;
pub mod scene;
mod select;
mod sights;
pub mod smokes;
mod surface;
mod terrain;
mod tile;
mod tile_errors;
mod views;

pub use bake::TerrainBake;
pub use engine_core::{Pyramid, TILE_QUADS, TILE_SAMPLES, TileKey};
pub use expr::Expr;
pub use layers::{Stamp, StampOp};
pub use path::{HeightGrid, Route, RouteOptions, route};
pub use placement::{
    AUTHORED_LAYER, AuthoredPlacement, FarPlacement, Footprint, Obstacle, Placed, PlacementChoice,
    PlacementLayer, Placing, Solid,
};
pub use runtime::LayerStats;
pub use runtime::{
    BRIDGE_SOLIDS, DECK_DEPTH, DECK_MARGIN, DECK_RISE, FALLS_SHOWN, FallView,
    LakeView, MAX_COLLIDER_EXTENT,
    MAX_PLACEMENT_LODS, MAX_PLACEMENT_PARTS, MIN_COLLIDER_EXTENT, MacroMap,
    MotionDetail, PlacedCollider, PlacedShape, PlacementAsset, PlacementLod,
    PlacementPart, RenderPlacementAsset, RenderPlacementLayer, RenderPlacementPart,
    RenderSurfaceMaps, RiverFlowView, RiverSightView,
    SurfaceMaterial, SurfaceProjection, Surroundings,
    SurroundingsStatus, Terrain, TerrainBakes, TerrainBrick, TerrainControl, TerrainControlResult,
    TerrainDebugView, TerrainEdit, TerrainKey, TerrainPrefetch, TerrainRiver, TerrainSample,
    TerrainSettings, TerrainStats, TerrainSurface, TerrainWaterView, Terrains, TerrainsView,
    stream_now, terrain_key,
    work_between_frames,
};
pub use runtime::{
    HeightSample, RouteFound, StandingWater, TerrainEditControl, TerrainEdited, TerrainHeightQuery,
    TerrainHeightResult, TerrainRedoControl, TerrainRiverControl, TerrainRouteQuery,
    TerrainRouteResult, TerrainUndoControl,
};
pub use select::{LodRanges, WantedTile, WantedView, wanted_tiles};
pub use sights::{FlowSummary, LakeSight, RiverCourse, RiverListing, RiverSight};
pub use surface::{
    Biome, BiomeLayer, BiomeRules, Biomes, Control, Facing, MAX_BIOMES, MAX_SURFACE_MATERIALS,
    MacroCauses, NoiseMask, SampleInputs, SampleShape, Steep, SteepLayer,
};
pub use tile::{Generator, RIVER_EDGE_FAR, RIVER_EDGE_STEP, RIVER_EDGE_ZERO, Tile, TileSample};
pub use tile_errors::{DRY_CELL, ERROR_CELLS, ERROR_MIPS, ERROR_TEXELS, ErrorPyramid};

pub const MAX_LEVEL: u32 = 15;

pub const TILE_STRIDE: u32 = TILE_SAMPLES + 2;
pub const PATCH_QUADS: u32 = 16;
pub const PATCHES_PER_TILE: u32 = TILE_QUADS / PATCH_QUADS;
