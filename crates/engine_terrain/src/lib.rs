#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

pub mod bake;
pub mod bed;
pub mod bounce;
pub mod codec;
pub mod expr;
pub mod expr_layer;
pub mod flow;
pub mod given;
pub mod gpu;
pub mod hydraulics;
pub mod hydrology;
pub mod jobs;
pub mod layer;
pub mod layers;
pub mod lighting;
pub mod macro_colors;
pub mod noise;
pub mod path;
mod placement;
mod pyramid;
pub mod rivers;
mod runtime;
mod select;
mod shading;
pub mod shallow_water;
mod sights;
pub mod smokes;
mod steady_water;
mod surface;
mod tile;
mod tile_errors;
mod tile_water;

pub use bake::TerrainBake;
pub use bed::{Bed, NARROWEST, Reach};
pub use engine_core::{Pyramid, TILE_QUADS, TILE_SAMPLES, TileKey};
pub use expr::Expr;
pub use flow::{Block, Pool, RiverFlow, WEIR};
pub use given::{
    GivenRiver, MAX_CREDIT, MAX_DEPTH, MAX_DISCHARGE, MAX_GIVEN_RIVERS, MAX_PRECISION,
    MAX_RIVER_POINTS, MAX_WIDTH,
};
pub use hydrology::{
    FLOW_RANGE, Fall, Hydrology, HydrologySample, HydrologyStats, LAKE_RANGE, RIVER_FAR,
    WETNESS_RANGE,
};
pub use layer::{
    AddTerrainLayers, Below, ChangeMap, FineRegion, LayerBake, LayerFiles, LayerKind, LayerTile,
    NoFiles, Place, Row, SamplePoint, TerrainLayer, TerrainLayerKinds, TerrainLayersList,
    TileFrame,
};
pub use layers::{Blend, HeightImage, Layers, Stamp, StampOp};
pub use path::{Bridge, HeightGrid, PATH_FAR, PathSample, Route, RouteOptions, route};
pub use placement::{
    AUTHORED_LAYER, AuthoredPlacement, FarPlacement, Footprint, Obstacle, Placed, PlacementChoice,
    PlacementLayer, Placing, Solid,
};
pub use rivers::{RIVER_STEP, Rivers};
pub use runtime::LayerStats;
pub use runtime::{
    BRIDGE_SOLIDS, Cover, DECK_DEPTH, DECK_MARGIN, DECK_RISE, FALLS_SHOWN, FallView, GroundCover,
    LIVE_PIXELS, LONGEST_FRAME, LakeView, MAX_BOUNCE_LEVELS, MAX_COLLIDER_EXTENT,
    MAX_OCCUPANCY_LEVELS, MAX_PLACEMENT_LODS, MAX_PLACEMENT_PARTS, MIN_COLLIDER_EXTENT, MacroMap,
    MotionDetail, Occupancy, PlacedCollider, PlacedShape, PlacementAsset, PlacementLod,
    PlacementPart, PoolView, RenderPlacementAsset, RenderPlacementLayer, RenderPlacementPart,
    RenderSurfaceMaps, RenderTerrain, RenderTerrains, RiverFlowView, RiverSightView, SETTLED_KIND,
    SETTLED_PARITY, SIM_TOUCHES, SurfaceMaterial, SurfaceProjection, Surroundings,
    SurroundingsStatus, Terrain, TerrainBakes, TerrainBrick, TerrainControl, TerrainControlResult,
    TerrainDebugView, TerrainEdit, TerrainKey, TerrainPrefetch, TerrainRiver, TerrainSample,
    TerrainSettings, TerrainStats, TerrainSurface, TerrainWaterView, Terrains, TerrainsView, Water,
    WaterRead, WaterSourceCell, WaterWindow, live_steps, settled_alike, stream_now, terrain_key,
    work_between_frames,
};
pub use runtime::{
    HeightSample, RouteFound, StandingWater, TerrainEditControl, TerrainEdited, TerrainHeightQuery,
    TerrainHeightResult, TerrainRedoControl, TerrainRiverControl, TerrainRouteQuery,
    TerrainRouteResult, TerrainUndoControl,
};
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
    Biome, BiomeLayer, BiomeRules, Biomes, Control, Facing, MAX_BIOMES, MAX_SURFACE_MATERIALS,
    MacroCauses, NoiseMask, SampleInputs, SampleShape, Steep, SteepLayer,
};
pub use tile::{Generator, RIVER_EDGE_FAR, RIVER_EDGE_STEP, RIVER_EDGE_ZERO, Tile, TileSample};
pub use tile_errors::{DRY_CELL, ERROR_CELLS, ERROR_MIPS, ERROR_TEXELS, ErrorPyramid};
pub use tile_water::{
    BREAK_FULL, BREAK_ONSET, BREAK_THINNEST, TURBULENT_SPEED, TURBULENT_SWIRL, TileWater,
    WATER_FLOW_RANGE, WATER_REACH, WATER_SHEAR_RANGE, WATER_SPAN_LEAST, WATER_SPAN_MOST, WaterMesh,
    WaterPoint, WaterState, pack as pack_water, unpack as unpack_water,
};

pub const MAX_LEVEL: u32 = 15;

pub const TILE_STRIDE: u32 = TILE_SAMPLES + 2;
pub const PATCH_QUADS: u32 = 16;
pub const PATCHES_PER_TILE: u32 = TILE_QUADS / PATCH_QUADS;
