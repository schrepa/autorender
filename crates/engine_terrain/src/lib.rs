//! Heightfield terrain data: tile pyramids, the layer stack
//! that generates heights, tiles with their samples, height queries, the
//! patches a view draws and the `ETR1` tile file format.
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
//! tile, or from `ETR1` files. Each tile is a pure function of its key, so
//! tiles stream in any order and neighbors agree on shared samples.
//!
//! Cells split into two triangles along the diagonal from their (x, z+1)
//! corner to their (x+1, z) corner, as `parry3d` heightfields split them;
//! renderers, collision and [`Tile::height_at`] use the same split.

mod bake;
pub mod codec;
mod erosion;
mod format;
pub mod gpu;
mod hydrology;
pub mod jobs;
mod layers;
mod nav;
mod noise;
mod path;
mod placement;
mod pyramid;
mod rivers;
mod select;
mod surface;
mod tile;

pub use bake::{BakeKind, Baked, TerrainBake};
pub use erosion::{Erosion, ErosionLayer, ErosionStats};
pub use format::{
    FileHeader, FileTile, FormatError, TERRAIN_FILE_MAGIC, TERRAIN_FILE_VERSION, decode_tile,
    encode_file,
};
pub use hydrology::{
    FALL_HEIGHT, FALL_SLOPE, FLOW_RANGE, Fall, Hydrology, HydrologyLayer, HydrologySample,
    HydrologyStats, LAKE_RANGE, RIVER_FAR, WETNESS_RANGE, WaterGrid,
};
pub use layers::{
    Blend, ErosionStage, HeightImage, HeightmapLayer, HydrologyStage, Layer, LayerError, Layers,
    NoiseLayer, PathStage, Stamp, StampOp,
};
pub use nav::{NavOptions, NavPath, Obstacle, WalkSearch, walk};
pub use path::{
    Bridge, HeightGrid, MASK_MARGIN, MAX_PATH_POINTS, PATH_FAR, PORTAL_HEIGHT, PROFILE_STEP,
    PathLayer, PathProfile, PathSample, PathStats, Route, RouteOptions, route,
};
pub use placement::{
    AUTHORED_LAYER, AuthoredPlacement, FarPlacement, Placed, PlacementChoice, PlacementLayer,
    Placing, merge_placed,
};
pub use pyramid::{Pyramid, PyramidError, TileKey};
pub use rivers::{BANK_OFFSET, RIVER_STEP, River, RiverPoint, Rivers};
pub use select::{
    LodRanges, Patch, Selection, WantedTile, WantedView, select_patches, wanted_tiles,
};
pub use surface::{
    Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Control, MAX_BIOMES, MAX_SURFACE_MATERIALS,
    Mask, NoiseMask, Remap, SampleInputs,
};
pub use tile::{Generator, Tile, TileBand, TileSample, merge_bands};

/// The finest-to-coarsest pyramid level any layer, bake, walk, route or
/// placement names at most.
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
