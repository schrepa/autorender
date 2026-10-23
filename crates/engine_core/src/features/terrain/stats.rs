//! What terrain streaming reports: its counters and each terrain's bakes.

use super::*;

/// Counters of the last streaming update, over every terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainStats {
    pub terrains: u32,
    /// Tiles resident now.
    pub resident_tiles: u32,
    /// Tiles whose records are asked for, and every tile the camera's view
    /// and collision want that is not resident yet.
    pub loading_tiles: u32,
    pub missing_tiles: u32,
    /// Tiles the view wants that wait while the camera moves fast.
    pub waiting_tiles: u32,
    /// Tiles kept in memory after the view left them and
    /// their bytes, tiles that returned from there since terrains started,
    /// and tiles loading into memory for places games name.
    pub cached_tiles: u32,
    pub cached_bytes: u64,
    pub restored_tiles: u64,
    pub prefetching_tiles: u32,
    /// Resident tiles whose water the update made again as their rivers
    /// run now, what stands in them holding them back, and those whose
    /// water was made from other rivers still waiting their turn.
    pub rewatered_tiles: u32,
    pub stale_water_tiles: u32,
    /// Records of level-0 tiles' settled water arrived, whose summaries
    /// coarser tiles take their flow and foam from (resident level-0 tiles
    /// take their water from those held whole), and those asked for.
    pub settled_records: u32,
    pub settling_records: u32,
    /// Resident tiles of level 1 and coarser shading with their shading
    /// (level 0's materials and normals averaged), and the records of such
    /// shading asked for: until a tile's arrives it shows the next coarser
    /// level's.
    pub shaded_tiles: u32,
    pub shading_records: u32,
    /// Wanted tiles beyond the tile budget or the device's capacity.
    pub dropped_tiles: u32,
    /// Patches the camera draws, and of those how many stand in for finer
    /// levels not resident yet.
    pub patches: u32,
    pub fallback_patches: u32,
    /// Terrains whose bakes are in progress: they
    /// load no tiles until done.
    pub baking: u32,
    /// Terrains whose lighting is here (the sky their ground sees, their
    /// horizons), and those whose lighting is asked for: until it arrives
    /// their ground sees the whole sky and casts no long shadows.
    pub lit: u32,
    pub lighting: u32,
    /// Terrains whose bounce transfer is here, and those whose transfer is
    /// asked for: the camera's view waits for it, as for their lighting.
    pub bounced: u32,
    pub bouncing: u32,
    /// Terrains whose macro colors are here (the tint and growth their
    /// bakes give their ground), and those asking for them: until they
    /// arrive their ground shows its materials' own colors.
    pub colored: u32,
    pub coloring: u32,
    /// Bridges terrains' roads span, tunnels they bore,
    /// and waterfalls their rivers drop over.
    pub bridges: u32,
    pub tunnels: u32,
    pub falls: u32,
    /// Emitters of spray and mist white water spawned at terrains' rapids,
    /// where their settled water loses energy fastest.
    pub white_water: u32,
    /// Of the resident tiles and patches, those of terrains' surroundings.
    pub surroundings_tiles: u32,
    pub surroundings_patches: u32,
    /// Whether the camera's view has yet to load whole since terrains
    /// arrived or [`Terrains::begin_loading`]: frames work
    /// on terrain for the settings' `loading_work_ms`, then `work_ms`.
    pub loading: bool,
}

/// A terrain's bakes: whether one is in progress,
/// and what its hydrology and erosion bakes found once done.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainBakes {
    pub baking: bool,
    pub hydrology: Option<HydrologyStats>,
    pub erosion: Option<ErosionStats>,
}
