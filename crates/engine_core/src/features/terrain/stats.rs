//! What terrain streaming reports: its counters and each terrain's bakes.

use super::*;

/// Counters of the last streaming update, over every terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainStats {
    pub terrains: u32,
    /// Tiles resident now.
    pub resident_tiles: u32,
    /// Tiles being generated, fetched or decoded, and every tile the
    /// camera's view and collision want that is not resident yet.
    pub loading_tiles: u32,
    pub missing_tiles: u32,
    /// Of the resident tiles, those holding their shape alone so far:
    /// heights and holes, drawn with coarser levels'
    /// materials while the rest loads.
    pub shaped_tiles: u32,
    /// Tiles the view wants that wait while the camera moves fast.
    pub waiting_tiles: u32,
    /// Tiles kept in memory after the view left them and
    /// their bytes, tiles that returned from there since terrains started,
    /// and tiles generating ahead of the view.
    pub cached_tiles: u32,
    pub cached_bytes: u64,
    pub restored_tiles: u64,
    pub idle_tiles: u32,
    /// Tiles that generated in bands of rows on several workers since
    /// terrains started.
    pub banded_tiles: u64,
    /// Resident tiles lacking objects their levels place with nothing
    /// under way to place them: none unless streaming lost
    /// track of them.
    pub unplaced_tiles: u32,
    /// Wanted tiles beyond the tile budget or the device's capacity.
    pub dropped_tiles: u32,
    /// Patches the camera draws, and of those how many stand in for finer
    /// levels not resident yet.
    pub patches: u32,
    pub fallback_patches: u32,
    /// Terrains whose bakes are in progress: they
    /// load no tiles until done.
    pub baking: u32,
    /// Bridges terrains' roads span, tunnels they bore,
    /// and waterfalls their rivers drop over.
    pub bridges: u32,
    pub tunnels: u32,
    pub falls: u32,
    /// Of the resident tiles and patches, those of terrains' surroundings.
    pub surroundings_tiles: u32,
    pub surroundings_patches: u32,
    /// Tile reads that failed since terrains started (read again later),
    /// and falls drawn without their white water, which the asset library
    /// refused; [`Terrains::latest_read_failure`] and
    /// [`Terrains::latest_fall_refusal`] say why the latest did.
    pub failed_reads: u64,
    pub refused_falls: u64,
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
