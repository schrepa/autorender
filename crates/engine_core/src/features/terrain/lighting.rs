//! Each terrain's lighting ([`engine_terrain::lighting`]): the sky its
//! ground sees and its horizons in every direction, which only its ground's
//! shape gives. Once its finest pinned level is resident throughout, those
//! tiles' heights are one job's input and its output one record, kept by the
//! input's content hash: the host hands it back from the cook's packs or its
//! store, or runs the job on a worker and keeps what it makes. Heights that
//! change (an edit reaching that level) make a new input and a new record;
//! the lighting before stays until it arrives.

use super::*;

/// A terrain's lighting and its asking for it: the record once made and the
/// content hash of the input it was made from; the job asking for it, with
/// its input's hash; the tiles and map sizes the last input read; and the
/// frame its job last failed at.
#[derive(Default)]
pub(super) struct Lighting {
    pub(super) made: Option<Arc<TerrainLighting>>,
    from: Option<u128>,
    pub(super) job: Option<(JobId, u128)>,
    tiles: Vec<Arc<Tile>>,
    sizes: (u32, u32),
    failed: Option<u64>,
}

impl Lighting {
    /// Whether it is asked for and not here yet, or not asked for at all.
    pub(super) fn left(&self) -> bool {
        self.job.is_some() || self.made.is_none()
    }

    /// Whether its job last failed: it asks again a while after.
    pub(super) fn failed(&self) -> bool {
        self.failed.is_some()
    }
}

/// The level a terrain's lighting reads: the finest of the coarsest
/// `pinned` levels, kept resident over the whole terrain. Its bounce
/// transfer and macro colors read the same level.
pub(super) fn lighting_level(pyramid: &Pyramid, pinned: u32) -> u32 {
    pyramid.levels().saturating_sub(pinned.max(1))
}

/// Whether `settings` ask for lighting: sky visibility or long sun
/// shadows.
pub(super) fn lit(settings: &TerrainSettings) -> bool {
    settings.enabled && (settings.sky_visibility || settings.sun_shadows)
}

/// Takes `state`'s lighting once its job is done, and asks for it anew,
/// within `slots`, once its level's heights differ from those it was made
/// from (or a while after its job failed); without `settings` asking for it
/// the lighting goes.
pub(super) fn light(
    state: &mut TerrainState,
    settings: &TerrainSettings,
    slots: &mut JobBudget<'_>,
    frame: u64,
) {
    let lighting = &mut state.whole.lighting;
    if !lit(settings) {
        if let Some((id, _)) = lighting.job.take() {
            slots.jobs.cancel(id);
        }
        *lighting = Lighting::default();
        return;
    }
    if let Some((id, hash)) = lighting.job
        && let Some(output) = slots.jobs.take(id)
    {
        lighting.job = None;
        match job_output(
            LightingJob::KIND.name(),
            output,
            engine_terrain::jobs::lighting_output,
        ) {
            Ok(made) => {
                lighting.made = Some(Arc::new(made));
                lighting.from = Some(hash);
                lighting.failed = None;
            }
            Err(error) => {
                state.failures.record(frame, error);
                lighting.failed = Some(frame);
            }
        }
    }
    if lighting.failed.is_some_and(|at| frame < at + RETRY_AFTER) {
        return;
    }
    let level = lighting_level(&state.terrain.pyramid, settings.pinned_levels);
    let Some(tiles) = source_tiles(state, level) else {
        return;
    };
    let lighting = &state.whole.lighting;
    // The same tiles make the same input: nothing to ask again.
    let same = tiles.len() == lighting.tiles.len()
        && tiles
            .iter()
            .zip(&lighting.tiles)
            .all(|(tile, held)| Arc::ptr_eq(tile, held));
    let sizes = (settings.sky_resolution, settings.shadow_resolution);
    if same && lighting.sizes == sizes {
        return;
    }
    let job = lighting_job(state, level, &tiles, sizes);
    let input = job.input();
    let hash = content_hash(&input);
    let lighting = &mut state.whole.lighting;
    lighting.tiles = tiles;
    lighting.sizes = sizes;
    if lighting.from == Some(hash) {
        if let Some((id, _)) = lighting.job.take() {
            slots.jobs.cancel(id);
        }
        return;
    }
    if lighting.job.is_some_and(|(_, asked)| asked == hash) {
        return;
    }
    let priority = job_priority(VIEW_JOBS, 0);
    if !slots.take(priority) {
        // Asked for again next frame.
        lighting.tiles.clear();
        return;
    }
    if let Some((id, _)) = lighting.job.take() {
        slots.jobs.cancel(id);
    }
    let record = RecordKey::new(
        &engine_terrain::SOURCE,
        LightingJob::KIND.name(),
        &hash.to_le_bytes(),
    );
    let id = slots
        .jobs
        .submit_record(LightingJob::KIND.name(), input, priority, record);
    lighting.job = Some((id, hash));
}

/// The tiles of `state`'s `level` its lighting reads, row by row: each one
/// resident and current, or where its surroundings leave its ground to
/// their terrain and load no tile, its nearest resident ancestor; `None`
/// while any is missing or regenerates.
pub(super) fn source_tiles(state: &TerrainState, level: u32) -> Option<Vec<Arc<Tile>>> {
    let pyramid = state.terrain.pyramid;
    pyramid
        .level_tiles(level)
        .map(|key| {
            if !state.store.loads(&key) {
                return state.store.get(&key).cloned();
            }
            let size = pyramid.tile_size(level);
            let low = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
            let cut = state.around.as_ref().is_some_and(|around| {
                let (first, last) = around.cutout;
                low.x >= first.x
                    && low.y >= first.y
                    && low.x + size <= last.x
                    && low.y + size <= last.y
            });
            if !cut {
                return None;
            }
            let mut ancestor = pyramid.parent(key);
            while let Some(at) = ancestor {
                if !state.store.loads(&at) {
                    return state.store.get(&at).cloned();
                }
                ancestor = pyramid.parent(at);
            }
            None
        })
        .collect()
}

/// The job lighting `state` from `tiles`, its `level`'s tiles row by row
/// (or their ancestors standing in), with maps `sizes` texels a side: the
/// level's samples over the whole terrain.
fn lighting_job(
    state: &TerrainState,
    level: u32,
    tiles: &[Arc<Tile>],
    (sky, horizons): (u32, u32),
) -> LightingJob {
    let grid = ground_grid(state, level, tiles);
    LightingJob {
        samples: grid.samples,
        spacing: grid.spacing,
        extent: state.terrain.pyramid.extent(),
        heights: grid.heights,
        sky,
        horizons,
    }
}

/// A level's samples over a whole terrain, as its lighting and its bounce
/// transfer read them: `samples` along x and z, `spacing` meters apart, row
/// by row along z.
pub(super) struct GroundGrid {
    pub(super) samples: [u32; 2],
    pub(super) spacing: f32,
    pub(super) heights: Box<[f32]>,
}

/// `state`'s `level` from `tiles`, its tiles row by row (or their ancestors
/// standing in).
pub(super) fn ground_grid(state: &TerrainState, level: u32, tiles: &[Arc<Tile>]) -> GroundGrid {
    let pyramid = state.terrain.pyramid;
    let [tiles_x, tiles_z] = pyramid.tiles(level);
    let quads = TILE_QUADS as usize;
    let columns = tiles_x as usize * quads + 1;
    let rows = tiles_z as usize * quads + 1;
    let mut heights = vec![0.0_f32; columns * rows];
    for (key, tile) in pyramid.level_tiles(level).zip(tiles) {
        let first = [usize::from(key.x) * quads, usize::from(key.z) * quads];
        let held = tile.key();
        // An ancestor stands in at its own samples, read between.
        let coarser = u32::from(held.level) - level;
        let scale = 1.0 / (1u32 << coarser) as f32;
        let offset = [
            (u32::from(key.x) * TILE_QUADS) as f32 * scale
                - (u32::from(held.x) * TILE_QUADS) as f32,
            (u32::from(key.z) * TILE_QUADS) as f32 * scale
                - (u32::from(held.z) * TILE_QUADS) as f32,
        ];
        for z in 0..=quads {
            for x in 0..=quads {
                let height = if coarser == 0 {
                    tile.height(x as i32, z as i32)
                } else {
                    between(
                        tile,
                        offset[0] + x as f32 * scale,
                        offset[1] + z as f32 * scale,
                    )
                };
                heights[(first[1] + z) * columns + first[0] + x] = height;
            }
        }
    }
    GroundGrid {
        samples: [columns as u32, rows as u32],
        spacing: pyramid.spacing(level),
        heights: heights.into_boxed_slice(),
    }
}

/// `tile`'s height at its fractional sample `(x, z)`, between the samples
/// around it.
fn between(tile: &Tile, x: f32, z: f32) -> f32 {
    let (x0, z0) = (x.floor(), z.floor());
    let (tx, tz) = (x - x0, z - z0);
    let (x0, z0) = (x0 as i32, z0 as i32);
    let a = tile.height(x0, z0);
    let b = tile.height(x0 + 1, z0);
    let c = tile.height(x0, z0 + 1);
    let d = tile.height(x0 + 1, z0 + 1);
    let near = a + (b - a) * tx;
    let far = c + (d - c) * tx;
    near + (far - near) * tz
}
