//! Each terrain's bounce transfer ([`engine_terrain::bounce`]): what probes
//! over its ground see of it, from the heights its lighting reads and its
//! recipe's surface. Once its lighting is here, those heights and its
//! recipe's key are one job's input and its output one record, kept by the
//! input's content hash: the host hands it back from the cook's packs or its
//! store, or runs the job on a worker and keeps what it makes. Heights that
//! change (an edit reaching that level) or another recipe make a new input
//! and a new record; the transfer before stays until it arrives.
//! Surroundings, ground seen from afar, take none.

use super::*;

/// A terrain's bounce transfer and its asking for it: the record once made
/// and the content hash of the input it was made from; the job asking for
/// it, with its input's hash; the tiles, probe columns and recipe the last
/// input read; and the frame its job last failed at.
#[derive(Default)]
pub(super) struct Bounce {
    pub(super) made: Option<Arc<TerrainBounce>>,
    from: Option<u128>,
    pub(super) job: Option<(JobId, u128)>,
    tiles: Vec<Arc<Tile>>,
    probes: u32,
    recipe: String,
    failed: Option<u64>,
}

impl Bounce {
    /// Whether it is asked for and not here yet, or not asked for at all.
    pub(super) fn left(&self) -> bool {
        self.job.is_some() || self.made.is_none()
    }
}

/// Whether `state`'s view waits for its bounce transfer, as it waits for
/// its lighting: `settings` ask for bounce light, it is no surroundings,
/// and its transfer is asked for, or not yet here without its job, or its
/// lighting's, having failed. Preparation ends only once none waits, so
/// views start on the transfer, never on the far field's estimate standing
/// in for it.
pub(super) fn waits(state: &TerrainState, settings: &TerrainSettings) -> bool {
    let held = &state.whole.bounce;
    bounces(settings)
        && state.around.is_none()
        && !state.whole.lighting.failed()
        && (held.job.is_some() || (held.made.is_none() && held.failed.is_none()))
}

/// The most probe columns along a terrain's side: larger terrains spread
/// theirs wider, keeping their records within tens of megabytes.
const MOST_PROBES: u32 = 512;

/// The fewest: a terrain smaller than its spacing still has a few.
const FEWEST_PROBES: u32 = 16;

/// Whether `settings` ask for bounce light: on, with the long sun shadows
/// renderers find the sun's light on the ground by.
pub(super) fn bounces(settings: &TerrainSettings) -> bool {
    settings.enabled && settings.sun_shadows && settings.bounce
}

/// The probe columns along each side of a terrain over `pyramid` that
/// `settings` ask for: the power of two keeping them within the settings'
/// spacing along its longer side, from `FEWEST_PROBES` to `MOST_PROBES`.
pub(super) fn probe_columns(pyramid: &Pyramid, settings: &TerrainSettings) -> u32 {
    let extent = pyramid.extent();
    let longer = crate::math::max(extent.x, extent.y);
    let wanted = (longer / settings.bounce_spacing).ceil();
    let wanted = if wanted.is_finite() && wanted < MOST_PROBES as f32 {
        wanted as u32
    } else {
        MOST_PROBES
    };
    wanted.next_power_of_two().clamp(FEWEST_PROBES, MOST_PROBES)
}

/// Takes `state`'s bounce transfer once its job is done, and asks for it
/// anew, within `slots`, once its lighting is here and the heights it reads,
/// its probe columns or its recipe differ from those the transfer was made
/// from (or a while after its job failed); without `settings` asking for it,
/// or for surroundings, the transfer goes.
pub(super) fn bounce(
    state: &mut TerrainState,
    settings: &TerrainSettings,
    slots: &mut JobBudget<'_>,
    frame: u64,
) {
    if !bounces(settings) || state.around.is_some() {
        if let Some((id, _)) = state.whole.bounce.job.take() {
            slots.jobs.cancel(id);
        }
        state.whole.bounce = Bounce::default();
        return;
    }
    if let Some((id, hash)) = state.whole.bounce.job
        && let Some(output) = slots.jobs.take(id)
    {
        state.whole.bounce.job = None;
        match job_output(
            BounceJob::KIND.name(),
            output,
            engine_terrain::jobs::bounce_output,
        ) {
            Ok(made) => {
                state.whole.bounce.made = Some(Arc::new(made));
                state.whole.bounce.from = Some(hash);
                state.whole.bounce.failed = None;
            }
            Err(error) => {
                state.failures.record(frame, error);
                state.whole.bounce.failed = Some(frame);
            }
        }
    }
    if state
        .whole
        .bounce
        .failed
        .is_some_and(|at| frame < at + RETRY_AFTER)
    {
        return;
    }
    // The lighting first: both read the same heights, and the lighting
    // shows sooner.
    if state.whole.lighting.made.is_none() {
        return;
    }
    let Some(recipe) = state.recipe.as_ref().map(|recipe| recipe.key.clone()) else {
        return;
    };
    let level = lighting_level(&state.terrain.pyramid, settings.pinned_levels);
    let Some(tiles) = source_tiles(state, level) else {
        return;
    };
    let probes = probe_columns(&state.terrain.pyramid, settings);
    let held = &state.whole.bounce;
    // The same tiles make the same heights: nothing to ask again.
    let same = tiles.len() == held.tiles.len()
        && tiles
            .iter()
            .zip(&held.tiles)
            .all(|(tile, held)| Arc::ptr_eq(tile, held));
    if same && held.probes == probes && held.recipe == recipe {
        return;
    }
    let grid = ground_grid(state, level, &tiles);
    let job = BounceJob {
        recipe: recipe.clone(),
        samples: grid.samples,
        spacing: grid.spacing,
        extent: state.terrain.pyramid.extent(),
        heights: grid.heights,
        probes,
    };
    let input = job.input();
    let hash = content_hash(&input);
    let bounce = &mut state.whole.bounce;
    bounce.tiles = tiles;
    bounce.probes = probes;
    bounce.recipe = recipe;
    if bounce.from == Some(hash) {
        if let Some((id, _)) = bounce.job.take() {
            slots.jobs.cancel(id);
        }
        return;
    }
    if bounce.job.is_some_and(|(_, asked)| asked == hash) {
        return;
    }
    // Views wait for it as for the lighting (`waits`).
    let priority = job_priority(VIEW_JOBS, 0);
    if !slots.take(priority) {
        // Asked for again next frame.
        bounce.tiles.clear();
        return;
    }
    if let Some((id, _)) = bounce.job.take() {
        slots.jobs.cancel(id);
    }
    let record = RecordKey::new(
        &engine_terrain::sources::BOUNCE,
        BounceJob::KIND.name(),
        &hash.to_le_bytes(),
    );
    let id = slots
        .jobs
        .submit_record(BounceJob::KIND.name(), input, priority, record);
    bounce.job = Some((id, hash));
}
