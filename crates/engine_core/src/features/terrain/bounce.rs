use super::*;

#[derive(Default)]
pub(super) struct Bounce {
    pub(super) made: Option<Arc<TerrainBounce>>,
    from: Option<RecordKey>,
    pub(super) job: Option<(JobId, RecordKey)>,
    found: Option<(Basis, BounceJob, RecordKey)>,
    failed: Option<u64>,
}

impl Bounce {
    pub(super) fn left(&self) -> bool {
        self.job.is_some() || self.made.is_none()
    }
}

pub(super) fn waits(state: &TerrainState, settings: &TerrainSettings) -> bool {
    let held = &state.whole.bounce;
    bounces(settings)
        && state.around.is_none()
        && !state.whole.lighting.failed()
        && (held.job.is_some() || (held.made.is_none() && held.failed.is_none()))
}

const MOST_PROBES: u32 = 512;

const FEWEST_PROBES: u32 = 16;

pub(super) fn bounces(settings: &TerrainSettings) -> bool {
    settings.enabled && settings.sun_shadows && settings.bounce
}

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
    if let Some((id, record)) = state.whole.bounce.job
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
                state.whole.bounce.from = Some(record);
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
    if state.whole.lighting.made.is_none() {
        return;
    }
    let Some(recipe) = state.recipe.as_ref().map(|recipe| recipe.key.clone()) else {
        return;
    };
    let level = lighting_level(&state.terrain.pyramid, settings.pinned_levels);
    let probes = probe_columns(&state.terrain.pyramid, settings);
    let basis = Basis::of(state, &recipe, level, (probes, 0));
    let found = state
        .whole
        .bounce
        .found
        .as_ref()
        .filter(|(held, _, _)| held.same(&basis))
        .map(|(_, job, record)| (job.clone(), *record));
    let (job, record) = found.unwrap_or_else(|| {
        let (tiles, keys) = ground_tiles(state, &recipe, level);
        let job = BounceJob {
            recipe,
            level,
            tiles,
            probes,
        };
        let mut identity = keys;
        identity.extend_from_slice(&probes.to_le_bytes());
        let record = RecordKey::new(
            &engine_terrain::sources::BOUNCE,
            BounceJob::KIND.name(),
            &identity,
        );
        state.whole.bounce.found = Some((basis, job.clone(), record));
        (job, record)
    });
    let bounce = &mut state.whole.bounce;
    if bounce.from == Some(record) {
        if let Some((id, _)) = bounce.job.take() {
            slots.jobs.cancel(id);
        }
        return;
    }
    if bounce.job.is_some_and(|(_, asked)| asked == record) {
        return;
    }
    let priority = job_priority(VIEW_JOBS, 0);
    if !slots.take(priority) {
        return;
    }
    if let Some((id, _)) = bounce.job.take() {
        slots.jobs.cancel(id);
    }
    let id = slots
        .jobs
        .submit_record(BounceJob::KIND.name(), job.input(), priority, record);
    bounce.job = Some((id, record));
}
