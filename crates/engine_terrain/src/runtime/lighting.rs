use super::*;

#[derive(Default)]
pub(super) struct Lighting {
    pub(super) made: Option<Arc<TerrainLighting>>,
    from: Option<RecordKey>,
    pub(super) job: Option<(JobId, RecordKey)>,
    found: Option<(Basis, LightingJob, RecordKey)>,
    failed: Option<u64>,
}

impl Lighting {
    pub(super) fn left(&self) -> bool {
        self.job.is_some() || self.made.is_none()
    }

    pub(super) fn failed(&self) -> bool {
        self.failed.is_some()
    }
}

#[derive(Clone)]
pub(super) struct Basis {
    recipe: String,
    edits: Arc<[TerrainEdit]>,
    level: u32,
    sizes: (u32, u32),
}

impl Basis {
    pub(super) fn of(state: &TerrainState, recipe: &str, level: u32, sizes: (u32, u32)) -> Self {
        Self {
            recipe: recipe.to_owned(),
            edits: state.terrain.edits.clone(),
            level,
            sizes,
        }
    }

    pub(super) fn same(&self, other: &Self) -> bool {
        self.recipe == other.recipe
            && Arc::ptr_eq(&self.edits, &other.edits)
            && self.level == other.level
            && self.sizes == other.sizes
    }
}

pub(super) fn lighting_level(pyramid: &Pyramid, pinned: u32) -> u32 {
    pyramid.levels().saturating_sub(pinned.max(1))
}

pub(super) fn lit(settings: &TerrainSettings) -> bool {
    settings.enabled && (settings.sky_visibility || settings.sun_shadows)
}

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
    if let Some((id, record)) = lighting.job
        && let Some(output) = slots.jobs.take(id)
    {
        lighting.job = None;
        match job_output(
            LightingJob::KIND.name(),
            output,
            crate::jobs::lighting_output,
        ) {
            Ok(made) => {
                lighting.made = Some(Arc::new(made));
                lighting.from = Some(record);
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
    let Some(recipe) = state.recipe.as_ref().map(|recipe| recipe.key.clone()) else {
        return;
    };
    let level = lighting_level(&state.terrain.pyramid, settings.pinned_levels);
    let sizes = (settings.sky_resolution, settings.shadow_resolution);
    let basis = Basis::of(state, &recipe, level, sizes);
    let found = state
        .whole
        .lighting
        .found
        .as_ref()
        .filter(|(held, _, _)| held.same(&basis))
        .map(|(_, job, record)| (job.clone(), *record));
    let (job, record) = found.unwrap_or_else(|| {
        let (tiles, keys) = ground_tiles(state, &recipe, level);
        let job = LightingJob {
            recipe,
            level,
            tiles,
            sky: sizes.0,
            horizons: sizes.1,
        };
        let mut identity = keys;
        identity.extend_from_slice(&sizes.0.to_le_bytes());
        identity.extend_from_slice(&sizes.1.to_le_bytes());
        let record = RecordKey::new(
            &crate::sources::LIGHTING,
            LightingJob::KIND.name(),
            &identity,
        );
        state.whole.lighting.found = Some((basis, job.clone(), record));
        (job, record)
    });
    let lighting = &mut state.whole.lighting;
    if lighting.from == Some(record) {
        if let Some((id, _)) = lighting.job.take() {
            slots.jobs.cancel(id);
        }
        return;
    }
    if lighting.job.is_some_and(|(_, asked)| asked == record) {
        return;
    }
    let priority = job_priority(VIEW_JOBS, 0);
    if !slots.take(priority) {
        return;
    }
    if let Some((id, _)) = lighting.job.take() {
        slots.jobs.cancel(id);
    }
    let id = slots
        .jobs
        .submit_record(LightingJob::KIND.name(), job.input(), priority, record);
    lighting.job = Some((id, record));
}

pub(super) fn ground_tiles(
    state: &TerrainState,
    recipe: &str,
    level: u32,
) -> (Vec<(TileKey, Vec<Stamp>)>, Vec<u8>) {
    let pyramid = state.terrain.pyramid;
    let cut = |key: TileKey| {
        let size = pyramid.tile_size(u32::from(key.level));
        let low = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
        state.around.as_ref().is_some_and(|around| {
            let (first, last) = around.cutout;
            low.x >= first.x && low.y >= first.y && low.x + size <= last.x && low.y + size <= last.y
        })
    };
    let mut tiles = Vec::new();
    let mut keys = Vec::new();
    for key in pyramid.level_tiles(level) {
        let mut held = key;
        while cut(held) {
            let Some(parent) = pyramid.parent(held) else {
                break;
            };
            held = parent;
        }
        let (job, record) = state.tile_record(recipe, held);
        keys.extend_from_slice(&record.0);
        tiles.push((held, job.stamps));
    }
    (tiles, keys)
}
