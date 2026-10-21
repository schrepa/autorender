//! Loading a terrain's tiles: after its bakes, the tiles a frame finds
//! missing load as records, most urgent first, each a whole tile with every
//! object its level places. The host hands a record back from the cook's
//! packs or what it kept before, or runs the tile's job (on a worker, or in
//! place on a host without workers) and keeps what it makes: where a tile
//! came from is the host's alone, and loading only ever loads records.

use super::*;

/// A frame's terrain work: slices of samples' worth handed out while its
/// time lasts, or a single slice without a clock, and the bytes tiles may
/// bring the GPU.
pub(super) struct Work {
    pub(super) slice: u64,
    pub(super) clock: Option<fn() -> f64>,
    pub(super) end: f64,
    /// Bytes tiles may still bring the GPU this frame.
    pub(super) upload: std::cell::Cell<u64>,
}

impl Work {
    /// Refills `budget` with a slice once spent, while the frame's time
    /// lasts; whether it did.
    pub(super) fn refill(&self, budget: &mut u64) -> bool {
        if *budget > 0 {
            return false;
        }
        match self.clock {
            Some(clock) if clock() < self.end => {
                *budget = self.slice;
                true
            }
            _ => false,
        }
    }

    /// Whether the work's time is up; never without a clock.
    pub(super) fn expired(&self) -> bool {
        self.clock.is_some_and(|clock| clock() >= self.end)
    }

    /// Whether tiles may still become resident within the frame's upload
    /// budget.
    pub(super) fn uploads(&self) -> bool {
        self.upload.get() > 0
    }

    /// Spends `bytes` of the frame's upload budget.
    pub(super) fn upload(&self, bytes: u64) {
        self.upload.set(self.upload.get().saturating_sub(bytes));
    }
}

/// Loads `missing` tiles of `state`, most important first, as records,
/// once its bakes are done, within `slots`, the frame's job budget and
/// `work`'s time and uploads: `target` is every tile it loads and `keep`
/// every tile it keeps; `named` are the tiles places games name want,
/// which load into memory once nothing else waits. Frames load what they
/// find missing, and hosts what is left between frames. Returns whether a
/// bake is still in progress, which nothing loads behind.
pub(super) fn load_tiles(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep, named): (&HashSet<TileKey>, &HashSet<TileKey>, &[TileKey]),
    work: &Work,
    frame: u64,
    slots: &mut JobBudget<'_>,
) -> bool {
    let authored = state.terrain.layers.clone();
    if bake(state, &authored, slots, frame) {
        return true;
    }
    load_records(state, missing, (target, keep, named), work, slots, frame);
    false
}

/// Loads `state`'s `missing` tiles as records of its shared recipe: what
/// is no longer kept, no longer loads (back from memory), or was asked for
/// under edits or materials since, stops; records handed back become
/// resident, most urgent first, the first at once and the rest within
/// `work`'s time and uploads; then the missing tiles still loading, not
/// under way nor failed a moment ago, are asked for while `slots` admits
/// them, and the tiles places games name (`named`) load into memory once
/// nothing is missing.
fn load_records(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep, named): (&HashSet<TileKey>, &HashSet<TileKey>, &[TileKey]),
    work: &Work,
    slots: &mut JobBudget<'_>,
    frame: u64,
) {
    let Some(recipe) = state.recipe.as_ref().map(|recipe| recipe.key.clone()) else {
        return;
    };
    let generation = state.store.generation();
    let store = &state.store;
    state.dispatched.retain(|key, &mut (id, made, _)| {
        let wanted = made == generation && keep.contains(key) && store.loads(key);
        if !wanted {
            slots.jobs.cancel(id);
        }
        wanted
    });
    let mut handed: Vec<(TileKey, (JobId, Generation, u32))> = state
        .dispatched
        .iter()
        .map(|(key, &job)| (key, job))
        .collect();
    handed.sort_by_key(|&(_, (_, _, priority))| priority);
    let mut inserted = false;
    for (key, (id, made, _)) in handed {
        if inserted && (work.expired() || !work.uploads()) {
            break;
        }
        let Some(output) = slots.jobs.take(id) else {
            continue;
        };
        state.dispatched.remove(&key);
        match job_output(
            TileJob::KIND.name(),
            output,
            engine_terrain::jobs::tile_output,
        ) {
            Ok(tile) if made == generation && target.contains(&key) && tile.key() == key => {
                work.upload(2 * TILE_ARRAY_BYTES);
                state.failed.remove(&key);
                state.store.arrive(Arc::new(tile), None);
                inserted = true;
            }
            Ok(_) => {}
            Err(error) => {
                state.failed.insert(key, frame);
                state.failures.record(frame, error);
            }
        }
    }
    for &(key, priority) in missing {
        // Tiles that arrived this frame no longer load.
        if state.dispatched.contains_key(&key)
            || !state.store.loads(&key)
            || state
                .failed
                .get(&key)
                .is_some_and(|&at| frame < at + RETRY_AFTER)
        {
            continue;
        }
        if !slots.take(priority) {
            return;
        }
        let (job, record) = state.tile_record(&recipe, key);
        let id = slots
            .jobs
            .submit_record(TileJob::KIND.name(), job.identity(), priority, record);
        state.dispatched.insert(key, (id, generation, priority));
    }
    prefetch(state, (missing, named), (&recipe, generation), slots, frame);
}

/// Loads the tiles places games name (`named`, nearest first) into memory
/// while nothing is `missing`: their records, neither resident nor in
/// memory, at idle priority; those no longer named, or asked for under
/// edits since, stop.
fn prefetch(
    state: &mut TerrainState,
    (missing, named): (&[(TileKey, u32)], &[TileKey]),
    (recipe, generation): (&str, Generation),
    slots: &mut JobBudget<'_>,
    frame: u64,
) {
    let idle: Vec<(TileKey, JobId, Generation)> = state
        .prefetched
        .iter()
        .map(|(key, &(id, made))| (key, id, made))
        .collect();
    for (key, id, made) in idle {
        if let Some(output) = slots.jobs.take(id) {
            state.prefetched.remove(&key);
            if let Ok(tile) = job_output(
                TileJob::KIND.name(),
                output,
                engine_terrain::jobs::tile_output,
            ) && made == generation
                && tile.key() == key
                && !state.store.contains(&key)
            {
                state.store.keep_in_memory(Arc::new(tile), None, frame);
            }
        } else if made != generation || !named.contains(&key) {
            slots.jobs.cancel(id);
            slots.release();
            state.prefetched.remove(&key);
        }
    }
    if !missing.is_empty() || !state.dispatched.is_empty() {
        return;
    }
    for (place, &key) in named.iter().enumerate() {
        if state.prefetched.contains_key(&key)
            || state.store.contains(&key)
            || state.store.in_memory(&key)
        {
            continue;
        }
        let priority = job_priority(IDLE_JOBS, place as u32);
        if !slots.take(priority) {
            return;
        }
        let (job, record) = state.tile_record(recipe, key);
        let id = slots
            .jobs
            .submit_record(TileJob::KIND.name(), job.identity(), priority, record);
        state.prefetched.insert(key, (id, generation));
    }
}

/// A terrain's recipe as jobs hold it: its key, the stack it shares (the
/// layers its tiles generate from, which surroundings see from afar) and
/// whether its tiles hold control values.
pub(crate) struct SharedRecipe {
    pub(crate) key: String,
    pub(crate) stack: Arc<Layers>,
    materials: bool,
}

/// Shares the recipe `state`'s tiles generate from once its bakes are
/// done, and again whenever the layers its tiles generate from change:
/// their stack, shared once for every terrain seeing it, and how its
/// terrain sees it. Edits stay out of both, each tile's job carrying those
/// reaching it. Shares are named by their content, so records of jobs
/// reading them have the same keys in every session; `stacks` holds the
/// stacks shared with their keys, and the key of the recipe replaced goes
/// to `released`.
pub(crate) fn share_recipe(
    state: &mut TerrainState,
    jobs: &mut Jobs,
    stacks: &mut Vec<(Arc<Layers>, String)>,
    released: &mut Vec<String>,
) {
    let layers = state
        .baked
        .clone()
        .unwrap_or_else(|| state.terrain.layers.clone());
    if layers.unbaked().is_some() {
        return;
    }
    // Surroundings see their terrain's stack from afar.
    let (stack, view) = match &state.around {
        Some(around) => (
            around.base.clone(),
            Some(StackView {
                offset: -around.cutout.0,
                low: Vec2::ZERO,
                high: around.cutout.1 - around.cutout.0,
                blend: around.surroundings.blend,
            }),
        ),
        None => (layers, None),
    };
    if state.recipe.as_ref().is_some_and(|recipe| {
        Arc::ptr_eq(&recipe.stack, &stack) && recipe.materials == state.materials
    }) {
        return;
    }
    let shared = stacks
        .iter()
        .find(|(shared, _)| Arc::ptr_eq(shared, &stack))
        .map(|(_, key)| key.clone());
    let stack_key = match shared {
        Some(key) => key,
        None => {
            let bytes = engine_terrain::jobs::share_stack(&stack);
            let key = format!("stack-{:032x}", content_hash(&bytes));
            if !jobs.is_shared(&key) {
                jobs.share(key.clone(), bytes);
            }
            stacks.push((stack.clone(), key.clone()));
            key
        }
    };
    let recipe = TerrainRecipe {
        stack: stack_key,
        view,
        pyramid: state.terrain.pyramid,
        surface: state.terrain.surface.rules.authored().clone(),
        materials: state.materials,
    };
    let bytes = engine_terrain::jobs::share_recipe(&recipe);
    let key = format!("recipe-{:032x}", content_hash(&bytes));
    if !jobs.is_shared(&key) {
        jobs.share(key.clone(), bytes);
    }
    let shared = SharedRecipe {
        key,
        stack,
        materials: state.materials,
    };
    if let Some(old) = state.recipe.replace(shared) {
        released.push(old.key);
    }
}
