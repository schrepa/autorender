use super::*;

pub(super) struct Work {
    pub(super) slice: u64,
    pub(super) clock: Option<fn() -> f64>,
    pub(super) end: f64,
    pub(super) upload: std::cell::Cell<u64>,
}

impl Work {
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

    pub(super) fn expired(&self) -> bool {
        self.clock.is_some_and(|clock| clock() >= self.end)
    }

    pub(super) fn uploads(&self) -> bool {
        self.upload.get() > 0
    }

    pub(super) fn upload(&self, bytes: u64) {
        self.upload.set(self.upload.get().saturating_sub(bytes));
    }
}

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

fn load_records(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep, named): (&HashSet<TileKey>, &HashSet<TileKey>, &[TileKey]),
    work: &Work,
    slots: &mut JobBudget<'_>,
    frame: u64,
) {
    let Some(recipe) = state
        .recipe
        .as_ref()
        .filter(|recipe| {
            state
                .baked
                .as_ref()
                .is_none_or(|baked| Arc::ptr_eq(baked, &recipe.stack))
        })
        .map(|recipe| recipe.key.clone())
    else {
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
        match job_output(TileJob::KIND.name(), output, crate::jobs::tile_output) {
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
                crate::jobs::tile_output,
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

pub(crate) struct SharedRecipe {
    pub(crate) key: String,
    pub(crate) stack: Arc<Layers>,
    materials: bool,
}

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
            let bytes = crate::jobs::share_stack(&stack);
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
    let bytes = crate::jobs::share_recipe(&recipe);
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
