use super::*;
use engine_core::RecordKey;

pub(super) struct Asking<'a, 'w, 's> {
    pub(super) jobs: &'a mut Jobs<'w, 's>,
    pub(super) records: &'a Records,
    pub(super) kinds: &'a [&'static LayerKind],
}

impl Asking<'_, '_, '_> {
    pub(super) fn take(&self, ticket: &Ticket) -> Option<Result<Arc<[u8]>, Error>> {
        self.records.get(ticket)
    }
}

pub(super) struct Work {
    pub(super) clock: Option<fn() -> f64>,
    pub(super) end: f64,
    pub(super) upload: std::cell::Cell<u64>,
}

impl Work {
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
    asking: &mut Asking<'_, '_, '_>,
) -> bool {
    let authored = state.terrain.layers.clone();
    if bake(state, &authored, asking, frame) {
        return true;
    }
    load_records(state, missing, (target, keep, named), work, asking, frame);
    false
}

fn load_records(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep, named): (&HashSet<TileKey>, &HashSet<TileKey>, &[TileKey]),
    work: &Work,
    asking: &mut Asking<'_, '_, '_>,
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
        .map(|recipe| recipe.key)
    else {
        return;
    };
    let generation = state.store.generation();
    let store = &state.store;
    for (ticket, _, _) in state.dispatched.take_where(|key, &(_, made, _)| {
        !(made == generation && keep.contains(key) && store.loads(key))
    }) {
        asking.jobs.cancel(ticket);
    }
    let mut handed: Vec<(TileKey, Generation, u32)> = state
        .dispatched
        .iter()
        .map(|(key, &(_, made, priority))| (key, made, priority))
        .collect();
    handed.sort_by_key(|&(_, _, priority)| priority);
    let mut inserted = false;
    for (key, made, _) in handed {
        if inserted && (work.expired() || !work.uploads()) {
            break;
        }
        let Some(output) = state
            .dispatched
            .get(&key)
            .and_then(|(ticket, _, _)| asking.take(ticket))
        else {
            continue;
        };
        if let Some((ticket, _, _)) = state.dispatched.remove(&key) {
            asking.jobs.cancel(ticket);
        }
        match job_output(TILE_BUILD.name(), output, crate::jobs::tile_output) {
            Ok(tile) if made == generation && target.contains(&key) && tile.key() == key => {
                work.upload(2 * TILE_ARRAY_BYTES);
                state.failed.remove(&key);
                let stamps = state.reaching(key);
                state.store.arrive(Arc::new(tile), stamps);
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
        let job = state.tile_job(recipe.hash(), key);
        let ticket = asking.jobs.fetch(&TILE_BUILD, &job.identity(), priority);
        state.dispatched.insert(key, (ticket, generation, priority));
    }
    prefetch(state, (missing, named), (recipe, generation), asking, frame);
}

fn prefetch(
    state: &mut TerrainState,
    (missing, named): (&[(TileKey, u32)], &[TileKey]),
    (recipe, generation): (RecordKey, Generation),
    asking: &mut Asking<'_, '_, '_>,
    frame: u64,
) {
    let idle: Vec<(TileKey, Generation)> = state
        .prefetched
        .iter()
        .map(|(key, &(_, made))| (key, made))
        .collect();
    for (key, made) in idle {
        let output = state
            .prefetched
            .get(&key)
            .and_then(|(ticket, _)| asking.take(ticket));
        if let Some(output) = output {
            if let Some((ticket, _)) = state.prefetched.remove(&key) {
                asking.jobs.cancel(ticket);
            }
            if let Ok(tile) = job_output(TILE_BUILD.name(), output, crate::jobs::tile_output)
                && made == generation
                && tile.key() == key
                && !state.store.contains(&key)
            {
                let stamps = state.reaching(key);
                let left = super::store::Left { tile: Arc::new(tile), stamps, stamp: engine_core::Stamp::next(), current: true };
                state.store.keep_in_memory(left, frame);
            }
        } else if (made != generation || !named.contains(&key))
            && let Some((ticket, _)) = state.prefetched.remove(&key)
        {
            asking.jobs.cancel(ticket);
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
        let job = state.tile_job(recipe.hash(), key);
        let ticket = asking.jobs.fetch(&TILE_BUILD, &job.identity(), priority);
        state.prefetched.insert(key, (ticket, generation));
    }
}

pub(crate) struct SharedRecipe {
    pub(crate) key: RecordKey,
    pub(crate) stack: Arc<Layers>,
    materials: bool,
}

pub(crate) fn share_recipe(
    state: &mut TerrainState,
    jobs: &mut Jobs<'_, '_>,
    stacks: &mut Vec<(Arc<Layers>, RecordKey)>,
    released: &mut Vec<RecordKey>,
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
        .map(|(_, key)| *key);
    let stack_key = match shared {
        Some(key) => key,
        None => {
            let key = jobs.share(crate::jobs::share_stack(&stack));
            stacks.push((stack.clone(), key));
            key
        }
    };
    let recipe = TerrainRecipe {
        stack: stack_key.hash(),
        view,
        pyramid: state.terrain.pyramid,
        surface: state.terrain.surface.rules.authored().clone(),
        materials: state.materials,
    };
    let key = jobs.share(crate::jobs::share_recipe(&recipe));
    let shared = SharedRecipe {
        key,
        stack,
        materials: state.materials,
    };
    if let Some(old) = state.recipe.replace(shared) {
        released.push(old.key);
    }
}

pub(super) fn release(
    entries: &[TerrainState],
    stacks: &mut Vec<(Arc<Layers>, RecordKey)>,
    mut released: Vec<RecordKey>,
    jobs: &mut Jobs<'_, '_>,
) {
    stacks.retain(|(stack, key)| {
        let held = entries.iter().any(|state| {
            state
                .recipe
                .as_ref()
                .is_some_and(|recipe| Arc::ptr_eq(&recipe.stack, stack))
        });
        if !held {
            released.push(*key);
        }
        held
    });
    for key in released {
        let held = stacks.iter().any(|(_, shared)| *shared == key)
            || entries.iter().any(|state| {
                state
                    .recipe
                    .as_ref()
                    .is_some_and(|recipe| recipe.key == key)
            });
        if !held {
            jobs.unshare(key);
        }
    }
}
