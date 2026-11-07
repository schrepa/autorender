use super::*;
use engine_core::{FrameBudget, RecordKey};

const BUDGET: &str = "terrain";

pub(super) struct Asking<'a, 'w, 's> {
    pub(super) jobs: &'a mut Jobs<'w, 's>,
    pub(super) records: &'a Records,
    pub(super) kinds: &'a [&'static LayerKind],
}

impl Asking<'_, '_, '_> {
    pub(super) fn take(&self, ticket: &Interest<RecordKey>) -> Option<Result<Arc<[u8]>, Error>> {
        self.records.get(ticket)
    }
}

pub(super) struct Work<'a> {
    until: Option<(fn() -> f64, f64)>,
    budget: std::cell::RefCell<&'a mut FrameBudget>,
    spent: std::cell::Cell<u64>,
}

impl<'a> Work<'a> {
    pub(super) fn frame(budget: &'a mut FrameBudget) -> Self {
        Self {
            until: None,
            budget: std::cell::RefCell::new(budget),
            spent: std::cell::Cell::new(0),
        }
    }

    pub(super) fn between(budget: &'a mut FrameBudget, until: (fn() -> f64, f64)) -> Self {
        Self {
            until: Some(until),
            budget: std::cell::RefCell::new(budget),
            spent: std::cell::Cell::new(0),
        }
    }

    pub(super) fn expired(&self) -> bool {
        match self.until {
            Some((clock, end)) => clock() >= end,
            None => self.budget.borrow().expired(),
        }
    }

    pub(super) fn spend(&self, bytes: u64) -> bool {
        let spent = self.budget.borrow_mut().spend(BUDGET, bytes);
        if spent {
            self.spent.set(self.spent.get() + bytes);
        }
        spent
    }

    pub(super) fn carry(&self, bytes: u64) {
        self.budget.borrow_mut().spend(BUDGET, bytes);
    }

    pub(super) fn spent(&self) -> u64 {
        self.spent.get()
    }
}

pub(super) fn load_tiles(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep, named): (&HashSet<TileKey>, &HashSet<TileKey>, &[TileKey]),
    work: &Work<'_>,
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
    work: &Work<'_>,
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
    drop(state.dispatched.take_where(|key, &(_, made, _)| {
        !(made == generation && keep.contains(key) && store.loads(key))
    }));
    drop(state.failed.take_where(|key, _| !keep.contains(key)));
    let mut handed: Vec<(TileKey, Generation, u32)> = state
        .dispatched
        .iter()
        .map(|(key, &(_, made, priority))| (key, made, priority))
        .collect();
    handed.sort_by_key(|&(_, _, priority)| priority);
    for (key, made, _) in handed {
        if work.spent() > 0 && work.expired() {
            break;
        }
        let Some(output) = state
            .dispatched
            .get(&key)
            .and_then(|(ticket, _, _)| asking.take(ticket))
        else {
            continue;
        };
        match job_output(TILE_BUILD.name(), output, crate::jobs::tile_output) {
            Ok(tile) if made == generation && target.contains(&key) && tile.key() == key => {
                if !work.spend(2 * TILE_ARRAY_BYTES) {
                    break;
                }
                state.dispatched.remove(&key);
                let stamps = state.reaching(key);
                state.store.arrive(Arc::new(tile), stamps);
            }
            Ok(_) => {
                state.dispatched.remove(&key);
            }
            Err(error) => {
                if let Some((interest, _, _)) = state.dispatched.remove(&key) {
                    state.failed.insert(key, interest);
                }
                state.failures.record(frame, error);
            }
        }
    }
    for &(key, priority) in missing {
        if state.dispatched.contains_key(&key) || !state.store.loads(&key) {
            continue;
        }
        let job = state.tile_job(recipe.hash(), key);
        let input = job.identity();
        if state
            .failed
            .get(&key)
            .is_some_and(|held| *held.key() == RecordKey::of(&TILE_BUILD, &input))
        {
            continue;
        }
        state.failed.remove(&key);
        let ticket = asking.jobs.fetch(&TILE_BUILD, &input, priority);
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
            state.prefetched.remove(&key);
            if let Ok(tile) = job_output(TILE_BUILD.name(), output, crate::jobs::tile_output)
                && made == generation
                && tile.key() == key
                && !state.store.contains(&key)
            {
                let stamps = state.reaching(key);
                state.store.keep_unseen(Arc::new(tile), stamps, frame);
            }
        } else if made != generation || !named.contains(&key) {
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
    let materials = state.recipe.as_ref().map(|recipe| recipe.materials);
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
    state.recipe_changed(materials);
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
