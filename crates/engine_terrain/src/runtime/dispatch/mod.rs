use super::*;
use engine_core::{FrameBudget, Landed, RecordKey};

const BUDGET: &str = "terrain";

pub(super) struct Asking<'a, 'w, 's> {
    pub(super) jobs: &'a mut Jobs<'w, 's>,
    pub(super) records: &'a Records,
    pub(super) kinds: &'a [&'static LayerKind],
}

pub(super) struct Work<'a> {
    budget: std::cell::RefCell<&'a mut FrameBudget>,
    spent: std::cell::Cell<u64>,
}

impl<'a> Work<'a> {
    pub(super) fn frame(budget: &'a mut FrameBudget) -> Self {
        Self {
            budget: std::cell::RefCell::new(budget),
            spent: std::cell::Cell::new(0),
        }
    }

    pub(super) fn expired(&self) -> bool {
        self.budget.borrow().expired()
    }

    pub(super) fn spend(&self, bytes: u64) -> bool {
        let spent = self.budget.borrow_mut().spend(BUDGET, bytes);
        if spent {
            self.spent.set(self.spent.get() + bytes);
        }
        spent
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
    let Some((recipe, materials)) = state
        .recipe
        .as_ref()
        .filter(|recipe| {
            state
                .baked
                .as_ref()
                .is_none_or(|baked| Arc::ptr_eq(baked, &recipe.stack))
        })
        .map(|recipe| (recipe.key, recipe.materials))
    else {
        return;
    };
    let generation = state.store.generation();
    let store = &state.store;
    let stale = state.slots.keys_where(|key, slot| match slot {
        TileSlot::Loading { made, .. } => {
            !(*made == generation && keep.contains(key) && store.loads(key))
        }
        TileSlot::Failed { .. } => !keep.contains(key),
        TileSlot::Prefetching { .. } => false,
    });
    for key in stale {
        state.slot_event(key, SlotEvent::Done);
    }
    let mut handed: Vec<(TileKey, Generation, u32)> = state
        .slots
        .iter()
        .filter_map(|(key, slot)| slot.loading().map(|(made, priority)| (key, made, priority)))
        .collect();
    handed.sort_by_key(|&(_, _, priority)| priority);
    for (key, made, _) in handed {
        if work.spent() > 0 && work.expired() {
            break;
        }
        let Some(output) = state
            .slots
            .get_mut(&key)
            .and_then(TileSlot::work_mut)
            .and_then(|tile| landed_tile(tile.poll(asking.records)))
        else {
            continue;
        };
        match output {
            Ok(tile) if made == generation && target.contains(&key) && tile.key() == key => {
                if !work.spend(2 * TILE_ARRAY_BYTES) {
                    break;
                }
                state.slot_event(key, SlotEvent::Done);
                let stamps = state.reaching(key);
                state.store.arrive(tile, stamps, materials);
            }
            Ok(_) => state.slot_event(key, SlotEvent::Done),
            Err(error) => {
                state.slot_event(key, SlotEvent::Failed);
                state.failures.record(frame, error);
            }
        }
    }
    for &(key, priority) in missing {
        let held = state.slots.get(&key);
        if held.and_then(TileSlot::loading).is_some() || !state.store.loads(&key) {
            continue;
        }
        let job = state.tile_job(recipe.hash(), key);
        let input = job.identity();
        let record = RecordKey::of(&TILE_BUILD, &input);
        if held.and_then(TileSlot::failed) == Some(record) {
            continue;
        }
        let fetched = fetch_tile(asking.jobs, &input, priority);
        state.slot_event(
            key,
            SlotEvent::Asked {
                work: fetched,
                record,
                made: generation,
                priority,
            },
        );
    }
    prefetch(state, (missing, named), (recipe, generation, materials), asking, frame);
}

fn prefetch(
    state: &mut TerrainState,
    (missing, named): (&[(TileKey, u32)], &[TileKey]),
    (recipe, generation, materials): (RecordKey, Generation, TileMaterials),
    asking: &mut Asking<'_, '_, '_>,
    frame: u64,
) {
    let idle: Vec<(TileKey, Generation)> = state
        .slots
        .iter()
        .filter_map(|(key, slot)| slot.prefetching().map(|made| (key, made)))
        .collect();
    for (key, made) in idle {
        let output = state
            .slots
            .get_mut(&key)
            .and_then(TileSlot::work_mut)
            .and_then(|tile| landed_tile(tile.poll(asking.records)));
        if let Some(output) = output {
            state.slot_event(key, SlotEvent::Done);
            if let Ok(tile) = output
                && made == generation
                && tile.key() == key
                && !state.store.contains(&key)
            {
                let stamps = state.reaching(key);
                state.store.keep_unseen(tile, stamps, (frame, materials));
            }
        } else if made != generation || !named.contains(&key) {
            state.slot_event(key, SlotEvent::Done);
        }
    }
    if !missing.is_empty() || state.loading() > 0 {
        return;
    }
    for (place, &key) in named.iter().enumerate() {
        if state.slots.contains_key(&key)
            || state.store.contains(&key)
            || state.store.in_memory(&key)
        {
            continue;
        }
        let priority = job_priority(IDLE_JOBS, place as u32);
        let job = state.tile_job(recipe.hash(), key);
        let work = fetch_tile(asking.jobs, &job.identity(), priority);
        state.slot_event(
            key,
            SlotEvent::Prefetched {
                work,
                made: generation,
            },
        );
    }
}

fn fetch_tile(jobs: &mut Jobs<'_, '_>, input: &[u8], priority: u32) -> TileWork {
    jobs.fetch(&TILE_BUILD, input, priority, |bytes| {
        crate::jobs::tile_output(&bytes)
            .map(Arc::new)
            .map_err(|error| TILE_BUILD.undecoded(error))
    })
}

fn landed_tile(landed: Landed<'_, Arc<Tile>>) -> Option<Result<Arc<Tile>, String>> {
    match landed {
        Landed::Waiting => None,
        Landed::Ready(tile) => Some(Ok(Arc::clone(tile))),
        Landed::Failed(error) => Some(Err(format!("{}: {error}", TILE_BUILD.name()))),
    }
}

pub(crate) struct SharedRecipe {
    pub(crate) key: RecordKey,
    pub(crate) stack: Arc<Layers>,
    materials: TileMaterials,
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
        materials: state.materials == TileMaterials::Baked,
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
