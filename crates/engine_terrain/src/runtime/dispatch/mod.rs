use super::*;
use engine_core::{FrameBudget, Landed, RecordKey};

const BUDGET: &str = "terrain";
const BAKED_STACK: u8 = 1;

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
    (frame, gpu): (u64, bool),
    asking: &mut Asking<'_, '_, '_>,
) -> bool {
    let authored = state.terrain.layers.clone();
    if bake(state, &authored, asking, frame) {
        if let Some(ahead) = ahead_recipe(state, gpu) {
            look_ahead(state, missing, (target, keep), (ahead, work), asking);
        }
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
    let left = |key: &TileKey, slot: &TileSlot| match (slot.loading(), slot.failed()) {
        (Some((made, _)), _) => !(made == generation && keep.contains(key) && store.loads(key)),
        (None, Some(_)) => !keep.contains(key),
        (None, None) => false,
    };
    let stale = state.slots.keys_where(left);
    for key in stale {
        state.slots.remove(&key);
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
                state.slots.remove(&key);
                let stamps = state.reaching(key);
                state.store.arrive(tile, stamps, materials);
            }
            Ok(_) => {
                state.slots.remove(&key);
            }
            Err(error) => state.failures.record(frame, error),
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
        let fetched = ask_tile(state, asking, key, (&input, priority));
        let slot = TileSlot::needed(fetched, (record, generation, priority));
        state.slots.insert(key, slot);
    }
    prefetch(
        state,
        (missing, named),
        (recipe, generation, materials),
        asking,
        frame,
    );
    if let Some(bake) = state.bake.as_mut() {
        bake.looked.clear();
    }
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
            state.slots.remove(&key);
            if let Ok(tile) = output
                && made == generation
                && tile.key() == key
                && !state.store.contains(&key)
            {
                let stamps = state.reaching(key);
                state.store.keep_unseen(tile, stamps, (frame, materials));
            }
        } else if made != generation || !named.contains(&key) {
            state.slots.remove(&key);
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
        let work = ask_tile(state, asking, key, (&job.identity(), priority));
        let slot = TileSlot::ahead(work, generation);
        state.slots.insert(key, slot);
    }
}

fn ahead_recipe(state: &mut TerrainState, gpu: bool) -> Option<(RecordKey, TileMaterials)> {
    if state.around.is_some() || state.recipe.is_some() || state.baked.is_some() {
        return None;
    }
    let bake = state.bake.as_ref().filter(|bake| bake.work.waiting())?;
    if let Some(ahead) = bake.ahead {
        return Some(ahead);
    }
    let materials = if gpu {
        TileMaterials::Program
    } else {
        TileMaterials::Baked
    };
    let recipe = recipe_of(state, baked_stack(bake.key?), (None, materials));
    let ahead = (RecordKey::shared(&crate::jobs::share_recipe(&recipe)), materials);
    state.bake.as_mut()?.ahead = Some(ahead);
    Some(ahead)
}

fn look_ahead(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep): (&HashSet<TileKey>, &HashSet<TileKey>),
    ((recipe, materials), work): ((RecordKey, TileMaterials), &Work<'_>),
    asking: &mut Asking<'_, '_, '_>,
) {
    state.looked_from(state.generation());
    let Some(mut looked) = state.bake.as_mut().map(|bake| std::mem::take(&mut bake.looked)) else {
        return;
    };
    looked.retain(|key, _| keep.contains(key) && state.store.loads(key));
    let mut order: Vec<(u32, TileKey)> = looked
        .iter()
        .map(|(key, (_, priority, _))| (*priority, *key))
        .collect();
    order.sort_unstable();
    for (_, key) in order {
        if work.spent() > 0 && work.expired() {
            break;
        }
        let Some((input, _, slot)) = looked.get_mut(&key) else {
            continue;
        };
        let Some(look) = slot.as_mut() else {
            continue;
        };
        let tile = match look.poll(asking.records) {
            Landed::Waiting => continue,
            Landed::Failed(_) => {
                *slot = None;
                continue;
            }
            Landed::Ready(tile) => Arc::clone(tile),
        };
        let placed = *input == state.tile_job(recipe.hash(), key).identity()
            && target.contains(&key)
            && tile.key() == key;
        if placed && !work.spend(2 * TILE_ARRAY_BYTES) {
            break;
        }
        looked.remove(&key);
        if placed {
            let stamps = state.reaching(key);
            state.store.arrive(tile, stamps, materials);
        }
    }
    for &(key, priority) in missing {
        if !state.store.loads(&key) {
            continue;
        }
        let input = state.tile_job(recipe.hash(), key).identity();
        if looked.get(&key).is_some_and(|(asked, _, _)| *asked == input) {
            continue;
        }
        let look = asking.jobs.look(&TILE_BUILD, &input, priority, decode_tile);
        looked.insert(key, (input, priority, Some(look)));
    }
    if let Some(bake) = state.bake.as_mut() {
        bake.looked = looked;
    }
}

fn ask_tile(
    state: &mut TerrainState,
    asking: &mut Asking<'_, '_, '_>,
    key: TileKey,
    (input, priority): (&[u8], u32),
) -> TileWork {
    let looked = state
        .bake
        .as_mut()
        .and_then(|bake| bake.looked.remove(&key));
    let Some((asked, _, Some(mut look))) = looked.filter(|(asked, _, _)| asked == input) else {
        return fetch_tile(asking.jobs, input, priority);
    };
    let landed = match look.poll(asking.records) {
        Landed::Ready(_) => Some(true),
        Landed::Failed(_) => Some(false),
        Landed::Waiting => None,
    };
    match landed {
        Some(true) => look,
        Some(false) => {
            drop(look);
            fetch_tile(asking.jobs, &asked, priority)
        }
        None => {
            let work = fetch_tile(asking.jobs, &asked, priority);
            drop(look);
            work
        }
    }
}

fn fetch_tile(jobs: &mut Jobs<'_, '_>, input: &[u8], priority: u32) -> TileWork {
    jobs.fetch(&TILE_BUILD, input, priority, decode_tile)
}

fn decode_tile(bytes: Arc<[u8]>) -> Result<Arc<Tile>, engine_core::Error> {
    crate::jobs::tile_output(&bytes)
        .map(Arc::new)
        .map_err(|error| TILE_BUILD.undecoded(error))
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

fn baked_stack(bake: RecordKey) -> RecordKey {
    RecordKey::shared_tagged(BAKED_STACK, &bake.hash())
}

fn recipe_of(
    state: &TerrainState,
    stack: RecordKey,
    (view, materials): (Option<StackView>, TileMaterials),
) -> TerrainRecipe {
    TerrainRecipe {
        stack: stack.hash(),
        view,
        pyramid: state.terrain.pyramid,
        surface: state.terrain.surface.rules.authored().clone(),
        materials: materials == TileMaterials::Baked,
    }
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
    let (stack, view, baked) = match &state.around {
        Some(around) => (
            around.base.clone(),
            Some(StackView {
                offset: -around.cutout.0,
                low: Vec2::ZERO,
                high: around.cutout.1 - around.cutout.0,
                blend: around.surroundings.blend,
            }),
            around.base_key,
        ),
        None => (layers, None, state.baked.as_ref().and(state.baked_key)),
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
            let bytes = crate::jobs::share_stack(&stack);
            let key = match baked {
                Some(baked) => {
                    let key = baked_stack(baked);
                    jobs.share_as(key, bytes);
                    key
                }
                None => jobs.share(bytes),
            };
            stacks.push((stack.clone(), key));
            key
        }
    };
    let recipe = recipe_of(state, stack_key, (view, state.materials));
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
