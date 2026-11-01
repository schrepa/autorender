use super::*;

pub(super) const URGENT_JOBS: u8 = 0;
pub(super) const VIEW_JOBS: u8 = 2;
pub(super) const IDLE_JOBS: u8 = 3;

pub(super) const fn job_priority(class: u8, place: u32) -> u32 {
    ((class as u32) << 24) | if place > 0xff_ffff { 0xff_ffff } else { place }
}

pub(super) fn content_hash(bytes: &[u8]) -> u128 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    let mix = |mut k: u64| {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^ (k >> 33)
    };
    let word = |bytes: &[u8]| {
        let mut word = [0_u8; 8];
        word[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(word)
    };
    let (mut h1, mut h2) = (0_u64, 0_u64);
    let (blocks, tail) = bytes.as_chunks::<16>();
    for block in blocks {
        let (k1, k2) = (word(&block[..8]), word(&block[8..]));
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 = h1
            .rotate_left(27)
            .wrapping_add(h2)
            .wrapping_mul(5)
            .wrapping_add(0x52dc_e729);
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 = h2
            .rotate_left(31)
            .wrapping_add(h1)
            .wrapping_mul(5)
            .wrapping_add(0x3849_5ab5);
    }
    if tail.len() > 8 {
        h2 ^= word(&tail[8..])
            .wrapping_mul(C2)
            .rotate_left(33)
            .wrapping_mul(C1);
    }
    if !tail.is_empty() {
        h1 ^= word(&tail[..tail.len().min(8)])
            .wrapping_mul(C1)
            .rotate_left(31)
            .wrapping_mul(C2);
    }
    let len = bytes.len() as u64;
    h1 ^= len;
    h2 ^= len;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = mix(h1);
    h2 = mix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    (u128::from(h2) << 64) | u128::from(h1)
}

pub(super) fn job_output<T>(
    kind: &str,
    output: Result<Arc<[u8]>, Error>,
    decode: impl FnOnce(&[u8]) -> Result<T, Error>,
) -> Result<T, String> {
    let bytes = output.map_err(|error| format!("{kind}: {error}"))?;
    decode(&bytes).map_err(|error| format!("{kind}: its output does not decode: {error}"))
}

pub(super) struct Asking<'a, 'w, 's> {
    pub(super) jobs: &'a mut Jobs<'w, 's>,
    pub(super) records: &'a Records,
    pub(super) drawn: Option<engine_places::terrain::DrawRivers>,
}

impl Asking<'_, '_, '_> {
    pub(super) fn take(&mut self, ticket: &Ticket) -> Option<Result<Arc<[u8]>, Error>> {
        self.records.get(ticket)
    }
}

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
        match job_output("terrain::tile", output, crate::jobs::tile_output) {
            Ok(tile) if made == generation && target.contains(&key) && tile.key() == key => {
                work.upload(2 * TILE_ARRAY_BYTES);
                state.failed.remove(&key);
                state.store.arrive(Arc::new(tile));
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
        let job = state.tile_job(recipe, key);
        let ticket = asking.jobs.fetch(&TILE_BUILD, &job.identity(), priority);
        state.dispatched.insert(key, (ticket, generation, priority));
    }
    prefetch(state, (missing, named), (recipe, generation), asking, frame);
}

fn prefetch(
    state: &mut TerrainState,
    (missing, named): (&[(TileKey, u32)], &[TileKey]),
    (recipe, generation): ([u8; 32], Generation),
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
            if let Ok(tile) = job_output("terrain::tile", output, crate::jobs::tile_output)
                && made == generation
                && tile.key() == key
                && !state.store.contains(&key)
            {
                state.store.keep_in_memory(Arc::new(tile), frame);
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
        let job = state.tile_job(recipe, key);
        let ticket = asking.jobs.fetch(&TILE_BUILD, &job.identity(), priority);
        state.prefetched.insert(key, (ticket, generation));
    }
}

pub(crate) struct SharedRecipe {
    pub(crate) key: [u8; 32],
    pub(crate) stack: Arc<Layers>,
    materials: bool,
}

pub(crate) fn share_recipe(
    state: &mut TerrainState,
    jobs: &mut Jobs<'_, '_>,
    stacks: &mut Vec<(Arc<Layers>, [u8; 32])>,
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
            let key = jobs.share(crate::jobs::share_stack(&stack)).hash();
            stacks.push((stack.clone(), key));
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
    let key = jobs.share(crate::jobs::share_recipe(&recipe)).hash();
    state.recipe = Some(SharedRecipe {
        key,
        stack,
        materials: state.materials,
    });
}
