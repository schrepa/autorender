//! Loading tiles in place: a frame's work time of generating, decoding and
//! baking.

use super::*;

/// A frame's terrain work: slices of samples' worth handed
/// out while its time lasts, or a single slice without a clock.
pub(super) struct Work {
    pub(super) slice: u64,
    pub(super) clock: Option<fn() -> f64>,
    pub(super) end: f64,
    /// Bytes tiles may still bring the GPU this frame.
    pub(super) upload: std::cell::Cell<u64>,
    /// The most bands a tile under the view splits into.
    pub(super) bands: usize,
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

/// Loads `missing` tiles of `state`, most important first, within `work`'s
/// time from slices of `budget`: generated after the terrain's bakes,
/// decoded, or requested and decoded once delivered; `target` is every tile
/// it loads and `keep` every tile it keeps. Frames load what they find
/// missing, and hosts what is left between frames. Where
/// `slots` has workers, they bake and generate instead, each
/// tile at its job priority. Returns whether a bake is still
/// in progress, which nothing loads behind.
#[expect(clippy::too_many_arguments, reason = "the frame's disjoint borrows")]
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(super) fn load_tiles(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep): (&HashSet<TileKey>, &HashSet<TileKey>),
    work: &Work,
    budget: &mut u64,
    frame: u64,
    requests: &mut Vec<TileRequest>,
    mut slots: Option<&mut JobBudget<'_>>,
) -> bool {
    let pyramid = state.terrain.pyramid;
    match state.terrain.source.clone() {
        TerrainSource::Generated(authored) => {
            // Bakes run before any tile loads,
            // each reading the layers below it, in stack order: on a
            // worker where workers run, unless one failed
            // there.
            let layers = state.baked.clone().unwrap_or(authored);
            if let Some(unbaked) = layers.unbaked() {
                if let Some(slots) = slots.as_deref_mut()
                    && slots.jobs.workers() > 0
                    && !state.bake_inline
                {
                    bake_on_worker(state, slots.jobs, &layers, &pyramid);
                    return true;
                }
                // A bake handed to workers that stopped runs here.
                if let Some(id) = state.bake_job.take()
                    && let Some(slots) = slots.as_deref_mut()
                {
                    slots.jobs.cancel(id);
                }
                let bake = state
                    .bake
                    .get_or_insert_with(|| TerrainBake::new(&layers, unbaked.clone(), &pyramid));
                let _profile = engine_profile::span("cpu", "terrain_bake");
                let baked = loop {
                    if let Some(baked) = bake.advance(&pyramid, budget) {
                        break Some(baked);
                    }
                    if !work.refill(budget) {
                        break None;
                    }
                };
                if let Some(baked) = baked {
                    match &baked {
                        Baked::Hydrology(found) => state.hydrology = Some(found.stats),
                        Baked::Erosion(found) => state.erosion = Some(found.stats),
                        Baked::Path(_) => {}
                    }
                    let index = unbaked.0;
                    finish_bake(state, Arc::new(layers.with_baked(index, baked)));
                }
                return true;
            }
            // Edits stand over every bake.
            let layers = state.generation(layers);
            // Workers generate tiles where they run, unless
            // they failed it again and again.
            if let Some(slots) = slots.as_deref_mut()
                && slots.jobs.workers() > 0
                && state.recipe.is_some()
                && state.failures.count < WORKER_FAILURES
            {
                generate_on_workers(state, missing, (target, keep), work, slots, frame);
                return false;
            }
            // Tiles handed to workers that stopped generate here, whole.
            for (key, (id, _, _)) in state.dispatched.drain() {
                if let Some(slots) = slots.as_deref_mut() {
                    slots.jobs.cancel(id);
                }
                state.failed.remove(&key);
            }
            for (key, (id, _, _)) in state.far_jobs.drain() {
                if let Some(slots) = slots.as_deref_mut() {
                    slots.jobs.cancel(id);
                }
                state.far_wanted.insert(key, 0);
            }
            let banded = state
                .banded
                .drain()
                .flat_map(|(_, banding)| banding.jobs.into_iter().map(|(id, _)| id));
            let placing = state
                .placements
                .drain()
                .flat_map(|(_, placements)| placements.jobs.into_iter().map(|(id, _)| id));
            for id in banded.chain(placing).collect::<Vec<_>>() {
                if let Some(slots) = slots.as_deref_mut() {
                    slots.jobs.cancel(id);
                }
            }
            for key in std::mem::take(&mut state.far_wanted).into_keys() {
                state.store.leave(&key);
            }
            // The coarsest level always draws: generate it whole at once.
            let first = state.store.is_empty();
            for &(key, priority) in missing {
                let whole = first && u32::from(key.level) + 1 == pyramid.levels();
                if !whole && *budget == 0 && !work.refill(budget) {
                    break;
                }
                let rules = &state.terrain.surface.rules;
                // Tiles not resident hand out their shape first.
                let staged = !state.store.contains(&key);
                let materials = state.materials;
                let job = state.jobs.entry(key).or_insert_with(|| {
                    Generator::new(layers.clone(), &pyramid, key)
                        .with_surface(rules.clone())
                        .with_staged(staged)
                        .with_materials(materials)
                });
                let tile = if whole {
                    let mut unlimited = u64::MAX;
                    job.advance(&mut unlimited)
                } else {
                    loop {
                        if let Some(tile) = job.advance(budget) {
                            break Some(tile);
                        }
                        if !work.refill(budget) {
                            break None;
                        }
                    }
                };
                if let Some(tile) = tile {
                    state.jobs.remove(&key);
                    state.arrive(tile, priority, None);
                } else if let Some(shape) = state.jobs.get_mut(&key).and_then(Generator::take_shape)
                {
                    state.shape(shape);
                }
            }
        }
        TerrainSource::File {
            header,
            url: _,
            data: Some(data),
        } => {
            // The whole file is here: decode within the frame's work,
            // then give each tile its surface.
            for &(key, _) in missing {
                if *budget == 0 && !work.refill(budget) {
                    break;
                }
                if state.surface_job(key, budget) {
                    continue;
                }
                let Some(entry) = header.tile(key) else {
                    continue;
                };
                let range = usize::try_from(entry.offset)
                    .ok()
                    .and_then(|start| data.get(start..start.checked_add(entry.length as usize)?));
                *budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
                match range.map(|bytes| decode_tile(&header, key, bytes)) {
                    Some(Ok(tile)) => state.resurface(tile, budget),
                    _ => {
                        state.failed.insert(key, frame);
                    }
                }
            }
        }
        TerrainSource::File {
            header,
            url,
            data: None,
        } => {
            for &(key, _) in missing {
                if *budget > 0 || work.refill(budget) {
                    state.surface_job(key, budget);
                }
                if state.in_flight.contains(&key)
                    || state.jobs.contains_key(&key)
                    || (state.store.contains(&key)
                        && !matches!(state.store.phase(&key), Some(Phase::Shape | Phase::Placing)))
                    || state
                        .failed
                        .get(&key)
                        .is_some_and(|&at| frame < at + RETRY_AFTER)
                {
                    continue;
                }
                let Some(entry) = header.tile(key) else {
                    continue;
                };
                state.in_flight.insert(key);
                requests.push(TileRequest {
                    terrain: state.entity,
                    key,
                    url: url.clone(),
                    offset: entry.offset,
                    length: entry.length,
                });
            }
            let arrivals = std::mem::take(&mut state.arrivals);
            let mut later = Vec::new();
            for (key, bytes) in arrivals {
                if *budget == 0 {
                    work.refill(budget);
                }
                if *budget == 0 || !target.contains(&key) {
                    if target.contains(&key) {
                        later.push((key, bytes));
                    }
                    continue;
                }
                *budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
                match bytes.map(|bytes| decode_tile(&header, key, &bytes)) {
                    Some(Ok(tile)) => {
                        state.failed.remove(&key);
                        state.resurface(tile, budget);
                    }
                    _ => {
                        state.failed.insert(key, frame);
                    }
                }
            }
            state.arrivals = later;
        }
    }
    false
}
