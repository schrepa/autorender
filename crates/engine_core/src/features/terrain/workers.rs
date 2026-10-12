//! Loading tiles on workers: the recipe shared with them, the jobs handed
//! out by priority and the tiles they finish.

use super::*;

/// Workers a frame hands terrain jobs to: the jobs, and how
/// many more it may hand out, of any priority and urgent.
pub(super) struct JobSlots<'a> {
    pub(super) jobs: &'a mut Jobs,
    slots: usize,
    urgent: usize,
}

impl JobSlots<'_> {
    /// The frame's jobs: two in flight per worker, so each has its next
    /// ready, and a third for urgent jobs, which workers take up first;
    /// less those still running.
    pub(super) fn of(jobs: &mut Jobs) -> JobSlots<'_> {
        let workers = jobs.workers() as usize;
        let running = jobs.running();
        JobSlots {
            slots: (workers * 2).saturating_sub(running),
            urgent: (workers * 3).saturating_sub(running),
            jobs,
        }
    }

    /// Takes a slot for a job of `priority`; whether one was free.
    fn take(&mut self, priority: u32) -> bool {
        let free = if job_class(priority) == URGENT_JOBS {
            self.urgent
        } else {
            self.slots
        };
        if free == 0 {
            return false;
        }
        self.slots = self.slots.saturating_sub(1);
        self.urgent = self.urgent.saturating_sub(1);
        true
    }

    /// Gives back the slot of a job cancelled while running.
    fn release(&mut self) {
        self.slots += 1;
        self.urgent += 1;
    }
}

/// A terrain's recipe as workers hold it: its key, the
/// stack it shares (the layers its tiles generate from, which surroundings
/// see from afar) and its revision, which its tiles' jobs carry.
pub(super) struct SharedRecipe {
    pub(super) key: String,
    pub(super) stack: Arc<Layers>,
    revision: u64,
    materials: bool,
}

/// Generates `state`'s missing tiles on workers, most
/// important first, while the frame may hand out jobs; makes the tiles
/// workers finished resident, the first at once and the rest within
/// `work`'s time, when still wanted and generated from the layers tiles
/// generate from now. Tiles that failed are asked for again later. Jobs
/// for tiles `keep` no longer holds, or from layers since replaced, stop,
/// and tiles of levels with objects standing far away
/// arrive without them: jobs of their own place those after, merged into
/// the tile in the order one pass places them.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(super) fn generate_on_workers(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep): (&HashSet<TileKey>, &HashSet<TileKey>),
    work: &Work,
    slots: &mut JobSlots<'_>,
    frame: u64,
) {
    let Some((recipe, revision)) = state
        .recipe
        .as_ref()
        .map(|recipe| (recipe.key.clone(), recipe.revision))
    else {
        return;
    };
    // Tiles generating here stop: workers generate them.
    state.jobs.clear();
    // Work no longer wanted stops, so its workers take up what is.
    state.dispatched.retain(|key, &mut (id, made, _)| {
        let wanted = made == revision && keep.contains(key);
        if !wanted {
            slots.jobs.cancel(id);
        }
        wanted
    });
    let tiles = &state.tiles;
    state.far_jobs.retain(|key, (id, tile)| {
        let wanted =
            keep.contains(key) && tiles.get(key).is_some_and(|held| Arc::ptr_eq(held, tile));
        if !wanted {
            slots.jobs.cancel(*id);
        }
        wanted
    });
    state
        .far_wanted
        .retain(|key, _| keep.contains(key) && tiles.contains_key(key));
    state.banded.retain(|key, banding| {
        let wanted = banding.made == revision && keep.contains(key);
        if !wanted {
            for &(id, ref band) in &banding.jobs {
                if band.is_none() {
                    slots.jobs.cancel(id);
                }
            }
        }
        wanted
    });
    state.placements.retain(|key, placements| {
        let wanted = placements.made == revision && keep.contains(key) && tiles.contains_key(key);
        if !wanted {
            for &(id, ref objects) in &placements.jobs {
                if objects.is_none() {
                    slots.jobs.cancel(id);
                }
            }
        }
        wanted
    });
    let handed: Vec<(TileKey, (JobId, u64, u32))> = state
        .dispatched
        .iter()
        .map(|(&key, &job)| (key, job))
        .collect();
    let rules = state.terrain.surface.rules.clone();
    let mut inserted = false;
    for (key, (id, made, priority)) in handed {
        if inserted && (work.expired() || !work.uploads()) {
            break;
        }
        let Some(output) = slots.jobs.take(id) else {
            continue;
        };
        state.dispatched.remove(&key);
        let tile = output
            .ok()
            .and_then(|bytes| engine_terrain::jobs::tile_output(&bytes).ok());
        match tile {
            Some(tile) if made == revision && target.contains(&key) && tile.key() == key => {
                // After its shape a tile uploads its control values alone.
                work.upload(if state.shaped.contains(&key) {
                    TILE_ARRAY_BYTES
                } else {
                    2 * TILE_ARRAY_BYTES
                });
                state.failed.remove(&key);
                state.arrive(tile, priority, Some(&mut *slots.jobs));
                inserted = true;
            }
            Some(_) => {}
            None => {
                state.failed.insert(key, frame);
                state.worker_failures += 1;
            }
        }
    }
    // Tiles generating in bands join once every band is
    // done: resident at once with their materials, their objects placed by
    // a job after.
    let joining: Vec<TileKey> = state.banded.keys().copied().collect();
    for key in joining {
        if inserted && (work.expired() || !work.uploads()) {
            break;
        }
        let Some(banding) = state.banded.get_mut(&key) else {
            continue;
        };
        let mut failed = false;
        for (id, band) in &mut banding.jobs {
            if band.is_none()
                && let Some(output) = slots.jobs.take(*id)
            {
                match output
                    .ok()
                    .and_then(|bytes| engine_terrain::jobs::band_output(&bytes).ok())
                {
                    Some(done) => *band = Some(done),
                    None => failed = true,
                }
            }
        }
        if !failed && banding.jobs.iter().any(|(_, band)| band.is_none()) {
            continue;
        }
        let Some(banding) = state.banded.remove(&key) else {
            continue;
        };
        let (mut bands, mut running) = (Vec::new(), Vec::new());
        for (id, band) in banding.jobs {
            match band {
                Some(band) => bands.push(band),
                None => running.push(id),
            }
        }
        let spacing = state.terrain.pyramid.spacing(u32::from(key.level));
        let tile = (!failed)
            .then(|| engine_terrain::merge_bands(key, spacing, &bands))
            .flatten();
        let Some(tile) = tile else {
            for id in running {
                slots.jobs.cancel(id);
                slots.release();
            }
            state.failed.insert(key, frame);
            state.worker_failures += 1;
            continue;
        };
        if banding.made != revision || !target.contains(&key) {
            continue;
        }
        work.upload(2 * TILE_ARRAY_BYTES);
        state.banded_tiles += 1;
        state.stale.remove(&key);
        state.failed.remove(&key);
        if rules.places(key.level, banding.placing) {
            // Its objects in bands of candidate rows across workers too,
            // joined in order.
            let rows = rules.place_rows(key, spacing, banding.placing);
            let starts: Vec<(u32, u32)> = rows
                .iter()
                .enumerate()
                .flat_map(|(layer, &rows)| (0..rows).map(move |row| (layer as u32, row)))
                .collect();
            let bands = work.bands.min(starts.len()).max(1);
            let mut jobs = Vec::with_capacity(bands);
            for band in 0..bands {
                let start = if band == 0 {
                    (0, 0)
                } else {
                    starts[band * starts.len() / bands]
                };
                let end = (band + 1 < bands).then(|| starts[(band + 1) * starts.len() / bands]);
                let input = engine_terrain::jobs::place_input(
                    &recipe,
                    &tile,
                    banding.placing,
                    (start, end),
                );
                let identity = engine_terrain::jobs::place_identity(
                    &recipe,
                    key,
                    banding.placing,
                    (start, end),
                );
                let id = slots
                    .jobs
                    .submit_cacheable(PLACE_JOB, input, banding.priority, identity);
                jobs.push((id, None));
            }
            state.placements.insert(
                key,
                Placements {
                    made: revision,
                    priority: banding.priority,
                    placing: banding.placing,
                    jobs,
                },
            );
            state.shaped.insert(key);
            state.insert(tile);
        } else {
            state.arrive(tile, banding.priority, Some(&mut *slots.jobs));
        }
        inserted = true;
    }
    // Then their objects, once every band of them is done,
    // joined in order.
    let placing_keys: Vec<TileKey> = state.placements.keys().copied().collect();
    for key in placing_keys {
        if inserted && work.expired() {
            break;
        }
        let Some(placements) = state.placements.get_mut(&key) else {
            continue;
        };
        let mut failed = false;
        for (id, objects) in &mut placements.jobs {
            if objects.is_none()
                && let Some(output) = slots.jobs.take(*id)
            {
                match output
                    .ok()
                    .and_then(|bytes| engine_terrain::jobs::far_output(&bytes).ok())
                {
                    Some(placed) => *objects = Some(placed),
                    None => failed = true,
                }
            }
        }
        if !failed && placements.jobs.iter().any(|(_, objects)| objects.is_none()) {
            continue;
        }
        let Some(placements) = state.placements.remove(&key) else {
            continue;
        };
        if failed {
            for (id, objects) in placements.jobs {
                if objects.is_none() {
                    slots.jobs.cancel(id);
                    slots.release();
                }
            }
            state.failed.insert(key, frame);
            state.worker_failures += 1;
            continue;
        }
        let Some(tile) = state.tiles.get(&key).cloned() else {
            continue;
        };
        if placements.made != revision || !target.contains(&key) {
            continue;
        }
        let objects: Vec<engine_terrain::Placed> = placements
            .jobs
            .into_iter()
            .flat_map(|(_, objects)| objects.unwrap_or_default())
            .collect();
        let tile = Tile::clone(&tile).with_placed(objects, placements.placing);
        state.arrive(tile, placements.priority, Some(&mut *slots.jobs));
        inserted = true;
    }
    // Tiles still generating hand out their shape first: it
    // draws, with coarser levels' materials, and collides until the whole
    // tile arrives.
    let shaping: Vec<(TileKey, JobId, u64)> = state
        .dispatched
        .iter()
        .map(|(&key, &(id, made, _))| (key, id, made))
        .collect();
    for (key, id, made) in shaping {
        if inserted && (work.expired() || !work.uploads()) {
            break;
        }
        let Some(part) = slots.jobs.take_part(id) else {
            continue;
        };
        if let Ok(shape) = engine_terrain::jobs::tile_output(&part)
            && made == revision
            && target.contains(&key)
            && shape.key() == key
            && state.shape(shape)
        {
            // Its heights, and its ancestor's control values for cover.
            work.upload(2 * TILE_ARRAY_BYTES);
            inserted = true;
        }
    }
    let placing: Vec<(TileKey, JobId)> = state
        .far_jobs
        .iter()
        .map(|(&key, &(id, _))| (key, id))
        .collect();
    for (key, id) in placing {
        if inserted && work.expired() {
            break;
        }
        let Some(output) = slots.jobs.take(id) else {
            continue;
        };
        let Some((_, tile)) = state.far_jobs.remove(&key) else {
            continue;
        };
        match output
            .ok()
            .and_then(|bytes| engine_terrain::jobs::far_output(&bytes).ok())
        {
            Some(far) => {
                let placed = merge_placed(tile.placed(), &far);
                state.insert(Tile::clone(&tile).with_placed(placed, Placing::All));
                inserted = true;
            }
            None => {
                state.far_wanted.insert(key, job_priority(VIEW_JOBS, 0));
                state.worker_failures += 1;
            }
        }
    }
    // Tiles generated ahead of need go to memory; with tiles
    // missing, or out of the idle reach, their jobs stop.
    let idle: Vec<(TileKey, JobId, u64)> = state
        .idle
        .iter()
        .map(|(&key, &(id, made))| (key, id, made))
        .collect();
    for (key, id, made) in idle {
        if let Some(output) = slots.jobs.take(id) {
            state.idle.remove(&key);
            if let Some(tile) = output
                .ok()
                .and_then(|bytes| engine_terrain::jobs::tile_output(&bytes).ok())
                && made == revision
                && tile.key() == key
                && !state.tiles.contains_key(&key)
            {
                state.cache_tile(key, Arc::new(tile), frame);
            }
        } else if !missing.is_empty() || made != revision || !state.ahead.contains(&key) {
            slots.jobs.cancel(id);
            slots.release();
            state.idle.remove(&key);
        }
    }
    // Then what is missing, most urgent first: tiles, and the objects
    // standing far away on tiles that arrived without them, ranked after
    // the urgent tiles where their tiles ranked.
    let mut wanted: Vec<(u32, TileKey, bool)> = missing
        .iter()
        .filter(|(key, _)| {
            !state.dispatched.contains_key(key)
                && !state.banded.contains_key(key)
                && !state.placements.contains_key(key)
                && !state
                    .failed
                    .get(key)
                    .is_some_and(|&at| frame < at + RETRY_AFTER)
        })
        .map(|&(key, priority)| (priority, key, false))
        .collect();
    wanted.extend(
        state
            .far_wanted
            .iter()
            .filter(|(key, _)| !state.far_jobs.contains_key(key))
            .map(|(&key, &priority)| (job_priority(VIEW_JOBS, priority & 0xff_ffff), key, true)),
    );
    wanted.sort_by_key(|&(priority, key, far)| (priority, far, key));
    // Tiles under the view split into bands of rows across the slots the
    // urgent tiles left share: each arrives sooner, though
    // workers do a little more in all.
    let mut urgent_left = wanted
        .iter()
        .filter(|&&(priority, _, far)| !far && job_class(priority) == URGENT_JOBS)
        .count();
    for (priority, key, far) in wanted {
        if !slots.take(priority) {
            break;
        }
        if far {
            let Some(tile) = state.tiles.get(&key).cloned() else {
                continue;
            };
            let input = engine_terrain::jobs::far_input(&recipe, &tile);
            let identity = engine_terrain::jobs::far_identity(&recipe, key);
            let id = slots
                .jobs
                .submit_cacheable(FAR_JOB, input, priority, identity);
            state.far_wanted.remove(&key);
            state.far_jobs.insert(key, (id, tile));
        } else {
            // A tile replacing one resident, as after an edit, comes whole,
            // so its objects standing far away never leave the view; one not
            // resident hands out its shape first.
            let resident = state.tiles.contains_key(&key);
            let placing = if rules.places(key.level, Placing::Far)
                && (!resident || state.shaped.contains(&key))
            {
                Placing::Near
            } else {
                Placing::All
            };
            let urgent = job_class(priority) == URGENT_JOBS;
            urgent_left = urgent_left.saturating_sub(usize::from(urgent));
            let bands = if urgent && !resident {
                std::cmp::min(1 + slots.urgent / (urgent_left + 1), work.bands)
            } else {
                1
            };
            if bands > 1 {
                let size = (TILE_STRIDE as usize).div_ceil(bands);
                let mut jobs = Vec::with_capacity(bands);
                for band in 0..bands {
                    if band > 0 && !slots.take(priority) {
                        break;
                    }
                    let end = std::cmp::min((band + 1) * size, TILE_STRIDE as usize);
                    let rows = (band * size) as u32..end as u32;
                    let input = engine_terrain::jobs::band_input(&recipe, key, rows);
                    let id = slots
                        .jobs
                        .submit_cacheable(BAND_JOB, input.clone(), priority, input);
                    jobs.push((id, None));
                }
                // Unless every band found a slot, the tile generates whole.
                let covered = jobs.len() * size;
                if covered < TILE_STRIDE as usize {
                    for (id, _) in jobs.drain(..) {
                        slots.jobs.cancel(id);
                        slots.release();
                    }
                } else {
                    state.banded.insert(
                        key,
                        Banding {
                            made: revision,
                            priority,
                            placing,
                            jobs,
                        },
                    );
                    continue;
                }
            }
            let input = engine_terrain::jobs::tile_input(&recipe, key, placing, !resident);
            // Staged or not, the same tile.
            let identity = engine_terrain::jobs::tile_input(&recipe, key, placing, false);
            let id = slots
                .jobs
                .submit_cacheable(TILE_JOB, input, priority, identity);
            state.dispatched.insert(key, (id, revision, priority));
        }
    }
    // With nothing else to hand out, a job a worker generates the nearest
    // tiles within the idle reach, whole, into memory.
    if missing.is_empty()
        && state.dispatched.is_empty()
        && state.banded.is_empty()
        && state.placements.is_empty()
        && state.far_wanted.is_empty()
        && state.far_jobs.is_empty()
    {
        let workers = slots.jobs.workers() as usize;
        for index in 0..state.ahead.len() {
            if state.idle.len() >= workers {
                break;
            }
            let key = state.ahead[index];
            if state.idle.contains_key(&key) {
                continue;
            }
            let priority = job_priority(IDLE_JOBS, index as u32);
            if !slots.take(priority) {
                break;
            }
            let input = engine_terrain::jobs::tile_input(&recipe, key, Placing::All, false);
            let id = slots
                .jobs
                .submit_cacheable(TILE_JOB, input.clone(), priority, input);
            state.idle.insert(key, (id, revision));
        }
    }
}

/// Shares the recipe workers generate `state`'s tiles from
/// once its bakes are done, and again whenever the layers its tiles
/// generate from change: their stack, shared once for every terrain seeing
/// it, and how its terrain sees it. Shares are named by their content,
/// so jobs reading them are the same bytes in every
/// session; `shares` counts recipes for their revisions, and the key of
/// the recipe replaced goes to `released`.
pub(super) fn share_recipe(
    state: &mut TerrainState,
    jobs: &mut Jobs,
    (stacks, shares): (&mut Vec<(Arc<Layers>, String)>, &mut u64),
    released: &mut Vec<String>,
) {
    let TerrainSource::Generated(authored) = &state.terrain.source else {
        return;
    };
    let layers = state.baked.clone().unwrap_or_else(|| authored.clone());
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
        None => (state.generation(layers), None),
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
    *shares += 1;
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
        revision: *shares,
        materials: state.materials,
    };
    if let Some(old) = state.recipe.replace(shared) {
        released.push(old.key);
    }
}

/// Pushes `key` onto `order` after those of its ancestors not yet there,
/// coarsest first: a tile draws only within its parent, so a
/// finer one loaded first would wait unused, while what stands on the
/// ground (blades, placed objects) already followed its finer heights.
pub(super) fn push_with_ancestors(
    pyramid: &engine_terrain::Pyramid,
    key: TileKey,
    order: &mut Vec<TileKey>,
    seen: &mut HashSet<TileKey>,
) {
    if seen.contains(&key) {
        return;
    }
    let mut chain = vec![key];
    let mut child = key;
    while let Some(parent) = pyramid.parent(child) {
        if seen.contains(&parent) {
            break;
        }
        chain.push(parent);
        child = parent;
    }
    for key in chain.into_iter().rev() {
        seen.insert(key);
        order.push(key);
    }
}
