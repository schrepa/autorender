//! Generating tiles on workers: the recipe shared with them, the jobs
//! handed out by priority within the job budget, and the tiles they finish.

use super::*;

/// A terrain's recipe as workers hold it: its key, the
/// stack it shares (the layers its tiles generate from, which surroundings
/// see from afar) and the generation its tiles' jobs make.
pub(crate) struct SharedRecipe {
    pub(crate) key: String,
    pub(crate) stack: Arc<Layers>,
    pub(crate) generation: Generation,
    materials: bool,
}

/// The host's workers generating a terrain's tiles from its shared recipe,
/// within `slots`, the frame's job budget: tiles workers finished become
/// resident when still wanted and generated from the layers tiles generate
/// from now, the first at once and the rest within `work`'s time; tiles
/// that failed are asked for again later. Jobs for tiles `keep` no longer
/// holds, or from layers since replaced, stop. Tiles under the view split
/// into bands of rows across the slots urgent tiles leave, and tiles of
/// levels with objects standing far away arrive without them: jobs of their
/// own place those after, merged into the tile in the order one pass
/// places them. With nothing else to do, workers generate tiles ahead of
/// need into memory.
pub(super) struct OnWorkers<'a, 'b> {
    slots: &'a mut JobBudget<'b>,
    work: &'a Work,
    target: &'a HashSet<TileKey>,
    keep: &'a HashSet<TileKey>,
    frame: u64,
    /// The recipe's key and generation, and the surface's rules.
    recipe: String,
    generation: Generation,
    rules: Arc<BiomeRules>,
    /// Whether a tile became resident this frame: the rest wait while
    /// the frame's time or uploads are spent.
    inserted: bool,
    /// Urgent tile tasks not yet started, which share the urgent slots
    /// left.
    urgent_left: usize,
}

impl<'a, 'b> OnWorkers<'a, 'b> {
    /// Workers generating `state`'s tiles, once its recipe is shared.
    pub(super) fn new(
        state: &TerrainState,
        (target, keep): (&'a HashSet<TileKey>, &'a HashSet<TileKey>),
        work: &'a Work,
        slots: &'a mut JobBudget<'b>,
        frame: u64,
    ) -> Option<Self> {
        let recipe = state.recipe.as_ref()?;
        Some(Self {
            recipe: recipe.key.clone(),
            generation: recipe.generation,
            rules: state.terrain.surface.rules.clone(),
            slots,
            work,
            target,
            keep,
            frame,
            inserted: false,
            urgent_left: 0,
        })
    }

    /// Whether more tiles may become resident this frame.
    fn arrives(&self) -> bool {
        !self.inserted || (!self.work.expired() && self.work.uploads())
    }

    /// Work no longer wanted stops, so its workers take up what is.
    fn stop_unwanted(&mut self, state: &mut TerrainState) {
        let (generation, keep, slots) = (self.generation, self.keep, &mut *self.slots);
        state.dispatched.retain(|key, &mut (id, made, _)| {
            let wanted = made == generation && keep.contains(key);
            if !wanted {
                slots.jobs.cancel(id);
            }
            wanted
        });
        let store = &state.store;
        state.far_jobs.retain(|key, &mut (id, _, arrival, _)| {
            let wanted = keep.contains(key) && store.arrival(key) == Some(arrival);
            if !wanted {
                slots.jobs.cancel(id);
            }
            wanted
        });
        state
            .far_wanted
            .retain(|key, _| keep.contains(key) && store.contains(key));
        state.banded.retain(|key, banding| {
            let wanted = banding.made == generation && keep.contains(key);
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
            let wanted = placements.made == generation && keep.contains(key) && store.contains(key);
            if !wanted {
                for &(id, ref objects) in &placements.jobs {
                    if objects.is_none() {
                        slots.jobs.cancel(id);
                    }
                }
            }
            wanted
        });
    }

    /// Tiles workers finished whole, most urgent first.
    fn take_tiles(&mut self, state: &mut TerrainState) {
        let mut handed: Vec<(TileKey, (JobId, Generation, u32))> = state
            .dispatched
            .iter()
            .map(|(key, &job)| (key, job))
            .collect();
        handed.sort_by_key(|&(_, (_, _, priority))| priority);
        for (key, (id, made, priority)) in handed {
            if !self.arrives() {
                break;
            }
            let Some(output) = self.slots.jobs.take(id) else {
                continue;
            };
            state.dispatched.remove(&key);
            match job_output(
                TileJob::KIND.name(),
                output,
                engine_terrain::jobs::tile_output,
            ) {
                Ok(tile)
                    if made == self.generation
                        && self.target.contains(&key)
                        && tile.key() == key =>
                {
                    // After its shape a tile uploads its control values alone.
                    self.work.upload(
                        if matches!(state.store.phase(&key), Some(Phase::Shape | Phase::Placing)) {
                            TILE_ARRAY_BYTES
                        } else {
                            2 * TILE_ARRAY_BYTES
                        },
                    );
                    state.failed.remove(&key);
                    state.arrive((tile, None), priority, Some(&mut *self.slots.jobs));
                    self.inserted = true;
                }
                Ok(_) => {}
                Err(error) => {
                    state.failed.insert(key, self.frame);
                    state.failures.record(self.frame, error);
                }
            }
        }
    }

    /// Tiles generating in bands join once every band is
    /// done, most urgent first: resident at once with their materials, their
    /// objects placed by a job after.
    fn join_bands(&mut self, state: &mut TerrainState) {
        let mut joining: Vec<(u32, TileKey)> = state
            .banded
            .iter()
            .map(|(key, banding)| (banding.priority, key))
            .collect();
        joining.sort_by_key(|&(priority, _)| priority);
        for (_, key) in joining {
            if !self.arrives() {
                break;
            }
            let Some(banding) = state.banded.get_mut(&key) else {
                continue;
            };
            let mut failed = None;
            for (id, band) in &mut banding.jobs {
                if band.is_none()
                    && let Some(output) = self.slots.jobs.take(*id)
                {
                    match job_output(
                        BandJob::KIND.name(),
                        output,
                        engine_terrain::jobs::band_output,
                    ) {
                        Ok(done) => *band = Some(done),
                        Err(error) => failed = Some(error),
                    }
                }
            }
            if failed.is_none() && banding.jobs.iter().any(|(_, band)| band.is_none()) {
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
            let tile = match failed {
                Some(error) => Err(error),
                None => engine_terrain::merge_bands(key, spacing, &bands).ok_or_else(|| {
                    format!("{}: the bands of {key:?} do not join", BandJob::KIND.name())
                }),
            };
            let tile = match tile {
                Ok(tile) => tile,
                Err(error) => {
                    for id in running {
                        self.slots.jobs.cancel(id);
                        self.slots.release();
                    }
                    state.failed.insert(key, self.frame);
                    state.failures.record(self.frame, error);
                    continue;
                }
            };
            if banding.made != self.generation || !self.target.contains(&key) {
                continue;
            }
            self.work.upload(2 * TILE_ARRAY_BYTES);
            state.banded_tiles += 1;
            state.failed.remove(&key);
            if self.rules.places(key.level, banding.placing) {
                self.place_in_bands(state, tile, (banding.priority, banding.placing));
            } else {
                state.arrive((tile, None), banding.priority, Some(&mut *self.slots.jobs));
            }
            self.inserted = true;
        }
    }

    /// Makes `tile`, which bands made, resident while its objects are
    /// placed in bands of candidate rows across workers too, joined in
    /// order.
    fn place_in_bands(
        &mut self,
        state: &mut TerrainState,
        tile: Tile,
        (priority, placing): (u32, Placing),
    ) {
        let key = tile.key();
        let spacing = state.terrain.pyramid.spacing(u32::from(key.level));
        let rows = self.rules.place_rows(key, spacing, placing);
        let starts: Vec<(u32, u32)> = rows
            .iter()
            .enumerate()
            .flat_map(|(layer, &rows)| (0..rows).map(move |row| (layer as u32, row)))
            .collect();
        let bands = self.work.bands.min(starts.len()).max(1);
        let mut jobs = Vec::with_capacity(bands);
        for band in 0..bands {
            let start = if band == 0 {
                (0, 0)
            } else {
                starts[band * starts.len() / bands]
            };
            let end = (band + 1 < bands).then(|| starts[(band + 1) * starts.len() / bands]);
            let job = PlaceJob {
                recipe: self.recipe.clone(),
                key,
                placing,
                from: start,
                end,
            };
            let id = self.slots.jobs.submit_cacheable(
                PlaceJob::KIND.name(),
                job.input(&tile),
                priority,
                job.identity(),
            );
            jobs.push((id, None));
        }
        state.placements.insert(
            key,
            Placements {
                made: self.generation,
                priority,
                placing,
                jobs,
            },
        );
        state.store.arrive(Arc::new(tile), Phase::Placing, None);
    }

    /// Then the objects of tiles bands made, once every band of them is
    /// done, joined in order, most urgent first.
    fn join_placements(&mut self, state: &mut TerrainState) {
        let mut placing: Vec<(u32, TileKey)> = state
            .placements
            .iter()
            .map(|(key, placements)| (placements.priority, key))
            .collect();
        placing.sort_by_key(|&(priority, _)| priority);
        for (_, key) in placing {
            if self.inserted && self.work.expired() {
                break;
            }
            let Some(placements) = state.placements.get_mut(&key) else {
                continue;
            };
            let mut failed = None;
            for (id, objects) in &mut placements.jobs {
                if objects.is_none()
                    && let Some(output) = self.slots.jobs.take(*id)
                {
                    match job_output(
                        PlaceJob::KIND.name(),
                        output,
                        engine_terrain::jobs::far_output,
                    ) {
                        Ok(placed) => *objects = Some(placed),
                        Err(error) => failed = Some(error),
                    }
                }
            }
            if failed.is_none() && placements.jobs.iter().any(|(_, objects)| objects.is_none()) {
                continue;
            }
            let Some(placements) = state.placements.remove(&key) else {
                continue;
            };
            if let Some(error) = failed {
                for (id, objects) in placements.jobs {
                    if objects.is_none() {
                        self.slots.jobs.cancel(id);
                        self.slots.release();
                    }
                }
                state.failed.insert(key, self.frame);
                state.failures.record(self.frame, error);
                continue;
            }
            let Some(tile) = state.store.get(&key).cloned() else {
                continue;
            };
            if placements.made != self.generation || !self.target.contains(&key) {
                continue;
            }
            let objects: Vec<engine_terrain::Placed> = placements
                .jobs
                .into_iter()
                .flat_map(|(_, objects)| objects.unwrap_or_default())
                .collect();
            // Its objects join the tile resident now, with its water.
            let watered = state.store.watered(&key).cloned().flatten();
            let tile = Tile::clone(&tile).with_placed(objects, placements.placing);
            state.arrive(
                (tile, watered),
                placements.priority,
                Some(&mut *self.slots.jobs),
            );
            self.inserted = true;
        }
    }

    /// Tiles still generating hand out their shape first, most urgent
    /// first: it draws, with coarser levels' materials, and collides until
    /// the whole tile arrives.
    fn take_shapes(&mut self, state: &mut TerrainState) {
        let mut shaping: Vec<(u32, TileKey, JobId, Generation)> = state
            .dispatched
            .iter()
            .map(|(key, &(id, made, priority))| (priority, key, id, made))
            .collect();
        shaping.sort_by_key(|&(priority, ..)| priority);
        for (_, key, id, made) in shaping {
            if !self.arrives() {
                break;
            }
            let Some(part) = self.slots.jobs.take_part(id) else {
                continue;
            };
            if let Ok(shape) = engine_terrain::jobs::tile_output(&part)
                && made == self.generation
                && self.target.contains(&key)
                && shape.key() == key
                && state.shape(shape)
            {
                // Its heights, and its ancestor's control values for cover.
                self.work.upload(2 * TILE_ARRAY_BYTES);
                self.inserted = true;
            }
        }
    }

    /// The objects standing far away that jobs placed on resident tiles,
    /// most urgent first.
    fn take_far(&mut self, state: &mut TerrainState) {
        let mut placing: Vec<(u32, TileKey, JobId)> = state
            .far_jobs
            .iter()
            .map(|(key, &(id, _, _, priority))| (priority, key, id))
            .collect();
        placing.sort_by_key(|&(priority, ..)| priority);
        for (_, key, id) in placing {
            if self.inserted && self.work.expired() {
                break;
            }
            let Some(output) = self.slots.jobs.take(id) else {
                continue;
            };
            let Some((_, tile, ..)) = state.far_jobs.remove(&key) else {
                continue;
            };
            match job_output(
                FarJob::KIND.name(),
                output,
                engine_terrain::jobs::far_output,
            ) {
                Ok(far) => {
                    // They join the tile resident now, whose ground they were
                    // placed on: its water may be newer.
                    let tile = state.store.get(&key).cloned().unwrap_or(tile);
                    let placed = merge_placed(tile.placed(), &far);
                    state.store.complete(Arc::new(
                        Tile::clone(&tile).with_placed(placed, Placing::All),
                    ));
                    self.inserted = true;
                }
                Err(error) => {
                    state.far_wanted.insert(key, job_priority(VIEW_JOBS, 0));
                    state.failures.record(self.frame, error);
                }
            }
        }
    }

    /// Tiles generated ahead of need go to memory; with tiles
    /// `missing`, or out of the idle reach, their jobs stop.
    fn take_ahead(&mut self, state: &mut TerrainState, missing: &[(TileKey, u32)]) {
        let idle: Vec<(TileKey, JobId, Generation)> = state
            .idle
            .iter()
            .map(|(key, &(id, made))| (key, id, made))
            .collect();
        for (key, id, made) in idle {
            if let Some(output) = self.slots.jobs.take(id) {
                state.idle.remove(&key);
                if let Some(tile) = output
                    .ok()
                    .and_then(|bytes| engine_terrain::jobs::tile_output(&bytes).ok())
                    && made == self.generation
                    && tile.key() == key
                    && !state.store.contains(&key)
                {
                    state.store.keep_in_memory(Arc::new(tile), None, self.frame);
                }
            } else if !missing.is_empty() || made != self.generation || !state.ahead.contains(&key)
            {
                self.slots.jobs.cancel(id);
                self.slots.release();
                state.idle.remove(&key);
            }
        }
    }

    /// Hands workers the objects standing far away on resident tile `key`.
    fn start_far(&mut self, state: &mut TerrainState, key: TileKey, priority: u32) {
        let (Some(tile), Some(arrival)) =
            (state.store.get(&key).cloned(), state.store.arrival(&key))
        else {
            return;
        };
        let job = FarJob {
            recipe: self.recipe.clone(),
            key,
        };
        let input = job.input(&tile);
        let id =
            self.slots
                .jobs
                .submit_cacheable(FarJob::KIND.name(), input, priority, job.identity());
        state.far_wanted.remove(&key);
        state.far_jobs.insert(key, (id, tile, arrival, priority));
    }

    /// Hands workers tile `key` in `bands` bands of rows; whether every
    /// band found a slot (unless it did, none runs and the tile generates
    /// whole).
    fn start_bands(
        &mut self,
        state: &mut TerrainState,
        (key, priority, placing): (TileKey, u32, Placing),
        bands: usize,
    ) -> bool {
        let size = (TILE_STRIDE as usize).div_ceil(bands);
        let mut jobs = Vec::with_capacity(bands);
        for band in 0..bands {
            if band > 0 && !self.slots.take(priority) {
                break;
            }
            let end = std::cmp::min((band + 1) * size, TILE_STRIDE as usize);
            let job = BandJob {
                recipe: self.recipe.clone(),
                key,
                first: (band * size) as u32,
                end: end as u32,
            };
            let id = self.slots.jobs.submit_cacheable(
                BandJob::KIND.name(),
                job.input(),
                priority,
                job.identity(),
            );
            jobs.push((id, None));
        }
        let covered = jobs.len() * size;
        if covered < TILE_STRIDE as usize {
            for (id, _) in jobs.drain(..) {
                self.slots.jobs.cancel(id);
                self.slots.release();
            }
            return false;
        }
        state.banded.insert(
            key,
            Banding {
                made: self.generation,
                priority,
                placing,
                jobs,
            },
        );
        true
    }
}

impl Executor for OnWorkers<'_, '_> {
    /// Tiles generating here stop: workers generate them.
    fn take_over(&mut self, state: &mut TerrainState) {
        state.jobs.clear();
    }

    fn collect(&mut self, state: &mut TerrainState, missing: &[(TileKey, u32)]) {
        self.stop_unwanted(state);
        self.take_tiles(state);
        self.join_bands(state);
        self.join_placements(state);
        self.take_shapes(state);
        self.take_far(state);
        self.take_ahead(state, missing);
    }

    /// Tiles under way on workers, and tiles that failed a moment ago,
    /// wait; so do objects standing far away being placed.
    fn waits(&self, state: &TerrainState, task: &TileTask) -> bool {
        let key = &task.key;
        if task.far {
            return state.far_jobs.contains_key(key);
        }
        state.dispatched.contains_key(key)
            || state.banded.contains_key(key)
            || state.placements.contains_key(key)
            || state
                .failed
                .get(key)
                .is_some_and(|&at| self.frame < at + RETRY_AFTER)
    }

    /// Tiles under the view split into bands of rows across the slots the
    /// urgent tiles leave, shared among them: each arrives sooner, though
    /// workers do a little more in all.
    fn begin(&mut self, _: &TerrainState, tasks: &[TileTask]) {
        self.urgent_left = tasks
            .iter()
            .filter(|task| !task.far && job_class(task.priority) == URGENT_JOBS)
            .count();
    }

    fn start(&mut self, state: &mut TerrainState, task: TileTask) -> bool {
        let TileTask { key, priority, far } = task;
        if !self.slots.take(priority) {
            return false;
        }
        if far {
            self.start_far(state, key, priority);
            return true;
        }
        // A tile replacing one resident, as after an edit, comes whole,
        // so its objects standing far away never leave the view; one not
        // resident hands out its shape first.
        let resident = state.store.contains(&key);
        let placing = if self.rules.places(key.level, Placing::Far)
            && (!resident || matches!(state.store.phase(&key), Some(Phase::Shape | Phase::Placing)))
        {
            Placing::Near
        } else {
            Placing::All
        };
        let urgent = job_class(priority) == URGENT_JOBS;
        self.urgent_left = self.urgent_left.saturating_sub(usize::from(urgent));
        let bands = if urgent && !resident {
            std::cmp::min(
                1 + self.slots.urgent() / (self.urgent_left + 1),
                self.work.bands,
            )
        } else {
            1
        };
        if bands > 1 && self.start_bands(state, (key, priority, placing), bands) {
            return true;
        }
        let job = TileJob {
            recipe: self.recipe.clone(),
            key,
            placing,
        };
        let id = self.slots.jobs.submit_cacheable(
            TileJob::KIND.name(),
            job.input(!resident),
            priority,
            job.identity(),
        );
        state
            .dispatched
            .insert(key, (id, self.generation, priority));
        true
    }

    /// With nothing else to hand out, a job a worker generates the nearest
    /// tiles within the idle reach, whole, into memory.
    fn ahead(&mut self, state: &mut TerrainState, missing: &[(TileKey, u32)]) {
        if !(missing.is_empty()
            && state.dispatched.is_empty()
            && state.banded.is_empty()
            && state.placements.is_empty()
            && state.far_wanted.is_empty()
            && state.far_jobs.is_empty())
        {
            return;
        }
        let workers = self.slots.jobs.workers() as usize;
        for index in 0..state.ahead.len() {
            if state.idle.len() >= workers {
                break;
            }
            let key = state.ahead[index];
            if state.idle.contains_key(&key) {
                continue;
            }
            let priority = job_priority(IDLE_JOBS, index as u32);
            if !self.slots.take(priority) {
                break;
            }
            let job = TileJob {
                recipe: self.recipe.clone(),
                key,
                placing: Placing::All,
            };
            let id = self.slots.jobs.submit_cacheable(
                TileJob::KIND.name(),
                job.input(false),
                priority,
                job.identity(),
            );
            state.idle.insert(key, (id, self.generation));
        }
    }
}

/// Shares the recipe workers generate `state`'s tiles from
/// once its bakes are done, and again whenever the layers its tiles
/// generate from change: their stack, shared once for every terrain seeing
/// it, and how its terrain sees it. Shares are named by their content,
/// so jobs reading them are the same bytes in every
/// session; `stacks` holds the stacks shared with their keys, and the key
/// of the recipe replaced goes to `released`.
pub(crate) fn share_recipe(
    state: &mut TerrainState,
    jobs: &mut Jobs,
    stacks: &mut Vec<(Arc<Layers>, String)>,
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
        generation: state.store.generation(),
        materials: state.materials,
    };
    if let Some(old) = state.recipe.replace(shared) {
        released.push(old.key);
    }
}
