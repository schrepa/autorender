//! Streaming each frame: what the view wants, loading it nearest and ahead
//! of the camera first, and the system running it.

use super::*;

impl Terrains {
    /// Tiles the device holds, which renderers report: residency never
    /// exceeds it. `None` for no limit.
    pub fn set_device_tiles(&mut self, tiles: Option<u32>) {
        self.device_tiles = tiles;
    }

    /// The clock frames time their terrain work by:
    /// milliseconds, from any origin. Hosts without one work a slice of
    /// `samples_per_frame` a frame.
    pub fn set_clock(&mut self, clock: fn() -> f64) {
        self.clock = Some(clock);
    }

    /// Whether the camera's view has yet to load whole;
    /// never without terrains.
    pub fn loading(&self) -> bool {
        !self.loaded && self.entries.iter().any(|state| state.around.is_none())
    }

    /// Works on terrain for the settings' `loading_work_ms` a frame again
    /// until the camera's view loads whole, as after the camera jumps far.
    pub fn begin_loading(&mut self) {
        self.loaded = false;
    }

    /// Whether the last frame found tiles to load: hosts then load them in
    /// time frames leave free.
    pub fn streaming(&self) -> bool {
        self.entries.iter().any(|state| !state.missing.is_empty())
    }

    /// Loads, for `milliseconds`, the tiles the last frame found missing,
    /// most important first, between frames: frames leave
    /// time free while the GPU draws them, which terrain otherwise only
    /// takes a few milliseconds of. Patches are selected anew with the next
    /// frame. Returns whether tiles are still to load; nothing without a
    /// clock (`set_clock`).
    pub fn work_between_frames(
        &mut self,
        settings: &TerrainSettings,
        milliseconds: f64,
        jobs: Option<&mut Jobs>,
    ) -> bool {
        let Some(clock) = self.clock else {
            return false;
        };
        let mut slots = jobs.map(JobSlots::of);
        // Smaller slices than frames take, so work ends close to its time.
        let work = Work {
            slice: u64::from(settings.samples_per_frame / 4).max(1),
            clock: Some(clock),
            end: clock() + milliseconds,
            upload: std::cell::Cell::new(self.upload_left),
            bands: settings.max_bands as usize,
        };
        let mut budget = work.slice;
        let frame = self.frame;
        let requests = &mut self.requests;
        let mut remaining = false;
        for state in &mut self.entries {
            if state.terrain.validate().is_err() || !settings.enabled {
                continue;
            }
            let mut missing: Vec<(TileKey, u32)> = std::mem::take(&mut state.missing)
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            // Tiles kept in memory return at once.
            if !state.cache.is_empty() {
                self.restored += state.restore(&missing, &work);
                missing.retain(|(key, _)| state.loads(key));
            }
            if missing.is_empty() && state.far_wanted.is_empty() && state.far_jobs.is_empty() {
                continue;
            }
            let target = std::mem::take(&mut state.target);
            let keep = std::mem::take(&mut state.keep);
            let baking = load_tiles(
                state,
                &missing,
                (&target, &keep),
                &work,
                &mut budget,
                frame,
                requests,
                slots.as_mut(),
            );
            state.missing = missing
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            state.target = target;
            state.keep = keep;
            // Tiles waiting on their fetch load once delivered.
            let waiting = matches!(state.terrain.source, TerrainSource::File { data: None, .. })
                && state.arrivals.is_empty();
            remaining |= baking
                || (!state.missing.is_empty() && !waiting)
                || !state.dispatched.is_empty()
                || !state.far_wanted.is_empty()
                || !state.far_jobs.is_empty();
            if budget == 0 && !work.refill(&mut budget) {
                break;
            }
        }
        // What arrived uploads with the next frame.
        self.between += self.upload_left - work.upload.get();
        self.upload_left = work.upload.get();
        remaining
    }

    /// Where `eye`, moved since the last frame `seconds` ago, is headed:
    /// its velocity, smoothed over frames, carried
    /// `AHEAD_SECONDS` on, at most `AHEAD_MAX` meters; the eye itself after
    /// a jump.
    fn heading(&mut self, eye: Vec3, seconds: f32) -> Vec3 {
        let moved = self.last_eye.map_or(Vec3::ZERO, |last| eye - last);
        self.last_eye = Some(eye);
        self.seconds = if seconds.is_finite() {
            crate::math::max(seconds, 0.0)
        } else {
            0.0
        };
        if moved.length() > JUMP {
            // A jump loads the view again at the loading budget,
            // as the first view loads, unless workers load it.
            self.velocity = Vec3::ZERO;
            if self.workers == 0 {
                self.loaded = false;
            }
        } else if seconds > 0.0 {
            self.velocity = self.velocity.lerp(moved / seconds, 0.25);
        }
        if !self.velocity.is_finite() {
            self.velocity = Vec3::ZERO;
        }
        eye + (self.velocity * AHEAD_SECONDS).clamp_length_max(AHEAD_MAX)
    }

    /// Streams every terrain toward `eye` (the camera's, unless frozen)
    /// and focus points, within the settings' work and tile budgets, and
    /// selects the patches seen from `eye`.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn stream(
        &mut self,
        settings: &TerrainSettings,
        camera: &Camera,
        (eye, ahead): (Vec3, Vec3),
        viewport: Viewport,
        (focus, places): (&[Vec3], &[(Vec3, f32)]),
        jobs: Option<&mut Jobs>,
    ) {
        self.frame += 1;
        self.sync_surroundings();
        // Jobs of terrains gone are dropped, and so are their recipes.
        let mut slots = jobs.map(JobSlots::of);
        let retired = std::mem::take(&mut self.retired);
        self.workers = slots.as_ref().map_or(0, |slots| slots.jobs.workers());
        let workers = self.workers;
        // Keys no recipe may hold any longer, unshared at the end unless
        // one does.
        let mut released = Vec::new();
        if let Some(slots) = slots.as_mut() {
            for id in retired.jobs {
                slots.jobs.cancel(id);
            }
            released.extend(retired.recipes);
        }
        let mut stats = TerrainStats {
            terrains: self
                .entries
                .iter()
                .filter(|state| state.around.is_none())
                .count() as u32,
            bridges: self
                .entries
                .iter()
                .flat_map(|state| state.bridges.iter())
                .filter(|bridge| !bridge.tunnel)
                .count() as u32,
            tunnels: self
                .entries
                .iter()
                .flat_map(|state| state.bridges.iter())
                .filter(|bridge| bridge.tunnel)
                .count() as u32,
            falls: self
                .entries
                .iter()
                .map(|state| state.falls.len() as u32)
                .sum(),
            ..TerrainStats::default()
        };
        let forward = camera.target - camera.eye();
        let forward = Vec2::new(forward.x, forward.z);
        let planes = ViewPlanes::new(camera, eye, viewport);
        let fov = camera.vertical_fov_radians;
        let fov = if fov.is_finite() && fov > 0.0 && fov < std::f32::consts::PI {
            fov
        } else {
            Camera::default().vertical_fov_radians
        };
        // The frame's work: slices until its time is up.
        let work = Work {
            slice: u64::from(settings.samples_per_frame),
            clock: self.clock,
            end: self.clock.map_or(0.0, |clock| {
                clock()
                    + if self.loaded {
                        f64::from(settings.work_ms)
                    } else {
                        f64::from(settings.loading_work_ms)
                    }
            }),
            // Tiles arriving between frames upload with this frame's.
            upload: std::cell::Cell::new(
                u64::from(settings.upload_bytes).saturating_sub(std::mem::take(&mut self.between)),
            ),
            bands: settings.max_bands as usize,
        };
        let mut budget = work.slice;
        // Speed-aware detail: the finest levels waiting at
        // the eye's speed, and whether it has yet to settle.
        let (waiting, unsettled) = match settings.motion_detail {
            Some(detail) => {
                let speed = self.velocity.length();
                self.fast_ago = if speed > detail.speed {
                    Some(0.0)
                } else {
                    self.fast_ago.map(|ago| ago + self.seconds)
                };
                (
                    detail.waiting(speed),
                    self.fast_ago.is_some_and(|ago| ago < detail.settle),
                )
            }
            None => {
                self.fast_ago = None;
                (0, false)
            }
        };
        let limit = match (settings.max_tiles, self.device_tiles) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        // Budgets cover every terrain: each takes what is left.
        let mut remaining = limit.map_or(usize::MAX, |limit| limit as usize);
        let frame = self.frame;
        // Memory's room for tiles generated ahead of need.
        let room = settings.cache_bytes.saturating_sub(
            self.entries
                .iter()
                .map(|state| state.cached_bytes)
                .sum::<u64>(),
        );
        for state in &mut self.entries {
            if state.terrain.validate().is_err() {
                continue;
            }
            // Surroundings leave the ground they surround to
            // their terrain, and stream while settings show them.
            let cutout = state.around.as_ref().map(|around| around.cutout);
            if cutout.is_some() && !settings.surroundings {
                state.selection.patches.clear();
                state.selection.fallbacks = 0;
                continue;
            }
            let pyramid = state.terrain.pyramid;
            let origin = state.terrain.origin;
            let ranges =
                LodRanges::new(&pyramid, settings.pixel_error, viewport.height as f32, fov);
            state.ranges.clear();
            state
                .ranges
                .extend((0..pyramid.levels()).map(|level| ranges.range(level)));
            let local_eye = eye - origin;
            // Collision's tiles load within its radius of the camera and
            // focus entities and stay, colliding, a quarter farther, as the
            // camera's tiles do: focus moving to and fro across a tile's
            // edge must not regenerate it each time.
            state.collision.clear();
            let mut loading = Vec::new();
            let mut under = Vec::new();
            if state.terrain.collision && settings.collision_radius > 0.0 {
                let held = settings.collision_radius * KEEP_REACH / LOAD_REACH;
                for point in focus {
                    let local = *point - origin;
                    collision_keys(&pyramid, local, settings.collision_radius, &mut loading);
                    collision_keys(&pyramid, local, held, &mut state.collision);
                    under.extend(pyramid.tile_at(0, Vec2::new(local.x, local.z)));
                }
                loading.sort_unstable();
                loading.dedup();
                state.collision.sort_unstable();
                state.collision.dedup();
            }
            // What the view wants, the camera's tiles within
            // their reach of the eye and where it is headed,
            // and what it keeps, within a wider reach; distances run to each
            // tile's ground.
            // Ahead of need, while nothing was missing, nor
            // generating, at the last frame and memory has room.
            let quiet = workers > 0
                && state.missing.is_empty()
                && state.dispatched.is_empty()
                && state.banded.is_empty()
                && state.placements.is_empty()
                && state.far_wanted.is_empty()
                && state.far_jobs.is_empty();
            let (wanted, headed, held, around, named) = {
                let tiles = &state.tiles;
                let errors = &state.errors;
                let ground = |key: TileKey| ground_bounds(tiles, &pyramid, errors, key);
                let seen = |low: Vec3, high: Vec3| planes.sees(low + origin, high + origin);
                let view = |eye: Vec3| WantedView {
                    eye,
                    forward,
                    ground: &ground,
                    seen: &seen,
                };
                let want = |eye: Vec3, reach: f32| {
                    let mut out = Vec::new();
                    engine_terrain::wanted_tiles(
                        &pyramid,
                        &ranges,
                        &view(eye),
                        reach,
                        settings.pinned_levels,
                        cutout,
                        &mut out,
                    );
                    out
                };
                (
                    want(local_eye, LOAD_REACH),
                    if ahead == eye {
                        Vec::new()
                    } else {
                        want(ahead - origin, LOAD_REACH)
                    },
                    want(local_eye, KEEP_REACH),
                    if quiet && settings.idle_reach > KEEP_REACH {
                        want(local_eye, settings.idle_reach)
                    } else {
                        Vec::new()
                    },
                    // What places games name want, each frame.
                    places
                        .iter()
                        .flat_map(|&(place, reach)| want(place - origin, reach))
                        .collect::<Vec<_>>(),
                )
            };
            // What to load, most important first, each tile after its
            // ancestors: the pinned levels; the tiles the
            // view draws and shows, then those under focus entities, which
            // jobs take up first; the tiles it draws out of sight, then
            // collision's; those only shading what it shows; where the eye
            // is headed; the rest.
            let tier = |pick: fn(&WantedTile) -> bool| {
                wanted
                    .iter()
                    .filter(move |tile| pick(tile))
                    .map(|tile| tile.key)
            };
            let mut order: Vec<TileKey> = Vec::with_capacity(wanted.len() + loading.len());
            let mut pushed: HashSet<TileKey> = HashSet::with_capacity(order.capacity());
            for key in tier(|tile| tile.score < 0.0)
                .chain(tier(|tile| tile.score >= 0.0 && tile.drawn && tile.seen))
                .chain(under.iter().copied())
            {
                push_with_ancestors(&pyramid, key, &mut order, &mut pushed);
            }
            let urgent = order.len();
            for key in tier(|tile| tile.score >= 0.0 && tile.drawn && !tile.seen)
                .chain(loading.iter().copied())
                .chain(tier(|tile| tile.score >= 0.0 && !tile.drawn && tile.seen))
                .chain(headed.iter().map(|tile| tile.key))
                .chain(tier(|tile| tile.score >= 0.0 && !tile.drawn && !tile.seen))
            {
                push_with_ancestors(&pyramid, key, &mut order, &mut pushed);
            }
            // Over a tile budget, the pinned levels and collision's tiles
            // stay first, then the rest in order.
            let target: HashSet<TileKey> = if order.len() <= remaining {
                order.iter().copied().collect()
            } else {
                let mut first = Vec::new();
                let mut held = HashSet::new();
                for key in tier(|tile| tile.score < 0.0).chain(loading.iter().copied()) {
                    push_with_ancestors(&pyramid, key, &mut first, &mut held);
                }
                first
                    .iter()
                    .copied()
                    .chain(order.iter().copied().filter(|key| !held.contains(key)))
                    .take(remaining)
                    .collect()
            };
            let kept = target.len();
            stats.dropped_tiles += (order.len() - kept) as u32;
            remaining -= kept;
            // What to keep: everything within a wider reach, unless over
            // budget.
            let keep: HashSet<TileKey> = held
                .iter()
                .map(|tile| tile.key)
                .chain(state.collision.iter().copied())
                .chain(target.iter().copied())
                .collect();
            let before = state.tiles.len();
            // Tiles leaving stay in memory.
            let leaving: Vec<TileKey> = state
                .tiles
                .keys()
                .filter(|key| !keep.contains(key))
                .copied()
                .collect();
            for key in leaving {
                if let Some(tile) = state.tiles.remove(&key) {
                    state.cache_leaving(key, tile, frame);
                }
            }
            if kept < order.len() && state.tiles.len() > kept {
                // Over budget: drop resident tiles outside the target, finest
                // first. Within it, tiles the view only keeps (between the
                // loading and keeping reaches) stay: a tile whose own
                // bounds put it past the loading reach, where its parent's
                // had it inside, must not leave and return frame by frame.
                let mut extra: Vec<TileKey> = state
                    .tiles
                    .keys()
                    .filter(|key| !target.contains(key))
                    .copied()
                    .collect();
                extra.sort_unstable_by_key(|key| key.level);
                for key in extra.into_iter().take(state.tiles.len() - kept) {
                    if let Some(tile) = state.tiles.remove(&key) {
                        state.cache_leaving(key, tile, frame);
                    }
                }
            }
            if state.tiles.len() != before {
                state.revision += 1;
            }
            state.jobs.retain(|key, _| target.contains(key));
            // Load what is missing, most important first; disabled terrains
            // load nothing.
            // Tiles under edits since they loaded regenerate
            // in their turn and draw until they do.
            state.stale.retain(|key| state.tiles.contains_key(key));
            state.shaped.retain(|key| state.tiles.contains_key(key));
            // While the eye moves fast, the finest levels'
            // tiles wait, and until it settles those it does not show; the
            // pinned levels, collision's tiles and those under focus
            // entities, with their ancestors, never wait.
            let mut never = HashSet::new();
            let mut shown = HashSet::new();
            if waiting > 0 || unsettled {
                let mut chain = Vec::new();
                for key in tier(|tile| tile.score < 0.0)
                    .chain(loading.iter().copied())
                    .chain(under.iter().copied())
                {
                    push_with_ancestors(&pyramid, key, &mut chain, &mut never);
                }
                if unsettled {
                    chain.clear();
                    for key in tier(|tile| tile.seen) {
                        push_with_ancestors(&pyramid, key, &mut chain, &mut shown);
                    }
                }
            }
            let waits = |key: &TileKey| {
                (waiting > 0 || unsettled)
                    && !never.contains(key)
                    && (u32::from(key.level) < waiting || (unsettled && !shown.contains(key)))
            };
            stats.waiting_tiles += order
                .iter()
                .filter(|key| target.contains(key) && state.loads(key) && waits(key))
                .count() as u32;
            let mut missing: Vec<(TileKey, u32)> = order
                .iter()
                .enumerate()
                .filter(|(_, key)| {
                    settings.enabled && target.contains(key) && state.loads(key) && !waits(key)
                })
                .map(|(index, &key)| {
                    let class = if index < urgent {
                        URGENT_JOBS
                    } else {
                        VIEW_JOBS
                    };
                    (key, job_priority(class, index as u32))
                })
                .collect();
            // Tiles kept in memory return at once.
            if !state.cache.is_empty() {
                self.restored += state.restore(&missing, &work);
                missing.retain(|(key, _)| state.loads(key));
            }
            // Tiles places games name want stay in memory before those
            // that only left the view.
            for tile in &named {
                if let Some(cached) = state.cache.get_mut(&tile.key) {
                    cached.left = frame;
                }
            }
            // What workers generate ahead of need: the nearest tiles within
            // the idle reach and that places games name want,
            // not resident, kept or kept in memory, as many as memory has
            // room for, so none it holds goes to make room for another.
            state.ahead.clear();
            if quiet {
                let mut nearest: Vec<&WantedTile> = around.iter().chain(&named).collect();
                nearest.sort_by(|a, b| {
                    a.score
                        .total_cmp(&b.score)
                        .then(b.key.level.cmp(&a.key.level))
                });
                let fits = room
                    .checked_div(state.tile_bytes)
                    .map_or(usize::MAX, |fits| fits as usize);
                let mut listed = HashSet::new();
                state.ahead.extend(
                    nearest
                        .into_iter()
                        .map(|tile| tile.key)
                        .filter(|key| {
                            listed.insert(*key)
                                && !keep.contains(key)
                                && !state.tiles.contains_key(key)
                                && !state.cache.contains_key(key)
                        })
                        .take(fits),
                );
            }
            // What the frame found missing loads between frames too.
            state.missing.clone_from(&missing);
            state.target.clone_from(&target);
            state.keep.clone_from(&keep);
            // GPU materials before the recipe that carries
            // them.
            state.update_materials(settings.gpu_materials);
            // Workers generate from its recipe.
            if let Some(slots) = slots.as_mut()
                && slots.jobs.workers() > 0
            {
                share_recipe(
                    state,
                    slots.jobs,
                    (&mut self.stacks, &mut self.shares),
                    &mut released,
                );
            }
            if load_tiles(
                state,
                &missing,
                (&target, &keep),
                &work,
                &mut budget,
                frame,
                &mut self.requests,
                slots.as_mut(),
            ) {
                stats.baking += 1;
                continue;
            }
            let missing = std::mem::take(&mut state.missing);
            state.missing = missing
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            stats.resident_tiles += state.tiles.len() as u32;
            stats.missing_tiles += state.missing.len() as u32;
            stats.shaped_tiles += state.shaped.len() as u32;
            stats.cached_tiles += state.cache.len() as u32;
            stats.cached_bytes += state.cached_bytes;
            stats.idle_tiles += state.idle.len() as u32;
            stats.loading_tiles += (state.jobs.len()
                + state.in_flight.len()
                + state.arrivals.len()
                + state.dispatched.len()
                + state.far_wanted.len()
                + state.far_jobs.len()
                + state.banded.len()
                + state.placements.len()) as u32;
            stats.banded_tiles += state.banded_tiles;
            stats.unplaced_tiles += state.unplaced_tiles();
            if settings.enabled {
                let tiles = &state.tiles;
                engine_terrain::select_patches(
                    &pyramid,
                    &ranges,
                    local_eye,
                    |key| tiles.get(&key).map(Arc::as_ref),
                    cutout,
                    &mut state.selection,
                );
            } else {
                state.selection.patches.clear();
                state.selection.fallbacks = 0;
            }
            stats.patches += state.selection.patches.len() as u32;
            stats.fallback_patches += state.selection.fallbacks;
            if cutout.is_some() {
                stats.surroundings_tiles += state.tiles.len() as u32;
                stats.surroundings_patches += state.selection.patches.len() as u32;
            }
        }
        // Tiles kept in memory beyond its budget go, those that left first.
        let cached: u64 = self.entries.iter().map(|state| state.cached_bytes).sum();
        if cached > settings.cache_bytes {
            let mut oldest: Vec<(u64, usize, TileKey)> = self
                .entries
                .iter()
                .enumerate()
                .flat_map(|(index, state)| {
                    state
                        .cache
                        .iter()
                        .map(move |(key, kept)| (kept.left, index, *key))
                })
                .collect();
            oldest.sort_unstable();
            let mut over = cached - settings.cache_bytes;
            for (_, index, key) in oldest {
                if over == 0 {
                    break;
                }
                let state = &mut self.entries[index];
                let before = state.cached_bytes;
                state.uncache(key);
                over = over.saturating_sub(before - state.cached_bytes);
            }
        }
        stats.restored_tiles = self.restored;
        stats.cached_bytes = self.entries.iter().map(|state| state.cached_bytes).sum();
        stats.cached_tiles = self
            .entries
            .iter()
            .map(|state| state.cache.len() as u32)
            .sum();
        self.upload_left = work.upload.get();
        // Stacks no terrain's recipe shares any longer leave the workers;
        // keys named by content leave once
        // no recipe or stack holds them.
        if let Some(slots) = slots.as_mut() {
            let entries = &self.entries;
            self.stacks.retain(|(stack, key)| {
                let held = entries.iter().any(|state| {
                    state
                        .recipe
                        .as_ref()
                        .is_some_and(|recipe| Arc::ptr_eq(&recipe.stack, stack))
                });
                if !held {
                    released.push(key.clone());
                }
                held
            });
            for key in released {
                let held = self.stacks.iter().any(|(_, shared)| *shared == key)
                    || self.entries.iter().any(|state| {
                        state
                            .recipe
                            .as_ref()
                            .is_some_and(|recipe| recipe.key == key)
                    });
                if !held {
                    slots.jobs.unshare(key);
                }
            }
        }
        // The camera's view has loaded whole once nothing bakes, loads or
        // stands in for a finer level.
        if !self.loaded
            && stats.terrains > 0
            && stats.baking == 0
            && stats.loading_tiles == 0
            && stats.fallback_patches == 0
        {
            self.loaded = true;
        }
        stats.loading = self.loading();
        self.stats = stats;
    }
}

/// The lowest and highest ground tile `key` can hold,
/// terrain-local, as distances to it take them: a resident tile's own,
/// widened by how far its level measured from the next coarser (its
/// patches morph toward that); else its nearest resident ancestor's over
/// its ground, widened by how far the levels between measured apart and by
/// four of the ancestor's spacings, as finer levels add detail. `None`
/// without either.
fn ground_bounds(
    tiles: &HashMap<TileKey, Arc<Tile>>,
    pyramid: &Pyramid,
    errors: &[f32],
    key: TileKey,
) -> Option<[f32; 2]> {
    let error = |level: u8| errors.get(usize::from(level)).copied().unwrap_or(0.0);
    if let Some(tile) = tiles.get(&key) {
        let [low, high] = tile.bounds();
        return Some([low - error(key.level), high + error(key.level)]);
    }
    let size = pyramid.tile_size(u32::from(key.level));
    let low = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
    let high = low + size;
    let mut ancestor = key;
    let mut margin = 0.0;
    while let Some(parent) = pyramid.parent(ancestor) {
        margin += error(ancestor.level);
        ancestor = parent;
        let Some(tile) = tiles.get(&ancestor) else {
            continue;
        };
        let level = u32::from(ancestor.level);
        let spacing = pyramid.spacing(level);
        let patch = PATCH_QUADS as f32 * spacing;
        let corner =
            Vec2::new(f32::from(ancestor.x), f32::from(ancestor.z)) * pyramid.tile_size(level);
        let last = (PATCHES_PER_TILE - 1) as f32;
        let first = ((low - corner) / patch)
            .floor()
            .clamp(Vec2::ZERO, Vec2::splat(last));
        let end = ((high - corner) / patch - 1.0)
            .ceil()
            .clamp(Vec2::ZERO, Vec2::splat(last));
        let (mut bottom, mut top) = (f32::INFINITY, f32::NEG_INFINITY);
        for z in first.y as u32..=end.y as u32 {
            for x in first.x as u32..=end.x as u32 {
                let [patch_low, patch_high] = tile.patch_bounds(x, z);
                bottom = crate::math::min(bottom, patch_low);
                top = crate::math::max(top, patch_high);
            }
        }
        let margin = margin + 4.0 * spacing;
        return Some([bottom - margin, top + margin]);
    }
    None
}

/// The view's side planes and near plane, inward, in world space:
/// which boxes show in it, however far away.
struct ViewPlanes(Option<[Vec4; 5]>);

impl ViewPlanes {
    /// `camera`'s view from `eye` (its own, unless frozen) on `viewport`.
    fn new(camera: &Camera, eye: Vec3, viewport: Viewport) -> Self {
        let aspect = viewport.width as f32 / viewport.height.max(1) as f32;
        let from = Camera {
            position: eye,
            target: eye + (camera.target - camera.eye()),
            ..*camera
        };
        let rows = from.view_projection(aspect).transpose();
        let planes = [
            rows.w_axis + rows.x_axis,
            rows.w_axis - rows.x_axis,
            rows.w_axis + rows.y_axis,
            rows.w_axis - rows.y_axis,
            // Reversed depth: the near plane is depth 1.
            rows.w_axis - rows.z_axis,
        ];
        Self(
            planes
                .iter()
                .all(|plane| plane.is_finite())
                .then_some(planes),
        )
    }

    /// Whether any of the box from `low` to `high` lies inside every plane;
    /// always without planes.
    fn sees(&self, low: Vec3, high: Vec3) -> bool {
        let Some(planes) = &self.0 else {
            return true;
        };
        planes.iter().all(|plane| {
            let corner = Vec3::new(
                if plane.x >= 0.0 { high.x } else { low.x },
                if plane.y >= 0.0 { high.y } else { low.y },
                if plane.z >= 0.0 { high.z } else { low.z },
            );
            plane.truncate().dot(corner) + plane.w >= 0.0
        })
    }
}

/// Appends the level-0 tiles within `radius` of terrain-local `point` to
/// `out`.
fn collision_keys(pyramid: &Pyramid, point: Vec3, radius: f32, out: &mut Vec<TileKey>) {
    let size = pyramid.tile_size(0);
    let [tiles_x, tiles_z] = pyramid.tiles(0);
    let low = ((Vec2::new(point.x, point.z) - radius) / size).floor();
    let high = ((Vec2::new(point.x, point.z) + radius) / size).floor();
    if high.x < 0.0 || high.y < 0.0 || low.x >= tiles_x as f32 || low.y >= tiles_z as f32 {
        return;
    }
    let first = [
        crate::math::max(low.x, 0.0) as u32,
        crate::math::max(low.y, 0.0) as u32,
    ];
    let last = [
        (high.x as u32).min(tiles_x - 1),
        (high.y as u32).min(tiles_z - 1),
    ];
    for z in first[1]..=last[1] {
        for x in first[0]..=last[0] {
            out.push(TileKey::new(0, x as u16, z as u16));
        }
    }
}

/// What terrain streaming reads each frame: the terrains, the focus
/// entities and the places named for prefetch, with its lists of them.
pub(crate) struct TerrainStreaming {
    terrains: QueryState<(Entity, &'static Terrain)>,
    focus: QueryState<Entity, With<crate::StreamingFocus>>,
    prefetch: QueryState<(Entity, &'static TerrainPrefetch)>,
    points: Vec<Vec3>,
    places: Vec<(Vec3, f32)>,
}

impl FromWorld for TerrainStreaming {
    fn from_world(world: &mut World) -> Self {
        Self {
            terrains: world.query::<(Entity, &Terrain)>(),
            focus: world.query_filtered::<Entity, With<crate::StreamingFocus>>(),
            prefetch: world.query::<(Entity, &TerrainPrefetch)>(),
            points: Vec::new(),
            places: Vec::new(),
        }
    }
}

/// Streams terrains toward the frame's camera and focus entities:
/// [`crate::ViewSyncSystems::Stream`].
pub(crate) fn stream_terrains(world: &mut World, mut streaming: Local<TerrainStreaming>) {
    streaming.update(world);
}

/// Loads tiles the last frame found missing for `milliseconds` between
/// frames; whether tiles are still to load.
pub(crate) fn work_between_frames(world: &mut World, milliseconds: f64) -> bool {
    let Some(settings) = world
        .get_resource::<TerrainSettings>()
        .copied()
        .map(TerrainSettings::sanitized)
    else {
        return false;
    };
    let mut jobs = world.remove_resource::<Jobs>();
    let remaining = world
        .get_resource_mut::<Terrains>()
        .is_some_and(|mut terrains| {
            terrains.work_between_frames(&settings, milliseconds, jobs.as_mut())
        });
    if let Some(jobs) = jobs {
        world.insert_resource(jobs);
    }
    remaining
}

impl TerrainStreaming {
    /// Follows added, changed and removed terrains, then streams them
    /// toward this frame's camera and focus entities.
    pub(crate) fn update(&mut self, world: &mut World) {
        let _profile = engine_profile::span("cpu", "terrain_stream");
        let settings = world
            .get_resource::<TerrainSettings>()
            .copied()
            .map(TerrainSettings::sanitized);
        let camera = *world.resource::<Camera>();
        let viewport = world
            .get_resource::<Viewport>()
            .copied()
            .unwrap_or_default();
        let eye = world
            .get_resource::<TerrainDebug>()
            .and_then(|debug| debug.freeze)
            .unwrap_or_else(|| camera.eye());
        let seconds = world
            .get_resource::<crate::FrameTime>()
            .map_or(0.0, |time| time.delta_seconds);
        self.points.clear();
        self.points.push(eye);
        self.places.clear();
        if let Some(globals) = world.get_resource::<GlobalTransforms>() {
            for entity in self.focus.iter(world) {
                if let Some(point) = globals.get(entity).and_then(|global| global.translation()) {
                    self.points.push(point);
                }
            }
            // Places games name, within their reach.
            for (entity, prefetch) in self.prefetch.iter(world) {
                let reach = prefetch.reach;
                if reach.is_finite()
                    && reach > 0.0
                    && let Some(point) = globals.get(entity).and_then(|global| global.translation())
                {
                    self.places.push((point, crate::math::min(reach, 8.0)));
                }
            }
        }
        let present: Vec<(Entity, Terrain)> = self
            .terrains
            .iter(world)
            .map(|(entity, terrain)| (entity, terrain.clone()))
            .collect();
        // Workers' jobs stream with the terrains.
        let mut jobs = world.remove_resource::<Jobs>();
        let Some(mut terrains) = world.get_resource_mut::<Terrains>() else {
            if let Some(jobs) = jobs {
                world.insert_resource(jobs);
            }
            return;
        };
        let terrains = terrains.as_mut();
        let retired = &mut terrains.retired;
        terrains.entries.retain(|state| {
            let kept = present.iter().any(|(entity, _)| *entity == state.entity);
            if !kept {
                state.retire(retired);
            }
            kept
        });
        for (entity, terrain) in present {
            match terrains
                .entries
                .iter_mut()
                .find(|state| state.entity == entity && state.around.is_none())
            {
                Some(state) => {
                    if !terrain.same(&state.terrain) {
                        state.retire(&mut terrains.retired);
                        *state = TerrainState::new(entity, terrain);
                        terrains.loaded = false;
                    } else {
                        if !Arc::ptr_eq(&terrain.edits, &state.terrain.edits) {
                            state.edit(terrain.edits.clone());
                        }
                        // Surroundings follow on their own.
                        state.terrain.surroundings = terrain.surroundings;
                        // What placement layers stand and the bridges'
                        // material change in place: the objects tiles placed
                        // stay where they are, and what reads the terrain
                        // (drawing, collision, walks, bridges) takes the new.
                        if terrain.surface.assets != state.terrain.surface.assets
                            || terrain.surface.bridge != state.terrain.surface.bridge
                        {
                            state.terrain.surface.assets = terrain.surface.assets;
                            state.terrain.surface.bridge = terrain.surface.bridge;
                            state.revision += 1;
                        }
                    }
                }
                None => {
                    terrains.entries.push(TerrainState::new(entity, terrain));
                    terrains.loaded = false;
                }
            }
        }
        let Some(settings) = settings else {
            for state in &mut terrains.entries {
                state.selection.patches.clear();
            }
            terrains.stats = TerrainStats {
                terrains: terrains
                    .entries
                    .iter()
                    .filter(|state| state.around.is_none())
                    .count() as u32,
                ..TerrainStats::default()
            };
            // Terrains that do not stream have nothing to load.
            terrains.loaded = true;
            if let Some(jobs) = jobs {
                world.insert_resource(jobs);
            }
            return;
        };
        // Where the eye is headed, from its motion since the
        // last frame.
        let ahead = terrains.heading(eye, seconds);
        terrains.stream(
            &settings,
            &camera,
            (eye, ahead),
            viewport,
            (&self.points, &self.places),
            jobs.as_mut(),
        );
        if let Some(jobs) = jobs {
            world.insert_resource(jobs);
        }
    }
}
