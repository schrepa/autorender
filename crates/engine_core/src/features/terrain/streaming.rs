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
    /// milliseconds, from any origin; or none (the default), where each
    /// frame works one slice of `samples_per_frame` and nothing loads
    /// between frames, so what streams depends on the frames alone.
    pub fn set_clock(&mut self, clock: Option<fn() -> f64>) {
        self.clock = clock;
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
        let mut slots = jobs.map(JobBudget::of);
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
            if state.store.kept_tiles() > 0 {
                self.restored += state.store.restore(&missing, &work);
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
        #[cfg(debug_assertions)]
        for state in &self.entries {
            state.check();
        }
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
        let mut slots = jobs.map(JobBudget::of);
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
            failed_reads: self.degraded.failed_reads,
            refused_falls: self.degraded.refused_falls,
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
                .map(|state| state.store.cached_bytes())
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
            // Ahead of need, while nothing was missing, nor
            // generating, at the last frame and memory has room.
            let quiet = workers > 0
                && state.missing.is_empty()
                && state.dispatched.is_empty()
                && state.banded.is_empty()
                && state.placements.is_empty()
                && state.far_wanted.is_empty()
                && state.far_jobs.is_empty();
            let Plan {
                collision,
                ordered,
                target,
                keep,
                mut missing,
                waiting: waiting_tiles,
                around,
                named,
            } = plan(PlanInput {
                settings,
                pyramid,
                origin,
                ranges: &ranges,
                eye,
                ahead,
                forward,
                planes: &planes,
                cutout,
                focus,
                places,
                collides: state.terrain.collision,
                collision: std::mem::take(&mut state.collision),
                quiet,
                remaining,
                waiting,
                unsettled,
                store: &state.store,
            });
            state.collision = collision;
            let kept = target.len();
            stats.dropped_tiles += (ordered - kept) as u32;
            remaining -= kept;
            // Tiles leaving stay in memory.
            let leaving: Vec<TileKey> = state
                .store
                .keys()
                .filter(|key| !keep.contains(key))
                .copied()
                .collect();
            for key in leaving {
                state.leave(&key, frame);
            }
            if kept < ordered && state.store.len() > kept {
                // Over budget: drop resident tiles outside the target, finest
                // first. Within it, tiles the view only keeps (between the
                // loading and keeping reaches) stay: a tile whose own
                // bounds put it past the loading reach, where its parent's
                // had it inside, must not leave and return frame by frame.
                let mut extra: Vec<TileKey> = state
                    .store
                    .keys()
                    .filter(|key| !target.contains(key))
                    .copied()
                    .collect();
                extra.sort_unstable_by_key(|key| key.level);
                for key in extra.into_iter().take(state.store.len() - kept) {
                    state.leave(&key, frame);
                }
            }
            state.jobs.retain(|key, _| target.contains(key));
            stats.waiting_tiles += waiting_tiles;
            // Tiles kept in memory return at once.
            if state.store.kept_tiles() > 0 {
                self.restored += state.store.restore(&missing, &work);
                missing.retain(|(key, _)| state.loads(key));
            }
            // Tiles places games name want stay in memory before those
            // that only left the view.
            for tile in &named {
                state.store.wanted_in_memory(&tile.key, frame);
            }
            state.ahead.clear();
            if quiet {
                let fits = room
                    .checked_div(state.store.tile_bytes())
                    .map_or(usize::MAX, |fits| fits as usize);
                state.ahead = ahead_of_need((&around, &named), &keep, fits, &state.store);
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
                share_recipe(state, slots.jobs, &mut self.stacks, &mut released);
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
            stats.resident_tiles += state.store.len() as u32;
            stats.missing_tiles += state.missing.len() as u32;
            stats.shaped_tiles += state.store.count(&[Phase::Shape, Phase::Placing]) as u32;
            stats.cached_tiles += state.store.kept_tiles() as u32;
            stats.cached_bytes += state.store.cached_bytes();
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
                let store = &state.store;
                engine_terrain::select_patches(
                    &pyramid,
                    &ranges,
                    local_eye,
                    |key| store.get(&key).map(Arc::as_ref),
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
                stats.surroundings_tiles += state.store.len() as u32;
                stats.surroundings_patches += state.selection.patches.len() as u32;
            }
        }
        // Tiles kept in memory beyond its budget go, those that left first.
        let cached: u64 = self
            .entries
            .iter()
            .map(|state| state.store.cached_bytes())
            .sum();
        if cached > settings.cache_bytes {
            let mut oldest: Vec<(u64, usize, TileKey)> = self
                .entries
                .iter()
                .enumerate()
                .flat_map(|(index, state)| {
                    state
                        .store
                        .kept()
                        .map(move |(left, key)| (left, index, key))
                })
                .collect();
            oldest.sort_unstable();
            let mut over = cached - settings.cache_bytes;
            for (_, index, key) in oldest {
                if over == 0 {
                    break;
                }
                let store = &mut self.entries[index].store;
                let before = store.cached_bytes();
                store.forget(key);
                over = over.saturating_sub(before - store.cached_bytes());
            }
        }
        stats.restored_tiles = self.restored;
        stats.cached_bytes = self
            .entries
            .iter()
            .map(|state| state.store.cached_bytes())
            .sum();
        stats.cached_tiles = self
            .entries
            .iter()
            .map(|state| state.store.kept_tiles() as u32)
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
        #[cfg(debug_assertions)]
        for state in &self.entries {
            state.check();
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
                            state.standing += 1;
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
                failed_reads: terrains.degraded.failed_reads,
                refused_falls: terrains.degraded.refused_falls,
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
