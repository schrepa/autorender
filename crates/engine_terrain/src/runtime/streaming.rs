use super::*;

impl Terrains {
    pub fn set_device_tiles(&mut self, tiles: Option<u32>) {
        self.device_tiles = tiles;
    }

    pub fn set_clock(&mut self, clock: Option<fn() -> f64>) {
        self.clock = clock;
    }

    pub fn loading(&self) -> bool {
        !self.loaded && self.entries.iter().any(|state| state.around.is_none())
    }

    pub fn begin_loading(&mut self) {
        self.loaded = false;
    }

    pub fn streaming(&self) -> bool {
        self.entries.iter().any(|state| !state.missing.is_empty())
    }

    pub fn work_between_frames(
        &mut self,
        settings: &TerrainSettings,
        milliseconds: f64,
        jobs: Option<&mut Jobs>,
    ) -> bool {
        let Some(clock) = self.clock else {
            return false;
        };
        let Some(mut slots) = jobs.map(JobBudget::of) else {
            return false;
        };
        let work = Work {
            slice: u64::from(settings.samples_per_frame / 4).max(1),
            clock: Some(clock),
            end: clock() + milliseconds,
            upload: std::cell::Cell::new(self.upload_left),
        };
        let frame = self.frame;
        let mut remaining = false;
        for state in &mut self.entries {
            if state.terrain.validate().is_err() || !settings.enabled {
                continue;
            }
            let mut missing: Vec<(TileKey, u32)> = std::mem::take(&mut state.missing)
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            if state.store.kept_tiles() > 0 {
                self.restored += state.store.restore(&missing, &work);
                missing.retain(|(key, _)| state.loads(key));
            }
            if missing.is_empty() && state.dispatched.is_empty() {
                continue;
            }
            let target = std::mem::take(&mut state.target);
            let keep = std::mem::take(&mut state.keep);
            let baking = load_tiles(
                state,
                &missing,
                (&target, &keep, &[]),
                &work,
                frame,
                &mut slots,
            );
            state.missing = missing
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            state.target = target;
            state.keep = keep;
            remaining |= baking || !state.missing.is_empty() || !state.dispatched.is_empty();
            if work.expired() {
                break;
            }
        }
        self.between += self.upload_left - work.upload.get();
        self.upload_left = work.upload.get();
        #[cfg(debug_assertions)]
        for state in &self.entries {
            state.check();
        }
        remaining
    }

    fn heading(&mut self, eye: Vec3, seconds: f32) -> Vec3 {
        let moved = self.last_eye.map_or(Vec3::ZERO, |last| eye - last);
        self.last_eye = Some(eye);
        self.seconds = if seconds.is_finite() {
            engine_core::math::max(seconds, 0.0)
        } else {
            0.0
        };
        if moved.length() > JUMP {
            self.velocity = Vec3::ZERO;
            self.loaded = false;
        } else if seconds > 0.0 {
            self.velocity = self.velocity.lerp(moved / seconds, 0.25);
        }
        if !self.velocity.is_finite() {
            self.velocity = Vec3::ZERO;
        }
        eye + (self.velocity * AHEAD_SECONDS).clamp_length_max(AHEAD_MAX)
    }

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
        (jobs, water): (Option<&mut Jobs>, Option<&Water>),
    ) {
        self.frame += 1;
        self.sync_surroundings();
        let mut slots = jobs.map(JobBudget::of);
        let retired = std::mem::take(&mut self.retired);
        let mut released = Vec::new();
        if let Some(slots) = slots.as_mut() {
            for id in retired.jobs {
                slots.jobs.cancel(id);
            }
            released.extend(retired.shares);
        }
        self.colors_gone.extend(retired.textures);
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
            white_water: self.stats.white_water,
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
            upload: std::cell::Cell::new(
                u64::from(settings.upload_bytes).saturating_sub(std::mem::take(&mut self.between)),
            ),
        };
        let mut budget = work.slice;
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
        let mut remaining = limit.map_or(usize::MAX, |limit| limit as usize);
        let frame = self.frame;
        let mut unbounced = 0_u32;
        for state in &mut self.entries {
            if state.terrain.validate().is_err() {
                continue;
            }
            let cutout = state.around.as_ref().map(|around| around.cutout);
            if cutout.is_some() && !settings.surroundings {
                state.drawn = false;
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
            let Plan {
                collision,
                ordered,
                target,
                keep,
                mut missing,
                waiting: waiting_tiles,
                fallback,
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
                remaining,
                waiting,
                unsettled,
                store: &state.store,
            });
            state.collision = collision;
            let kept = target.len();
            stats.dropped_tiles += (ordered - kept) as u32;
            remaining -= kept;
            for key in state.store.keys_where(|key| !keep.contains(key)) {
                state.leave(&key, frame);
            }
            if kept < ordered && state.store.len() > kept {
                let mut extra = state.store.keys_where(|key| !target.contains(key));
                extra.sort_by_key(|key| key.level);
                for key in extra.into_iter().take(state.store.len() - kept) {
                    state.leave(&key, frame);
                }
            }
            stats.waiting_tiles += waiting_tiles;
            if state.store.kept_tiles() > 0 {
                self.restored += state.store.restore(&missing, &work);
                missing.retain(|(key, _)| state.loads(key));
            }
            for tile in &named {
                state.store.wanted_in_memory(&tile.key, frame);
            }
            let named: Vec<TileKey> = named
                .iter()
                .map(|tile| tile.key)
                .filter(|key| !keep.contains(key))
                .collect();
            state.missing.clone_from(&missing);
            state.target.clone_from(&target);
            state.keep.clone_from(&keep);
            state.update_materials(settings.gpu_materials);
            if let Some(slots) = slots.as_mut() {
                share_recipe(state, slots.jobs, &mut self.stacks, &mut released);
            }
            if let Some(slots) = slots.as_mut()
                && load_tiles(
                    state,
                    &missing,
                    (&target, &keep, &named),
                    &work,
                    frame,
                    slots,
                )
            {
                stats.baking += 1;
                continue;
            }
            if let Some(slots) = slots.as_mut() {
                light(state, settings, slots, frame);
                bounce(state, settings, slots, frame);
                unbounced += u32::from(waits(state, settings));
            }
            want_colors(state, settings);
            stats.lit += u32::from(state.whole.lighting.made.is_some());
            stats.lighting += u32::from(state.whole.lighting.job.is_some());
            stats.bounced += u32::from(state.whole.bounce.made.is_some());
            stats.bouncing += u32::from(state.whole.bounce.job.is_some());
            stats.colored += u32::from(state.whole.colors.shown());
            stats.coloring += u32::from(state.whole.colors.asking());
            let missing = std::mem::take(&mut state.missing);
            state.missing = missing
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            stats.resident_tiles += state.store.len() as u32;
            stats.missing_tiles += state.missing.len() as u32;
            stats.cached_tiles += state.store.kept_tiles() as u32;
            stats.cached_bytes += state.store.cached_bytes();
            stats.prefetching_tiles += state.prefetched.len() as u32;
            stats.loading_tiles += state.dispatched.len() as u32;
            let seen = (Vec2::new(local_eye.x, local_eye.z), frame);
            if let Some(slots) = slots.as_mut() {
                shading::follow(state, settings, (&mut *slots, &work), &mut released, seen);
                settled::follow(
                    state,
                    water,
                    (slots, &work),
                    (&mut self.settling, &mut released),
                    seen,
                );
            }
            let (settled_records, settling_records) = state.settled.counts();
            stats.settled_records += settled_records;
            stats.settling_records += settling_records;
            let (shaded_tiles, shading_records) = state.shading.counts();
            stats.shaded_tiles += shaded_tiles;
            stats.shading_records += shading_records;
            let [rewatered, stale] = rewater::rewater(state, &work, &mut budget);
            stats.rewatered_tiles += rewatered;
            stats.stale_water_tiles += stale;
            state.drawn = settings.enabled && state.store.len() > 0;
            stats.fallback_tiles += fallback;
            if cutout.is_some() {
                stats.surroundings_tiles += state.store.len() as u32;
            }
        }
        self.settling.drop_untaken();
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
                            || state.settled.holds_share(&key)
                            || state.shading.holds_share(&key)
                    });
                if !held {
                    slots.jobs.unshare(key);
                }
            }
        }
        if !self.loaded
            && stats.terrains > 0
            && stats.baking == 0
            && stats.loading_tiles == 0
            && stats.fallback_tiles == 0
            && stats.lighting == 0
            && unbounced == 0
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

pub(crate) struct TerrainStreaming {
    terrains: QueryState<(Entity, &'static Terrain)>,
    focus: QueryState<Entity, With<engine_core::StreamingFocus>>,
    prefetch: QueryState<(Entity, &'static TerrainPrefetch)>,
    points: Vec<Vec3>,
    places: Vec<(Vec3, f32)>,
    reliefs: Reliefs,
}

#[derive(Default)]
struct Reliefs {
    textures: HashMap<engine_assets::TextureHandle, u128>,
}

impl Reliefs {
    fn of(&mut self, materials: &[SurfaceMaterial], assets: Option<&AssetLibrary>) -> u128 {
        let mut bytes = Vec::with_capacity(32 * materials.len());
        for look in materials {
            for value in [look.depth, look.tiling, look.height_contrast] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes.push(match look.projection {
                SurfaceProjection::Top => 0,
                SurfaceProjection::Biplanar => 1,
            });
            let map = look.material.and_then(|handle| {
                let material = assets?.material(handle)?;
                Some(material.metallic_roughness_texture()?.texture)
            });
            let content = map.map_or(0, |handle| self.texture(handle, assets));
            bytes.extend_from_slice(&content.to_le_bytes());
        }
        content_hash(&bytes)
    }

    fn texture(
        &mut self,
        handle: engine_assets::TextureHandle,
        assets: Option<&AssetLibrary>,
    ) -> u128 {
        if let Some(&content) = self.textures.get(&handle) {
            return content;
        }
        let Some(texture) = assets.and_then(|assets| assets.texture(handle)) else {
            return 0;
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&texture.width().to_le_bytes());
        bytes.extend_from_slice(&texture.height().to_le_bytes());
        for level in texture.levels() {
            bytes.extend_from_slice(level);
        }
        let content = content_hash(&bytes);
        self.textures.insert(handle, content);
        content
    }
}

impl FromWorld for TerrainStreaming {
    fn from_world(world: &mut World) -> Self {
        Self {
            terrains: world.query::<(Entity, &Terrain)>(),
            focus: world.query_filtered::<Entity, With<engine_core::StreamingFocus>>(),
            prefetch: world.query::<(Entity, &TerrainPrefetch)>(),
            points: Vec::new(),
            places: Vec::new(),
            reliefs: Reliefs::default(),
        }
    }
}

pub(crate) fn stream_terrains(world: &mut World, mut streaming: Local<TerrainStreaming>) {
    streaming.update(world);
}

pub fn stream_now(world: &mut World) {
    world
        .run_system_cached(stream_terrains)
        .expect("terrain streaming takes no input");
}

pub fn work_between_frames(world: &mut World, milliseconds: f64) -> bool {
    join(world);
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
    pub(crate) fn update(&mut self, world: &mut World) {
        let _profile = engine_profile::span("cpu", "terrain_stream");
        join(world);
        let settings = world
            .get_resource::<TerrainSettings>()
            .copied()
            .map(TerrainSettings::sanitized);
        let camera = *world.resource::<Camera>();
        let viewport = world
            .get_resource::<Viewport>()
            .copied()
            .unwrap_or_default();
        let eye = engine_core::TerrainDebug::detail_eye(world);
        let seconds = world
            .get_resource::<engine_core::FrameTime>()
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
            for (entity, prefetch) in self.prefetch.iter(world) {
                let reach = prefetch.reach;
                if reach.is_finite()
                    && reach > 0.0
                    && let Some(point) = globals.get(entity).and_then(|global| global.translation())
                {
                    self.places
                        .push((point, engine_core::math::min(reach, 8.0)));
                }
            }
        }
        let present: Vec<(Entity, Terrain)> = self
            .terrains
            .iter(world)
            .map(|(entity, terrain)| (entity, terrain.clone()))
            .collect();
        let assets = world.get_resource::<AssetLibrary>();
        let solids: Vec<(Entity, Vec<Option<Solid>>)> = present
            .iter()
            .map(|(entity, terrain)| {
                let solids = terrain.surface.assets.iter();
                (*entity, solids.map(|asset| asset.solid(assets)).collect())
            })
            .collect();
        let reliefs: Vec<(Entity, u128)> = present
            .iter()
            .map(|(entity, terrain)| (*entity, self.reliefs.of(&terrain.surface.materials, assets)))
            .collect();
        let water = world.get_resource::<Water>().copied();
        let mut jobs = world.remove_resource::<Jobs>();
        let mut cooking = world.remove_resource::<Cooking>();
        if let Some(mut terrains) = world.get_resource_mut::<Terrains>() {
            let terrains = terrains.as_mut();
            terrains.follow(present);
            terrains.stand(&solids);
            terrains.relieve(&reliefs);
            match settings {
                Some(settings) => {
                    let ahead = terrains.heading(eye, seconds);
                    terrains.stream(
                        &settings,
                        &camera,
                        (eye, ahead),
                        viewport,
                        (&self.points, &self.places),
                        (jobs.as_mut(), water.as_ref()),
                    );
                    if let (Some(jobs), Some(cooking)) = (jobs.as_mut(), cooking.as_mut()) {
                        terrains.cook(&settings, water.as_ref(), jobs, cooking);
                    }
                }
                None => terrains.stand_still(),
            }
        }
        if let Some(jobs) = jobs {
            world.insert_resource(jobs);
        }
        if let Some(cooking) = cooking {
            world.insert_resource(cooking);
        }
        ask_colors(world);
    }
}

impl Terrains {
    fn follow(&mut self, present: Vec<(Entity, Terrain)>) {
        let retired = &mut self.retired;
        self.entries.retain(|state| {
            let kept = present.iter().any(|(entity, _)| *entity == state.entity);
            if !kept {
                state.retire(retired);
            }
            kept
        });
        for (entity, terrain) in present {
            match self
                .entries
                .iter_mut()
                .find(|state| state.entity == entity && state.around.is_none())
            {
                Some(state) => {
                    if !terrain.same(&state.terrain) {
                        state.retire(&mut self.retired);
                        *state = TerrainState::new(entity, terrain);
                        self.loaded = false;
                    } else {
                        if !Arc::ptr_eq(&terrain.edits, &state.terrain.edits) {
                            state.edit(terrain.edits.clone());
                        }
                        state.terrain.rivers = terrain.rivers.clone();
                        state.terrain.surroundings = terrain.surroundings;
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
                    self.entries.push(TerrainState::new(entity, terrain));
                    self.loaded = false;
                }
            }
        }
    }

    fn stand(&mut self, solids: &[(Entity, Vec<Option<Solid>>)]) {
        for state in &mut self.entries {
            if let Some((_, found)) = solids.iter().find(|(entity, _)| *entity == state.entity)
                && state.solids[..] != found[..]
            {
                state.solids = Arc::from(found.as_slice());
            }
        }
    }

    fn relieve(&mut self, reliefs: &[(Entity, u128)]) {
        for state in &mut self.entries {
            if let Some(&(_, relief)) = reliefs.iter().find(|(entity, _)| *entity == state.entity) {
                state.relief = relief;
            }
        }
    }

    fn stand_still(&mut self) {
        for state in &mut self.entries {
            state.drawn = false;
        }
        self.stats = TerrainStats {
            terrains: self
                .entries
                .iter()
                .filter(|state| state.around.is_none())
                .count() as u32,
            white_water: self.stats.white_water,
            ..TerrainStats::default()
        };
        self.loaded = true;
    }
}
