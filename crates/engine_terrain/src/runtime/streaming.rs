use super::*;
use engine_core::{DeviceLimits, FrameBudget, FrameTime, GlobalTransforms, Joined, Views};
use engine_open_world_kit::terrain::TerrainLayers;
use engine_places_world::streaming::StreamingFocus;
use glam::UVec2;

pub(super) fn vertical_fov(camera: &Camera) -> f32 {
    camera.vertical_fov().unwrap_or(f32::NAN)
}

fn device_tiles(limits: &DeviceLimits) -> Option<u32> {
    (limits.texture_layers > 0).then_some(limits.texture_layers)
}

impl Terrains {
    pub fn streaming(&self) -> bool {
        self.entries.iter().any(|state| !state.missing.is_empty())
    }

    fn heading(&mut self, eye: Vec3, seconds: f32) -> Vec3 {
        let moved = self.last_eye.map_or(Vec3::ZERO, |last| eye - last);
        self.last_eye = Some(eye);
        self.seconds = if seconds.is_finite() {
            engine_math::max(seconds, 0.0)
        } else {
            0.0
        };
        if moved.length() > JUMP {
            self.velocity = Vec3::ZERO;
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
        (camera, looking): (&Camera, Vec3),
        (eye, ahead): (Vec3, Vec3),
        (size, device_tiles): (UVec2, Option<u32>),
        (focus, places): (&[Vec3], &[(Vec3, f32)]),
        (asking, budget): (&mut Asking<'_, '_, '_>, &mut FrameBudget),
    ) {
        self.frame += 1;
        self.sync_surroundings(asking.kinds);
        let mut released = std::mem::take(&mut self.retired.shares);
        let mut missing_left = 0_u32;
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
            ..TerrainStats::default()
        };
        let forward = Vec2::new(looking.x, looking.z);
        let planes = ViewPlanes::new(camera, (eye, looking), size);
        let fov = vertical_fov(camera);
        let fov = if fov.is_finite() && fov > 0.0 && fov < std::f32::consts::PI {
            fov
        } else {
            vertical_fov(&Camera::default())
        };
        let work = Work::frame(budget);
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
        let limit = match (settings.max_tiles, device_tiles) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let mut remaining = limit.map_or(usize::MAX, |limit| limit as usize);
        let frame = self.frame;
        for state in &mut self.entries {
            if state.terrain.validate().is_err() {
                continue;
            }
            let cutout = state.around.as_ref().map(|around| around.cutout);
            if cutout.is_some() && !settings.surroundings {
                state.show(false);
                continue;
            }
            let pyramid = state.terrain.pyramid;
            let origin = state.terrain.origin;
            let ranges = LodRanges::new(&pyramid, settings.pixel_error, size.y as f32, fov);
            if !(0..pyramid.levels())
                .map(|level| ranges.range(level))
                .eq(state.ranges.iter().copied())
            {
                state.ranges.clear();
                state
                    .ranges
                    .extend((0..pyramid.levels()).map(|level| ranges.range(level)));
                state.changed = true;
            }
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
            stats.tile_cut.asked += ordered as u32;
            stats.tile_cut.kept += kept as u32;
            stats.tile_cut.cut += (ordered - kept) as u32;
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
            share_recipe(state, asking.jobs, &mut self.stacks, &mut released);
            if load_tiles(
                state,
                &missing,
                (&target, &keep, &named),
                &work,
                frame,
                asking,
            ) {
                stats.baking += u32::from(state.baking());
                continue;
            }
            let missing = std::mem::take(&mut state.missing);
            state.missing = missing
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            stats.resident_tiles += state.store.len() as u32;
            stats.missing_tiles += state.missing.len() as u32;
            missing_left += state
                .missing
                .iter()
                .filter(|(key, _)| !state.failed(key))
                .count() as u32;
            stats.cached_tiles += state.store.kept_tiles() as u32;
            stats.cached_bytes += state.store.cached_bytes();
            stats.prefetching_tiles += state.prefetching() as u32;
            stats.loading_tiles += state.loading() as u32;
            state.show(state.store.len() > 0);
            stats.fallback_tiles += fallback;
            if cutout.is_some() {
                stats.surroundings_tiles += state.store.len() as u32;
            }
        }
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
        release(&self.entries, &mut self.stacks, released, asking.jobs);
        self.settling.left(
            missing_left
                .saturating_add(stats.baking)
                .saturating_add(stats.waiting_tiles),
        );
        stats.dropped_tiles = stats.tile_cut.cut;
        let (failed_jobs, latest) = self.job_failures();
        stats.failed_jobs = failed_jobs;
        stats.latest_job_failure = latest.map(str::to_owned);
        self.stats = stats;
        #[cfg(debug_assertions)]
        for state in &self.entries {
            state.check();
        }
    }
}

#[derive(Default)]
pub(crate) struct TerrainStreaming {
    points: Vec<Vec3>,
    places: Vec<(Vec3, f32)>,
}

pub(crate) type StreamReads<'w> = (
    Res<'w, Views>,
    Res<'w, FrameTime>,
    Res<'w, DeviceLimits>,
    Res<'w, GlobalTransforms>,
    Res<'w, Records>,
);

#[expect(
    clippy::too_many_arguments,
    reason = "a system's parameters are its reads"
)]
pub(crate) fn stream_terrains(
    settings: Res<TerrainSettings>,
    (views, time, limits, globals, records): StreamReads<'_>,
    kinds: Joined<TerrainLayers>,
    focus: Query<Entity, With<StreamingFocus>>,
    prefetch: Query<(Entity, &TerrainPrefetch)>,
    scenes: Query<(Entity, &TerrainScene)>,
    mut jobs: Jobs,
    mut terrains: ResMut<Terrains>,
    mut stats: ResMut<TerrainStats>,
    mut budget: ResMut<FrameBudget>,
    mut streaming: Local<TerrainStreaming>,
) {
    let _profile = engine_core::profile::span("cpu", "terrain_stream");
    let streaming = &mut *streaming;
    let kinds: Vec<&'static LayerKind> = kinds.iter().copied().collect();
    let main = views.main();
    let eye = main.map_or(Vec3::ZERO, |main| main.detail_eye);
    let seconds = time.real_seconds as f32;
    streaming.points.clear();
    streaming
        .points
        .extend(views.listed().iter().map(|facts| facts.detail_eye));
    streaming.places.clear();
    for entity in &focus {
        if let Some(point) = globals.get(entity).and_then(|global| global.translation()) {
            streaming.points.push(point);
        }
    }
    for (entity, prefetch) in &prefetch {
        let reach = prefetch.reach;
        if reach.is_finite()
            && reach > 0.0
            && let Some(point) = globals.get(entity).and_then(|global| global.translation())
        {
            streaming.places.push((point, engine_math::min(reach, 8.0)));
        }
    }
    let present: Vec<(Entity, Terrain)> = scenes
        .iter()
        .filter_map(|(entity, scene)| Some((entity, scene.built.clone()?)))
        .collect();
    let terrains = terrains.as_mut();
    terrains.follow(present, &kinds);
    let ahead = terrains.heading(eye, seconds);
    let mut asking = Asking {
        jobs: &mut jobs,
        records: &records,
        kinds: &kinds,
    };
    terrains.stream(
        &settings,
        (
            &main.map(|main| main.lens).unwrap_or_default(),
            main.map_or(Vec3::NEG_Z, |main| main.forward),
        ),
        (eye, ahead),
        (
            main.map_or(views.frame().display_size, |main| main.size()),
            device_tiles(&limits),
        ),
        (&streaming.points, &streaming.places),
        (&mut asking, &mut *budget),
    );
    stats.set_if_neq(terrains.stats.clone());
}

impl Terrains {
    fn follow(&mut self, present: Vec<(Entity, Terrain)>, kinds: &[&'static LayerKind]) {
        let mut kept = Vec::with_capacity(self.entries.len());
        for state in std::mem::take(&mut self.entries) {
            if present.iter().any(|(entity, _)| *entity == state.entity) {
                kept.push(state);
            } else {
                state.retire(&mut self.retired);
                self.changed = true;
            }
        }
        self.entries = kept;
        for (entity, terrain) in present {
            match self
                .entries
                .iter_mut()
                .find(|state| state.entity == entity && state.around.is_none())
            {
                Some(state) => {
                    if !terrain.same(&state.terrain) {
                        let old =
                            std::mem::replace(state, TerrainState::new(entity, terrain, kinds));
                        old.retire(&mut self.retired);
                    } else {
                        if !Arc::ptr_eq(&terrain.edits, &state.terrain.edits) {
                            state.edit(terrain.edits.clone(), kinds);
                        }
                        state.terrain.rivers = terrain.rivers.clone();
                        state.terrain.surroundings = terrain.surroundings;
                        if terrain.surface.assets != state.terrain.surface.assets
                            || terrain.surface.bridge != state.terrain.surface.bridge
                        {
                            state.terrain.surface.assets = terrain.surface.assets;
                            state.terrain.surface.bridge = terrain.surface.bridge;
                            state.ground_changed = true;
                        }
                    }
                }
                None => {
                    self.entries.push(TerrainState::new(entity, terrain, kinds));
                }
            }
        }
    }
}
