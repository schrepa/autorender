//! Optional finite-volume fog in generated terrain depressions.

use engine_math as math;
use std::collections::{BTreeMap, HashMap};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use bevy_ecs::prelude::{Entity, Resource, World};
use engine_settings::settings;
use engine_terrain::{BasinGrid, BasinJob, Layers, Pyramid};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

settings! {
    /// An artistic fog medium with a finite volume, not a weather forecast.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema), schemars(rename = "SceneBasinFog"))]
    #[serde(default, deny_unknown_fields)]
    pub struct BasinFog {
        /// Whether the optional medium is enabled.
        pub enabled: bool = true => any,
        /// Cubic meters supplied per square meter of valid terrain.
        pub budget_m: f32 = 2.0 => at_least(0.0),
        /// Optical extinction per meter through full-density fog.
        pub extinction_per_m: f32 = 0.025 => at_least(0.0),
        /// The vertical density ramp below each horizontal pool top.
        pub top_fade_m: f32 = 3.0 => at_least(0.0),
        /// Fixed terrain pyramid level used by the volume solver.
        pub level: u32 = 4 => range(0, engine_terrain::MAX_LEVEL),
        /// Workspace, retained maps and upload replacements; omitted is unbounded.
        pub max_bytes: Option<u64> = None => at_least(0.0),
    }
}

impl BasinFog {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// Host-supplied storage capacity for optional fog data, after shared
/// binding prefixes. Absent on a headless engine means no device limit.
#[derive(Resource, Clone, Copy)]
pub struct BasinFogDeviceLimit(pub u64);

/// A ready terrain medium shared with the renderer by immutable revision.
#[derive(Clone, Debug)]
pub struct RenderBasinFog {
    pub entity: Entity,
    pub origin: Vec3,
    pub revision: u64,
    pub grid: Arc<BasinGrid>,
    pub uploaded: Arc<AtomicU64>,
}

#[derive(Debug)]
struct Entry {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    level: u32,
    water: bool,
    sea: Option<f32>,
    budget: f32,
    origin: Vec3,
    revision: u64,
    source_revision: u64,
    requested: u64,
    job: BasinJob,
    grid: Option<Arc<BasinGrid>>,
    uploaded: Arc<AtomicU64>,
    summary: Option<BasinTerrainView>,
}

/// One connected wet pool's geometric summary.
#[derive(Clone, Debug, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BasinPoolView {
    pub id: u32,
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub top: f32,
    pub volume_m3: f64,
    pub deepest: [f32; 3],
    pub depth_m: f32,
}

/// Current immutable solution and upload status for a terrain.
#[derive(Clone, Debug, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BasinTerrainView {
    pub terrain: u64,
    pub state: String,
    pub origin: [f32; 3],
    pub actual_level: u32,
    pub source_revision: u64,
    pub solve_revision: u64,
    pub upload_revision: u64,
    pub size: [u32; 2],
    pub spacing: f32,
    pub dem_hash: String,
    pub wet_cells: u64,
    pub pools: u64,
    pub lid_range: Option<[f32; 2]>,
    pub largest: Option<BasinPoolView>,
    pub requested_bytes: u64,
    pub map_bytes: u64,
    pub workspace_bytes: u64,
    pub input_m3: f64,
    pub retained_m3: f64,
    pub drained_m3: f64,
    pub published_retained_m3: f64,
    pub rounding_m3: f64,
}

/// Basin-fog preparation, admission and upload observations.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BasinFogView {
    pub settings: BasinFog,
    pub pending_jobs: u64,
    pub pending_uploads: u64,
    pub dropped_terrains: u64,
    pub unsupported_terrains: u64,
    pub requested_bytes: u64,
    pub terrains: Vec<BasinTerrainView>,
}

/// CPU ownership of optional immutable terrain-fog solutions. An off
/// resource retains no terrain entries, map allocations or jobs.
#[derive(Resource, Default)]
pub struct BasinFogs {
    entries: HashMap<Entity, Entry>,
    skipped: Vec<BasinTerrainView>,
    settings: Option<BasinFog>,
    serial: u64,
}

impl BasinFogs {
    pub fn loading(&self) -> bool {
        self.skipped.iter().any(|s| s.state == "pending")
            || self
                .entries
                .values()
                .any(|e| e.grid.is_none() || e.uploaded.load(Ordering::Relaxed) != e.revision)
    }

    pub fn render(&self, output: &mut Vec<RenderBasinFog>) {
        output.clear();
        for (&entity, entry) in &self.entries {
            if let Some(grid) = &entry.grid {
                output.push(RenderBasinFog {
                    entity,
                    origin: entry.origin,
                    revision: entry.revision,
                    grid: grid.clone(),
                    uploaded: entry.uploaded.clone(),
                });
            }
        }
        output.sort_by_key(|e| e.entity.to_bits());
    }

    pub fn sample(&self, terrain: Entity, point: Vec2) -> Option<(bool, [f32; 3], u32)> {
        let entry = self.entries.get(&terrain)?;
        let grid = entry.grid.as_ref()?;
        let local = point - Vec2::new(entry.origin.x, entry.origin.z);
        if !local.is_finite()
            || local.x < 0.0
            || local.y < 0.0
            || local.x >= grid.extent[0]
            || local.y >= grid.extent[1]
        {
            return None;
        }
        let at = (local.y / grid.spacing) as usize * grid.size[0] as usize
            + (local.x / grid.spacing) as usize;
        let floor = grid.floors[at];
        let top = grid.tops[at];
        Some((
            grid.valid[at],
            [floor + entry.origin.y, top + entry.origin.y, top - floor],
            grid.pools[at],
        ))
    }

    pub fn view(&self) -> BasinFogView {
        let mut terrains = self.skipped.clone();
        let mut pending_jobs = 0;
        let mut pending_uploads = 0;
        for (&entity, entry) in &self.entries {
            let uploaded = entry.uploaded.load(Ordering::Relaxed);
            let mut view = entry.summary.clone().unwrap_or_else(|| BasinTerrainView {
                terrain: entity.to_bits(),
                actual_level: entry.level,
                source_revision: entry.source_revision,
                requested_bytes: entry.requested,
                ..Default::default()
            });
            view.origin = entry.origin.to_array();
            view.workspace_bytes = entry.job.bytes();
            view.upload_revision = uploaded;
            view.state = if entry.grid.is_none() {
                pending_jobs += 1;
                "pending"
            } else if uploaded != entry.revision {
                pending_uploads += 1;
                "uploading"
            } else {
                "ready"
            }
            .into();
            terrains.push(view);
        }
        terrains.sort_by_key(|e| e.terrain);
        BasinFogView {
            settings: self.settings.unwrap_or_else(BasinFog::off),
            pending_jobs,
            pending_uploads,
            dropped_terrains: terrains.iter().filter(|e| e.state == "dropped").count() as u64,
            unsupported_terrains: terrains.iter().filter(|e| e.state == "unsupported").count()
                as u64,
            requested_bytes: terrains.iter().map(|e| e.requested_bytes).sum(),
            terrains,
        }
    }
}

fn summarize(entity: Entity, entry: &Entry, grid: &BasinGrid) -> BasinTerrainView {
    let mut pools = BTreeMap::<u32, BasinPoolView>::new();
    let mut wet_cells = 0;
    for (i, &id) in grid.pools.iter().enumerate() {
        if id == 0 {
            continue;
        }
        wet_cells += 1;
        let x = (i as u32 % grid.size[0]) as f32 * grid.spacing;
        let z = (i as u32 / grid.size[0]) as f32 * grid.spacing;
        let high = [
            math::min(x + grid.spacing, grid.extent[0]),
            math::min(z + grid.spacing, grid.extent[1]),
        ];
        let depth = grid.tops[i] - grid.floors[i];
        let pool = pools.entry(id).or_insert(BasinPoolView {
            id,
            min: [x, z],
            max: high,
            top: grid.tops[i],
            ..Default::default()
        });
        pool.min = [math::min(pool.min[0], x), math::min(pool.min[1], z)];
        pool.max = [
            math::max(pool.max[0], high[0]),
            math::max(pool.max[1], high[1]),
        ];
        pool.volume_m3 += f64::from(high[0] - x)
            * f64::from(high[1] - z)
            * (f64::from(grid.tops[i]) - f64::from(grid.floors[i]));
        if depth > pool.depth_m {
            pool.depth_m = depth;
            pool.deepest = [
                x + grid.spacing * 0.5,
                grid.floors[i],
                z + grid.spacing * 0.5,
            ];
        }
    }
    let lid_range = pools
        .values()
        .map(|p| p.top)
        .fold(None::<[f32; 2]>, |range, top| {
            Some(range.map_or([top, top], |[low, high]| {
                [math::min(low, top), math::max(high, top)]
            }))
        });
    let mut hash = 0xcbf29ce484222325u64;
    for (floor, valid) in grid.floors.iter().zip(&grid.valid) {
        for byte in floor
            .to_bits()
            .to_le_bytes()
            .into_iter()
            .chain([u8::from(*valid)])
        {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    BasinTerrainView {
        terrain: entity.to_bits(),
        origin: entry.origin.to_array(),
        actual_level: grid.level,
        dem_hash: format!("fnv1a64:{hash:016x}"),
        source_revision: entry.source_revision,
        solve_revision: entry.revision,
        size: grid.size,
        spacing: grid.spacing,
        wet_cells,
        pools: pools.len() as u64,
        lid_range,
        largest: pools
            .into_values()
            .max_by(|a, b| a.volume_m3.total_cmp(&b.volume_m3).then(b.id.cmp(&a.id))),
        requested_bytes: entry.requested,
        map_bytes: grid.bytes(),
        workspace_bytes: entry.job.bytes(),
        input_m3: grid.input_m3,
        retained_m3: grid.retained_m3,
        drained_m3: grid.drained_m3,
        published_retained_m3: grid.published_retained_m3,
        rounding_m3: grid.published_retained_m3 - grid.retained_m3,
        ..Default::default()
    }
}

pub(crate) fn sync(world: &mut World) {
    let settings = world
        .get_resource::<BasinFog>()
        .copied()
        .unwrap_or_else(BasinFog::off)
        .sanitized();
    if !settings.enabled || settings.budget_m == 0.0 {
        if let Some(mut state) = world.get_resource_mut::<BasinFogs>() {
            state.entries = HashMap::new();
            state.skipped = Vec::new();
            state.settings = Some(settings);
        }
        return;
    }
    let mut state = world.remove_resource::<BasinFogs>().unwrap_or_default();
    state.settings = Some(settings);
    state.skipped.clear();
    let eye = world
        .get_resource::<crate::Camera>()
        .map_or(Vec3::ZERO, |c| c.eye());
    let mut terrains: Vec<_> = world
        .query::<(Entity, &crate::Terrain)>()
        .iter(world)
        .map(|(e, t)| (e, t.clone()))
        .collect();
    terrains.sort_by(|(a, ta), (b, tb)| {
        let distance = |t: &crate::Terrain| {
            let local = eye - t.origin;
            let high = t.pyramid.extent();
            let closest = Vec3::new(
                math::clamp(local.x, 0.0, high.x),
                local.y,
                math::clamp(local.z, 0.0, high.y),
            );
            (local - closest).length_squared()
        };
        distance(ta)
            .total_cmp(&distance(tb))
            .then(a.to_bits().cmp(&b.to_bits()))
    });
    state
        .entries
        .retain(|e, _| terrains.iter().any(|(id, _)| e == id));
    let water = world
        .get_resource::<crate::Water>()
        .is_some_and(|w| w.enabled);
    let ocean = world
        .get_resource::<crate::Ocean>()
        .filter(|o| o.enabled)
        .map(|o| o.level);
    let device = world
        .get_resource::<BasinFogDeviceLimit>()
        .map_or(u64::MAX, |limit| limit.0);
    let mut remaining = settings.max_bytes.unwrap_or(u64::MAX);
    let mut gpu_remaining = device;
    let terrain_settings = world
        .get_resource::<crate::TerrainSettings>()
        .copied()
        .unwrap_or_default()
        .sanitized();
    let clock = world
        .get_resource::<crate::Terrains>()
        .and_then(crate::Terrains::work_clock);
    let until = clock.map(|now| now() + f64::from(terrain_settings.loading_work_ms));
    let mut work_left = u64::from(terrain_settings.samples_per_frame);
    for (entity, terrain) in terrains {
        let level = settings.level.min(terrain.pyramid.levels() - 1);
        let requested = BasinJob::requested_bytes(&terrain.pyramid, level).unwrap_or(u64::MAX);
        let gpu = requested
            .checked_div(1024)
            .and_then(|n| n.checked_mul(96))
            .and_then(|n| n.checked_add(64))
            .unwrap_or(u64::MAX);
        if !matches!(terrain.source, crate::TerrainSource::Generated(_)) {
            skip(
                &mut state,
                entity,
                &terrain,
                level,
                requested,
                "unsupported",
            );
            continue;
        }
        if requested > remaining || gpu > gpu_remaining {
            skip(&mut state, entity, &terrain, level, requested, "dropped");
            continue;
        }
        let layers = world
            .get_resource_mut::<crate::Terrains>()
            .and_then(|mut terrains| terrains.completed_layers(entity));
        let Some(layers) = layers.filter(|layers| layers.unbaked().is_none()) else {
            skip(&mut state, entity, &terrain, level, requested, "pending");
            continue;
        };
        remaining -= requested;
        gpu_remaining -= gpu;
        let sea = ocean.map(|height| height - terrain.origin.y);
        let matches = state.entries.get(&entity).is_some_and(|e| {
            Arc::ptr_eq(&e.layers, &layers)
                && e.pyramid == terrain.pyramid
                && e.level == level
                && e.water == water
                && e.sea == sea
        });
        if !matches {
            state.serial = state.serial.wrapping_add(1);
            let Some(job) = BasinJob::new(
                layers.clone(),
                terrain.pyramid,
                level,
                settings.budget_m,
                water,
                sea,
            ) else {
                skip(&mut state, entity, &terrain, level, requested, "dropped");
                continue;
            };
            state.entries.insert(
                entity,
                Entry {
                    layers,
                    pyramid: terrain.pyramid,
                    level,
                    water,
                    sea,
                    budget: settings.budget_m,
                    origin: terrain.origin,
                    revision: state.serial,
                    source_revision: state.serial,
                    requested,
                    job,
                    grid: None,
                    uploaded: Arc::new(AtomicU64::new(0)),
                    summary: None,
                },
            );
        }
        let entry = state.entries.get_mut(&entity).expect("admitted");
        if entry.origin != terrain.origin {
            state.serial = state.serial.wrapping_add(1);
            entry.revision = state.serial;
            entry.uploaded.store(0, Ordering::Relaxed);
            entry.origin = terrain.origin;
        }
        if entry.budget != settings.budget_m {
            state.serial = state.serial.wrapping_add(1);
            entry.revision = state.serial;
            entry.grid = None;
            entry.summary = None;
            let _profile = engine_profile::span("cpu", "basin_fog_fill");
            if !entry.job.refill(settings.budget_m) {
                entry.job = BasinJob::new(
                    entry.layers.clone(),
                    entry.pyramid,
                    entry.level,
                    settings.budget_m,
                    entry.water,
                    entry.sea,
                )
                .expect("previously admitted dimensions");
            }
            entry.budget = settings.budget_m;
        }
        if entry.grid.is_none() {
            let _profile = engine_profile::span("cpu", "basin_fog_topology");
            while work_left > 0 && clock.zip(until).is_none_or(|(now, until)| now() < until) {
                let slice = work_left.min(256);
                work_left -= slice;
                if entry.job.step(slice) {
                    let grid = Arc::new(entry.job.grid().expect("finished").clone());
                    entry.summary = Some(summarize(entity, entry, &grid));
                    entry.grid = Some(grid);
                    break;
                }
            }
        }
    }
    world.insert_resource(state);
}

fn skip(
    state: &mut BasinFogs,
    entity: Entity,
    terrain: &crate::Terrain,
    level: u32,
    requested: u64,
    reason: &str,
) {
    state.entries.remove(&entity);
    state.skipped.push(BasinTerrainView {
        terrain: entity.to_bits(),
        origin: terrain.origin.to_array(),
        actual_level: level,
        state: reason.into(),
        requested_bytes: requested,
        ..Default::default()
    });
}
