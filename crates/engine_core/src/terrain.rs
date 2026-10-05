//! Terrain (decision 051): heightfields whose tiles stream around the camera,
//! drawn in continuous level of detail and sampled by queries and collision.
//!
//! An entity's [`Terrain`] describes one: where it lies, its tile pyramid,
//! where its tiles come from (generated from a layer stack, or read from an
//! `ETR1` file by byte ranges) and its material. The [`Terrains`] resource
//! keeps every terrain's resident tiles. Each extraction streams toward the
//! camera: the coarsest levels stay resident everywhere, so the whole
//! terrain always draws; finer levels load nearest and ahead of the camera
//! first, and level-0 tiles load around the camera and [`TerrainFocus`]
//! entities for collision. Generating and decoding share a per-frame work
//! budget; past a tile budget or the device's capacity the least important
//! tiles drop and count as dropped. Frames list each terrain's resident
//! tiles and the patches the camera draws in [`RenderSnapshot::terrain`].
//!
//! Queries ([`Terrains::sample`], [`Terrains::height_at`] and the rest)
//! answer from the finest resident tile at a point, on the triangles
//! renderers and collision use, and return `None` where no tile is
//! resident.
//!
//! The [`TerrainSettings`] are inserted by default; without them, or when
//! disabled, terrains neither stream nor draw.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy_ecs::{
    prelude::{Component, Entity, Resource, With, World},
    query::QueryState,
};
use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use engine_terrain::{
    FileHeader, FormatError, Generator, Layers, LodRanges, Patch, Pyramid, Selection, TILE_STRIDE,
    Tile, TileKey, TileSample, WantedTile, decode_tile,
};

use crate::environment::{InvalidSetting, check_budget, check_range};
use crate::{Camera, GlobalTransforms, RenderSnapshot};

/// Where a terrain's tiles come from.
#[derive(Clone, Debug, PartialEq)]
pub enum TerrainSource {
    /// Generated in the engine from a layer stack, a tile at a time.
    Generated(Arc<Layers>),
    /// Read from an `ETR1` file: its header and directory, and the URL
    /// hosts fetch its tiles' byte ranges from, or the whole file, which
    /// tiles then decode from without fetching.
    File {
        header: Arc<FileHeader>,
        url: Arc<str>,
        data: Option<Arc<[u8]>>,
    },
}

/// A terrain's procedural material: rock where slopes steepen, snow above a
/// height on gentler slopes, and grass below, broken up by noise across the
/// ground. Colors are linear.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneTerrainMaterial")
)]
#[serde(default, deny_unknown_fields)]
pub struct TerrainMaterial {
    pub grass: [f32; 3],
    /// Bare ground between grass and rock.
    pub dirt: [f32; 3],
    pub rock: [f32; 3],
    pub snow: [f32; 3],
    /// Degrees from level where rock takes over, 0 to 90.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 90.0)))]
    pub rock_slope: f32,
    /// Meters above the terrain's origin where snow starts.
    pub snow_height: f32,
    /// Surface roughness, 0.03 to 1.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.03, max = 1.0)))]
    pub roughness: f32,
}

impl Default for TerrainMaterial {
    fn default() -> Self {
        Self {
            grass: [0.105, 0.135, 0.045],
            dirt: [0.21, 0.165, 0.11],
            rock: [0.25, 0.235, 0.22],
            snow: [0.82, 0.84, 0.88],
            rock_slope: 34.0,
            snow_height: 650.0,
            roughness: 0.92,
        }
    }
}

impl TerrainMaterial {
    /// Whether every value is in range.
    pub fn is_usable(&self) -> bool {
        let color = |c: [f32; 3]| c.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v));
        color(self.grass)
            && color(self.dirt)
            && color(self.rock)
            && color(self.snow)
            && self.rock_slope.is_finite()
            && (0.0..=90.0).contains(&self.rock_slope)
            && self.snow_height.is_finite()
            && self.roughness.is_finite()
            && (0.03..=1.0).contains(&self.roughness)
    }
}

/// A heightfield terrain (decision 051) whose first corner (least x and z)
/// lies at `origin`, heights measured up from it. Its pyramid sets its
/// level-0 tiles and spacing; its source gives its tiles.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Terrain {
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub source: TerrainSource,
    pub material: TerrainMaterial,
    /// Whether it collides where collision is installed.
    pub collision: bool,
}

impl Terrain {
    /// Whether `other` describes the same terrain: equal fields and the
    /// same shared source, so streaming keeps its tiles.
    fn same(&self, other: &Self) -> bool {
        let source = match (&self.source, &other.source) {
            (TerrainSource::Generated(a), TerrainSource::Generated(b)) => Arc::ptr_eq(a, b),
            (
                TerrainSource::File {
                    header: a, url: u, ..
                },
                TerrainSource::File {
                    header: b, url: v, ..
                },
            ) => Arc::ptr_eq(a, b) && u == v,
            _ => false,
        };
        source
            && self.origin == other.origin
            && self.pyramid == other.pyramid
            && self.material == other.material
            && self.collision == other.collision
    }

    /// A terrain generated from `layers`.
    pub fn generated(origin: Vec3, pyramid: Pyramid, layers: Layers) -> Self {
        Self {
            origin,
            pyramid,
            source: TerrainSource::Generated(Arc::new(layers)),
            material: TerrainMaterial::default(),
            collision: true,
        }
    }

    /// A terrain read from the `ETR1` file at `url` whose prefix parsed into
    /// `header`.
    pub fn from_file(origin: Vec3, header: FileHeader, url: impl Into<Arc<str>>) -> Self {
        Self {
            origin,
            pyramid: *header.pyramid(),
            source: TerrainSource::File {
                header: Arc::new(header),
                url: url.into(),
                data: None,
            },
            material: TerrainMaterial::default(),
            collision: true,
        }
    }

    /// A terrain read from a whole `ETR1` file in memory.
    pub fn from_file_bytes(origin: Vec3, bytes: Arc<[u8]>) -> Result<Self, FormatError> {
        let header = FileHeader::parse(&bytes)?;
        Ok(Self {
            origin,
            pyramid: *header.pyramid(),
            source: TerrainSource::File {
                header: Arc::new(header),
                url: Arc::from(""),
                data: Some(bytes),
            },
            material: TerrainMaterial::default(),
            collision: true,
        })
    }

    pub fn with_material(mut self, material: TerrainMaterial) -> Self {
        self.material = material;
        self
    }

    pub fn with_collision(mut self, collision: bool) -> Self {
        self.collision = collision;
        self
    }

    /// Whether streaming can use this terrain.
    pub fn is_usable(&self) -> bool {
        self.origin.is_finite() && self.material.is_usable()
    }
}

/// What terrain shows (decision 051): its material, or for development its
/// levels of detail, normals or tiles and patches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TerrainView {
    #[default]
    Shaded,
    Levels,
    Normals,
    Tiles,
}

/// Temporary terrain state for development (decision 051): an eye that
/// streaming and level of detail follow instead of the camera's, so agents
/// can look at the patches it selects from elsewhere, and what terrain
/// shows. Absent, terrain follows the camera, shaded.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainDebug {
    pub freeze: Option<Vec3>,
    pub view: TerrainView,
}

/// Keeps level-0 terrain tiles resident around its entity, as the camera
/// does, so collision and queries hold there.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainFocus;

/// The size in pixels of the viewport frames render to, which terrain level
/// of detail measures pixels on; hosts set it each frame. Without it,
/// 1920x1080.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
        }
    }
}

/// Terrain settings (decision 051). Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneTerrainSettings")
)]
#[serde(default, deny_unknown_fields)]
pub struct TerrainSettings {
    pub enabled: bool,
    /// Pixels a level's samples span before the next coarser level takes
    /// over, 1 to 64; smaller draws more triangles.
    #[cfg_attr(feature = "schema", schemars(range(min = 1.0, max = 64.0)))]
    pub pixel_error: f32,
    /// The coarsest levels kept resident over the whole terrain, so it
    /// always draws whole, from 1 to 8.
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 8)))]
    pub pinned_levels: u32,
    /// An optional budget of resident tiles (about 270 KB each, twice that
    /// on the GPU); the least important beyond it drop.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tiles: Option<u32>,
    /// Work generated or decoded per frame, in samples' worth (67,000 a
    /// decoded tile, about 500,000 a generated one).
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    pub samples_per_frame: u32,
    /// Meters around the camera and focus entities within which level-0
    /// tiles load for collision, 0 to 4096.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4096.0)))]
    pub collision_radius: f32,
    /// Long sun shadows through each terrain's horizon map (decision 052),
    /// with shadows on; shadow maps then draw terrain near the camera only.
    /// Off, terrain casts into the shadow maps alone.
    pub sun_shadows: bool,
    /// Texels along a horizon map's sides, 64 to 4096 (8 bytes each).
    #[cfg_attr(feature = "schema", schemars(range(min = 64, max = 4096)))]
    pub shadow_resolution: u32,
    /// Shadow edge height per meter to the slope casting it (the sun's disc
    /// gives 0.0093).
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 0.2)))]
    pub shadow_softness: f32,
}

impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 6.0,
            pinned_levels: 2,
            max_tiles: None,
            samples_per_frame: 24_576,
            collision_radius: 192.0,
            sun_shadows: true,
            shadow_resolution: 1024,
            shadow_softness: 0.01,
        }
    }
}

impl TerrainSettings {
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("pixel_error", self.pixel_error, 1.0..=64.0)?;
        if !(1..=8).contains(&self.pinned_levels) {
            return Err(InvalidSetting::new("pinned_levels", "must be in [1, 8]"));
        }
        check_budget("max_tiles", self.max_tiles)?;
        if self.samples_per_frame == 0 {
            return Err(InvalidSetting::new(
                "samples_per_frame",
                "must be at least 1",
            ));
        }
        check_range("collision_radius", self.collision_radius, 0.0..=4096.0)?;
        if !(64..=4096).contains(&self.shadow_resolution) {
            return Err(InvalidSetting::new(
                "shadow_resolution",
                "must be in [64, 4096]",
            ));
        }
        check_range("shadow_softness", self.shadow_softness, 0.0..=0.2)
    }

    /// Every value in range.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let finite = |value: f32, fallback: f32| if value.is_finite() { value } else { fallback };
        Self {
            enabled: self.enabled,
            pixel_error: crate::math::clamp(
                finite(self.pixel_error, defaults.pixel_error),
                1.0,
                64.0,
            ),
            pinned_levels: self.pinned_levels.clamp(1, 8),
            max_tiles: self.max_tiles.map(|budget| budget.max(1)),
            samples_per_frame: self.samples_per_frame.max(1),
            collision_radius: crate::math::clamp(
                finite(self.collision_radius, defaults.collision_radius),
                0.0,
                4096.0,
            ),
            sun_shadows: self.sun_shadows,
            shadow_resolution: self.shadow_resolution.clamp(64, 4096),
            shadow_softness: crate::math::clamp(
                finite(self.shadow_softness, defaults.shadow_softness),
                0.0,
                0.2,
            ),
        }
    }
}

/// The surface of a terrain at a point: its terrain, world height, the
/// normal of its triangle, whether its cell is a hole, and the level of the
/// tile it came from (0 is the finest).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSample {
    pub terrain: Entity,
    pub height: f32,
    pub normal: Vec3,
    pub hole: bool,
    pub level: u32,
}

/// A byte range of a terrain file a host should fetch and hand back with
/// [`Terrains::deliver`].
#[derive(Clone, Debug, PartialEq)]
pub struct TileRequest {
    pub terrain: Entity,
    pub key: TileKey,
    pub url: Arc<str>,
    pub offset: u64,
    pub length: u32,
}

/// Counters of the last streaming update, over every terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TerrainStats {
    pub terrains: u32,
    /// Tiles resident now.
    pub resident_tiles: u32,
    /// Tiles being generated, fetched or decoded.
    pub loading_tiles: u32,
    /// Wanted tiles beyond the tile budget or the device's capacity.
    pub dropped_tiles: u32,
    /// Patches the camera draws, and of those how many stand in for finer
    /// levels not resident yet.
    pub patches: u32,
    pub fallback_patches: u32,
}

/// Frames a failed fetch waits before it is requested again.
const RETRY_AFTER: u64 = 120;
/// Ranges, as multiples of each level's, within which tiles load and stay:
/// beyond its range a level's tiles still shade the coarser level's
/// patches (their normals), so they load further than they draw.
const LOAD_REACH: f32 = 2.0;
const KEEP_REACH: f32 = 2.5;

/// One terrain's streaming state.
struct TerrainState {
    entity: Entity,
    terrain: Terrain,
    tiles: HashMap<TileKey, Arc<Tile>>,
    /// Generators of tiles in progress.
    jobs: HashMap<TileKey, Generator>,
    /// File tiles requested and not yet delivered.
    in_flight: HashSet<TileKey>,
    arrivals: Vec<(TileKey, Option<Vec<u8>>)>,
    failed: HashMap<TileKey, u64>,
    selection: Selection,
    /// Level-0 tiles collision wants.
    collision: Vec<TileKey>,
    /// Advances whenever the resident tiles change.
    revision: u64,
    /// The largest height difference measured between each level and the
    /// next coarser one ([`Tile::morph_error`]), what its patches morph by;
    /// zero until measured.
    errors: Vec<f32>,
    /// The ranges the last selection used.
    ranges: Vec<f32>,
}

impl TerrainState {
    fn new(entity: Entity, terrain: Terrain) -> Self {
        let levels = terrain.pyramid.levels() as usize;
        Self {
            entity,
            terrain,
            tiles: HashMap::new(),
            jobs: HashMap::new(),
            in_flight: HashSet::new(),
            arrivals: Vec::new(),
            failed: HashMap::new(),
            selection: Selection::default(),
            collision: Vec::new(),
            revision: 1,
            errors: vec![0.0; levels],
            ranges: Vec::new(),
        }
    }

    /// Makes `tile` resident, measuring how far it lies from its resident
    /// parent and its resident children from it.
    fn insert(&mut self, tile: Tile) {
        let key = tile.key();
        let pyramid = self.terrain.pyramid;
        let level = usize::from(key.level);
        if let Some(parent) = pyramid
            .parent(key)
            .and_then(|parent| self.tiles.get(&parent))
        {
            self.errors[level] = crate::math::max(self.errors[level], tile.morph_error(parent));
        }
        if let Some(finer) = level.checked_sub(1) {
            for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let child = TileKey::new(finer as u8, key.x * 2 + dx, key.z * 2 + dz);
                if let Some(child) = self.tiles.get(&child) {
                    self.errors[finer] =
                        crate::math::max(self.errors[finer], child.morph_error(&tile));
                }
            }
        }
        self.tiles.insert(key, Arc::new(tile));
        self.revision += 1;
    }

    /// The finest resident tile holding terrain-local `local`, and its
    /// sample there.
    fn sample(&self, local: Vec2) -> Option<(u32, TileSample)> {
        let pyramid = &self.terrain.pyramid;
        (0..pyramid.levels()).find_map(|level| {
            let key = pyramid.tile_at(level, local)?;
            let tile = self.tiles.get(&key)?;
            let sample = tile.sample(local - pyramid.tile_origin(key))?;
            Some((level, sample))
        })
    }
}

/// Every terrain's resident tiles, streaming state and queries (decision
/// 051). Inserted by default; terrains stream during extraction.
#[derive(Resource, Default)]
pub struct Terrains {
    entries: Vec<TerrainState>,
    device_tiles: Option<u32>,
    frame: u64,
    stats: TerrainStats,
    requests: Vec<TileRequest>,
    // Scratch.
    wanted: Vec<WantedTile>,
}

impl Terrains {
    /// The surface under world point `(x, z)` of the first terrain with a
    /// resident tile there, from the finest such tile.
    pub fn sample(&self, x: f32, z: f32) -> Option<TerrainSample> {
        self.entries.iter().find_map(|state| {
            let origin = state.terrain.origin;
            let (level, sample) = state.sample(Vec2::new(x - origin.x, z - origin.z))?;
            Some(TerrainSample {
                terrain: state.entity,
                height: origin.y + sample.height,
                normal: sample.normal,
                hole: sample.hole,
                level,
            })
        })
    }

    /// The world height of the terrain under `(x, z)`; `None` where no tile
    /// is resident or the cell is a hole.
    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        self.sample(x, z)
            .filter(|sample| !sample.hole)
            .map(|sample| sample.height)
    }

    /// The surface normal under `(x, z)`, as [`Self::height_at`].
    pub fn normal_at(&self, x: f32, z: f32) -> Option<Vec3> {
        self.sample(x, z)
            .filter(|sample| !sample.hole)
            .map(|sample| sample.normal)
    }

    /// Whether the cell under `(x, z)` is a hole; `None` where no tile is
    /// resident.
    pub fn is_hole(&self, x: f32, z: f32) -> Option<bool> {
        self.sample(x, z).map(|sample| sample.hole)
    }

    /// Heights of a grid of `count` points from `min` to `max` (world x and
    /// z), row by row along z, into `out`: `None` where no tile is resident
    /// or the cell is a hole.
    pub fn region(&self, min: Vec2, max: Vec2, count: [u32; 2], out: &mut Vec<Option<f32>>) {
        out.clear();
        let steps = Vec2::new((count[0].max(2) - 1) as f32, (count[1].max(2) - 1) as f32);
        for row in 0..count[1] {
            for column in 0..count[0] {
                let point = min + (max - min) * Vec2::new(column as f32, row as f32) / steps;
                out.push(self.height_at(point.x, point.y));
            }
        }
    }

    /// Counters of the last streaming update.
    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    /// The terrain of `entity`, while it streams.
    pub fn terrain(&self, entity: Entity) -> Option<&Terrain> {
        self.entries
            .iter()
            .find(|state| state.entity == entity)
            .map(|state| &state.terrain)
    }

    /// Each level's largest measured height difference from the next
    /// coarser one (meters; zero until measured) and the range it drew to
    /// at the last selection (meters from the eye), finest first, of
    /// `entity`'s terrain.
    pub fn level_detail(&self, entity: Entity) -> Option<(&[f32], &[f32])> {
        self.entries
            .iter()
            .find(|state| state.entity == entity)
            .map(|state| (state.errors.as_slice(), state.ranges.as_slice()))
    }

    /// Resident tiles of `entity`'s terrain.
    pub fn tiles(&self, entity: Entity) -> impl Iterator<Item = &Arc<Tile>> {
        self.entries
            .iter()
            .filter(move |state| state.entity == entity)
            .flat_map(|state| state.tiles.values())
    }

    /// Level-0 tiles resident around the camera and focus entities, of
    /// terrains that collide: the terrain, its origin and the tile.
    pub fn collision_tiles(&self) -> impl Iterator<Item = (Entity, Vec3, &Arc<Tile>)> {
        self.entries
            .iter()
            .filter(|state| state.terrain.collision)
            .flat_map(|state| {
                state.collision.iter().filter_map(move |key| {
                    let tile = state.tiles.get(key)?;
                    Some((state.entity, state.terrain.origin, tile))
                })
            })
    }

    /// Advances whenever any terrain's resident tiles change.
    pub fn revision(&self) -> u64 {
        self.entries
            .iter()
            .map(|state| state.revision)
            .fold(self.entries.len() as u64, u64::wrapping_add)
    }

    /// Tiles the device holds, which renderers report: residency never
    /// exceeds it. `None` for no limit.
    pub fn set_device_tiles(&mut self, tiles: Option<u32>) {
        self.device_tiles = tiles;
    }

    /// Moves the byte ranges hosts should fetch into `out`.
    pub fn take_requests(&mut self, out: &mut Vec<TileRequest>) {
        out.append(&mut self.requests);
    }

    /// Hands back a fetched tile's bytes, or `None` when the fetch failed:
    /// it is requested again later. Tiles no longer wanted are ignored.
    pub fn deliver(&mut self, terrain: Entity, key: TileKey, bytes: Option<Vec<u8>>) {
        if let Some(state) = self
            .entries
            .iter_mut()
            .find(|state| state.entity == terrain)
            && state.in_flight.remove(&key)
        {
            state.arrivals.push((key, bytes));
        }
    }

    /// Streams every terrain toward `eye` (the camera's, unless frozen)
    /// and focus points, within the settings' work and tile budgets, and
    /// selects the patches seen from `eye`.
    fn stream(
        &mut self,
        settings: &TerrainSettings,
        camera: &Camera,
        eye: Vec3,
        viewport: Viewport,
        focus: &[Vec3],
    ) {
        self.frame += 1;
        let mut stats = TerrainStats {
            terrains: self.entries.len() as u32,
            ..TerrainStats::default()
        };
        let forward = camera.target - eye;
        let forward = Vec2::new(forward.x, forward.z);
        let fov = camera.vertical_fov_radians;
        let fov = if fov.is_finite() && fov > 0.0 && fov < std::f32::consts::PI {
            fov
        } else {
            Camera::default().vertical_fov_radians
        };
        let mut budget = u64::from(settings.samples_per_frame);
        let limit = match (settings.max_tiles, self.device_tiles) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        // Budgets cover every terrain: each takes what is left.
        let mut remaining = limit.map_or(usize::MAX, |limit| limit as usize);
        let frame = self.frame;
        for state in &mut self.entries {
            if !state.terrain.is_usable() {
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
            // What to load: pinned levels and the camera's tiles, then
            // collision's, most important first.
            self.wanted.clear();
            engine_terrain::wanted_tiles(
                &pyramid,
                &ranges,
                local_eye,
                forward,
                LOAD_REACH,
                settings.pinned_levels,
                &mut self.wanted,
            );
            state.collision.clear();
            if state.terrain.collision && settings.collision_radius > 0.0 {
                for point in focus {
                    collision_keys(
                        &pyramid,
                        *point - origin,
                        settings.collision_radius,
                        &mut state.collision,
                    );
                }
                state.collision.sort_unstable();
                state.collision.dedup();
            }
            let pinned_end = self
                .wanted
                .iter()
                .position(|tile| {
                    u32::from(tile.key.level) + settings.pinned_levels < pyramid.levels()
                })
                .unwrap_or(self.wanted.len());
            let mut order: Vec<TileKey> =
                Vec::with_capacity(self.wanted.len() + state.collision.len());
            let mut seen: HashSet<TileKey> = HashSet::with_capacity(order.capacity());
            for key in self.wanted[..pinned_end]
                .iter()
                .map(|tile| tile.key)
                .chain(state.collision.iter().copied())
                .chain(self.wanted[pinned_end..].iter().map(|tile| tile.key))
            {
                if seen.insert(key) {
                    order.push(key);
                }
            }
            let kept = order.len().min(remaining);
            stats.dropped_tiles += (order.len() - kept) as u32;
            remaining -= kept;
            let target: HashSet<TileKey> = order[..kept].iter().copied().collect();
            // What to keep: everything within a wider reach, unless over
            // budget.
            self.wanted.clear();
            engine_terrain::wanted_tiles(
                &pyramid,
                &ranges,
                local_eye,
                forward,
                KEEP_REACH,
                settings.pinned_levels,
                &mut self.wanted,
            );
            let keep: HashSet<TileKey> = self
                .wanted
                .iter()
                .map(|tile| tile.key)
                .chain(state.collision.iter().copied())
                .collect();
            let before = state.tiles.len();
            state.tiles.retain(|key, _| keep.contains(key));
            if state.tiles.len() > kept {
                // Over budget: drop resident tiles outside the target, finest
                // first.
                let mut extra: Vec<TileKey> = state
                    .tiles
                    .keys()
                    .filter(|key| !target.contains(key))
                    .copied()
                    .collect();
                extra.sort_unstable_by_key(|key| key.level);
                for key in extra.into_iter().take(state.tiles.len() - kept) {
                    state.tiles.remove(&key);
                }
            }
            if state.tiles.len() != before {
                state.revision += 1;
            }
            state.jobs.retain(|key, _| target.contains(key));
            // Load what is missing, most important first; disabled terrains
            // load nothing.
            let missing: Vec<TileKey> = order[..kept]
                .iter()
                .copied()
                .filter(|key| settings.enabled && !state.tiles.contains_key(key))
                .collect();
            match state.terrain.source.clone() {
                TerrainSource::Generated(layers) => {
                    // The coarsest level always draws: generate it whole at once.
                    let first = state.tiles.is_empty();
                    for key in missing {
                        let mut unlimited = u64::MAX;
                        let work = if first && u32::from(key.level) + 1 == pyramid.levels() {
                            &mut unlimited
                        } else {
                            &mut budget
                        };
                        if *work == 0 {
                            break;
                        }
                        let job = state
                            .jobs
                            .entry(key)
                            .or_insert_with(|| Generator::new(layers.clone(), &pyramid, key));
                        if let Some(tile) = job.advance(work) {
                            state.jobs.remove(&key);
                            state.insert(tile);
                        }
                    }
                }
                TerrainSource::File {
                    header,
                    url: _,
                    data: Some(data),
                } => {
                    // The whole file is here: decode within the budget.
                    for key in missing {
                        if budget == 0 {
                            break;
                        }
                        let Some(entry) = header.tile(key) else {
                            continue;
                        };
                        let range = usize::try_from(entry.offset).ok().and_then(|start| {
                            data.get(start..start.checked_add(entry.length as usize)?)
                        });
                        match range.map(|bytes| decode_tile(&header, key, bytes)) {
                            Some(Ok(tile)) => state.insert(tile),
                            _ => {
                                state.failed.insert(key, frame);
                            }
                        }
                        budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
                    }
                }
                TerrainSource::File {
                    header,
                    url,
                    data: None,
                } => {
                    for &key in &missing {
                        if state.in_flight.contains(&key)
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
                        self.requests.push(TileRequest {
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
                        if budget == 0 || !target.contains(&key) {
                            if target.contains(&key) {
                                later.push((key, bytes));
                            }
                            continue;
                        }
                        match bytes.map(|bytes| decode_tile(&header, key, &bytes)) {
                            Some(Ok(tile)) => {
                                state.failed.remove(&key);
                                state.insert(tile);
                            }
                            _ => {
                                state.failed.insert(key, frame);
                            }
                        }
                        budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
                    }
                    state.arrivals = later;
                }
            }
            stats.resident_tiles += state.tiles.len() as u32;
            stats.loading_tiles +=
                (state.jobs.len() + state.in_flight.len() + state.arrivals.len()) as u32;
            if settings.enabled {
                let tiles = &state.tiles;
                engine_terrain::select_patches(
                    &pyramid,
                    &ranges,
                    local_eye,
                    |key| tiles.get(&key).map(Arc::as_ref),
                    &mut state.selection,
                );
            } else {
                state.selection.patches.clear();
                state.selection.fallbacks = 0;
            }
            stats.patches += state.selection.patches.len() as u32;
            stats.fallback_patches += state.selection.fallbacks;
        }
        self.stats = stats;
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

/// One terrain of a frame (decision 051).
#[derive(Clone, Debug)]
pub struct RenderTerrain {
    /// The terrain's entity bits, which renderers key its tiles by.
    pub id: u64,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub material: TerrainMaterial,
    /// Its resident tiles: renderers upload those they lack and release
    /// the rest. The list changes only when `tiles_revision` advances.
    pub tiles: Arc<[Arc<Tile>]>,
    pub tiles_revision: u64,
    /// The patches the camera draws.
    pub patches: Vec<Patch>,
    /// Meters from the eye within which each level's tiles load, finest
    /// first: where renderers can shade with them.
    pub reach: Vec<f32>,
}

/// The frame's terrains (decision 051).
#[derive(Clone, Debug, Default)]
pub struct RenderTerrains {
    pub terrains: Vec<RenderTerrain>,
    /// The camera's view and projection without its translation: terrain
    /// draws camera-relative, from positions less the camera's.
    pub relative_view_projection: Mat4,
    pub stats: TerrainStats,
    /// What terrain shows ([`TerrainDebug`]).
    pub view: TerrainView,
}

/// Streams terrains and lists them in each frame.
pub(crate) struct TerrainExtraction {
    terrains: QueryState<(Entity, &'static Terrain)>,
    focus: QueryState<Entity, With<TerrainFocus>>,
    points: Vec<Vec3>,
}

impl TerrainExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            terrains: world.query::<(Entity, &Terrain)>(),
            focus: world.query_filtered::<Entity, With<TerrainFocus>>(),
            points: Vec::new(),
        }
    }

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
        self.points.clear();
        self.points.push(eye);
        if let Some(globals) = world.get_resource::<GlobalTransforms>() {
            for entity in self.focus.iter(world) {
                if let Some(point) = globals.get(entity).and_then(|global| global.translation()) {
                    self.points.push(point);
                }
            }
        }
        let present: Vec<(Entity, Terrain)> = self
            .terrains
            .iter(world)
            .map(|(entity, terrain)| (entity, terrain.clone()))
            .collect();
        let Some(mut terrains) = world.get_resource_mut::<Terrains>() else {
            return;
        };
        let terrains = terrains.as_mut();
        terrains
            .entries
            .retain(|state| present.iter().any(|(entity, _)| *entity == state.entity));
        for (entity, terrain) in present {
            match terrains
                .entries
                .iter_mut()
                .find(|state| state.entity == entity)
            {
                Some(state) => {
                    if !terrain.same(&state.terrain) {
                        *state = TerrainState::new(entity, terrain);
                    }
                }
                None => terrains.entries.push(TerrainState::new(entity, terrain)),
            }
        }
        let Some(settings) = settings else {
            for state in &mut terrains.entries {
                state.selection.patches.clear();
            }
            terrains.stats = TerrainStats {
                terrains: terrains.entries.len() as u32,
                ..TerrainStats::default()
            };
            return;
        };
        terrains.stream(&settings, &camera, eye, viewport, &self.points);
    }

    /// Lists every terrain's resident tiles and the camera's patches.
    pub(crate) fn extract(&mut self, world: &World, aspect: f32, snapshot: &mut RenderSnapshot) {
        let listing = &mut snapshot.terrain;
        let Some(terrains) = world.get_resource::<Terrains>() else {
            listing.terrains.clear();
            listing.stats = TerrainStats::default();
            return;
        };
        let camera = world.resource::<Camera>();
        listing.relative_view_projection = camera.relative_view_projection(aspect);
        listing.stats = terrains.stats;
        listing.view = world
            .get_resource::<TerrainDebug>()
            .map_or(TerrainView::Shaded, |debug| debug.view);
        let enabled = world
            .get_resource::<TerrainSettings>()
            .is_some_and(|settings| settings.enabled);
        listing.terrains.truncate(terrains.entries.len());
        for (index, state) in terrains.entries.iter().enumerate() {
            if index == listing.terrains.len() {
                listing.terrains.push(RenderTerrain {
                    id: 0,
                    origin: Vec3::ZERO,
                    pyramid: state.terrain.pyramid,
                    material: TerrainMaterial::default(),
                    tiles: Arc::from(Vec::new()),
                    tiles_revision: 0,
                    patches: Vec::new(),
                    reach: Vec::new(),
                });
            }
            let render = &mut listing.terrains[index];
            render.id = state.entity.to_bits();
            render.origin = state.terrain.origin;
            render.pyramid = state.terrain.pyramid;
            render.material = state.terrain.material;
            if render.tiles_revision != state.revision || render.id != state.entity.to_bits() {
                let mut tiles: Vec<Arc<Tile>> = state.tiles.values().cloned().collect();
                tiles.sort_unstable_by_key(|tile| tile.key());
                render.tiles = Arc::from(tiles);
                render.tiles_revision = state.revision;
            }
            render.patches.clear();
            if enabled {
                render.patches.extend_from_slice(&state.selection.patches);
            }
            render.reach.clear();
            render
                .reach
                .extend(state.ranges.iter().map(|range| range * LOAD_REACH));
        }
    }
}
