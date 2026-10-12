//! Each terrain's resident tiles and what streaming keeps for it, and the
//! Terrains resource holding them.

use super::*;

/// Frames a failed fetch waits before it is requested again.
pub(super) const RETRY_AFTER: u64 = 120;

/// Tiles failing on workers after which a terrain generates
/// its own, as when the workers' module does not match the engine's.
pub(super) const WORKER_FAILURES: u32 = 8;

/// Ranges, as multiples of each level's, within which tiles load and stay:
/// beyond its range a level's tiles still shade the coarser level's
/// patches (their normals), so they load further than they draw.
pub(super) const LOAD_REACH: f32 = 2.0;

pub(super) const KEEP_REACH: f32 = 2.5;

/// Tiles also load around where the eye is headed: its
/// velocity carried this many seconds on, at most this many meters; the
/// eye moving farther in a frame jumped there.
pub(super) const AHEAD_SECONDS: f32 = 2.0;

pub(super) const AHEAD_MAX: f32 = 1000.0;

pub(super) const JUMP: f32 = 100.0;

/// One terrain's streaming state.
pub(super) struct TerrainState {
    pub(super) entity: Entity,
    pub(super) terrain: Terrain,
    pub(super) tiles: HashMap<TileKey, Arc<Tile>>,
    /// Generators of tiles in progress.
    pub(super) jobs: HashMap<TileKey, Generator>,
    /// File tiles requested and not yet delivered.
    pub(super) in_flight: HashSet<TileKey>,
    pub(super) arrivals: Vec<(TileKey, Option<Vec<u8>>)>,
    pub(super) failed: HashMap<TileKey, u64>,
    pub(super) selection: Selection,
    /// Level-0 tiles collision wants.
    pub(super) collision: Vec<TileKey>,
    /// Advances whenever the resident tiles, or the placement assets and
    /// bridge material standing on them, change.
    pub(super) revision: u64,
    /// The largest height difference measured between each level and the
    /// next coarser one ([`Tile::morph_error`]), what its patches morph by;
    /// zero until measured.
    pub(super) errors: Vec<f32>,
    /// The ranges the last selection used.
    pub(super) ranges: Vec<f32>,
    /// The layers tiles generate from once its bakes are done,
    /// the bake in progress, and what the last hydrology and
    /// erosion bakes found.
    pub(super) baked: Option<Arc<Layers>>,
    pub(super) bake: Option<TerrainBake>,
    pub(super) hydrology: Option<HydrologyStats>,
    pub(super) erosion: Option<ErosionStats>,
    /// Its roads' bridges, terrain-local, once every bake
    /// is done, and its rivers' waterfalls.
    pub(super) bridges: Arc<[Bridge]>,
    pub(super) falls: Arc<[Fall]>,
    /// The layers its tiles generate from with its edits over them,
    /// and the baked layers they were made over; and the
    /// resident tiles under edits since, which draw until regenerated.
    pub(super) edited: Option<(Arc<Layers>, Arc<Layers>)>,
    pub(super) stale: HashSet<TileKey>,
    /// Resident tiles holding their shape alone so far:
    /// heights and holes, drawn with coarser levels' materials and collided
    /// with while the rest loads.
    pub(super) shaped: HashSet<TileKey>,
    /// For a terrain's surroundings, what they were drawn
    /// from.
    pub(super) around: Option<Around>,
    /// The tiles the last frame found missing, most important first with
    /// their jobs' priorities, every tile it loads and every
    /// tile it keeps: what loads between frames.
    pub(super) missing: Vec<(TileKey, u32)>,
    pub(super) target: HashSet<TileKey>,
    pub(super) keep: HashSet<TileKey>,
    /// Workers' share: the recipe they generate its tiles
    /// from, the tiles generating there (each one's job, the recipe's
    /// revision it generates from and its priority) and its bakes running
    /// there, unless they failed there; and the tiles that failed there.
    pub(super) recipe: Option<SharedRecipe>,
    pub(super) dispatched: HashMap<TileKey, (JobId, u64, u32)>,
    /// Resident tiles without their objects standing far away,
    /// with their tiles' priorities, and the jobs placing those, each
    /// with the tile it places on.
    pub(super) far_wanted: HashMap<TileKey, u32>,
    pub(super) far_jobs: HashMap<TileKey, (JobId, Arc<Tile>)>,
    pub(super) bake_job: Option<JobId>,
    pub(super) bake_inline: bool,
    pub(super) worker_failures: u32,
    /// Tiles that left whole, kept in memory to return at
    /// once while the layers they generated from stay, and their bytes.
    pub(super) cache: HashMap<TileKey, CachedTile>,
    pub(super) cached_bytes: u64,
    /// Tiles generating ahead of the view on workers while nothing is
    /// missing, for memory: each one's job and the recipe's
    /// revision it generates from; and the tiles to generate so, nearest
    /// first.
    pub(super) idle: HashMap<TileKey, (JobId, u64)>,
    pub(super) ahead: Vec<TileKey>,
    /// The most memory one tile kept took: the room each tile generated
    /// ahead of need is given.
    pub(super) tile_bytes: u64,
    /// Tiles generating in bands of rows on several workers;
    /// tiles their bands made whose objects a job places, each with
    /// that job, the recipe's revision, its priority and its placing; and
    /// how many tiles generated so.
    pub(super) banded: HashMap<TileKey, Banding>,
    pub(super) placements: HashMap<TileKey, Placements>,
    pub(super) banded_tiles: u64,
    /// The layers its tiles generate from and its surface's data for the
    /// GPU from them, `None` where the program needs more
    /// than the GPU's interpreter holds; and whether its tiles hold control
    /// values, as without GPU materials.
    pub(super) surface: Option<(Arc<Layers>, Option<Arc<engine_terrain::gpu::SurfaceData>>)>,
    pub(super) materials: bool,
}

/// The objects of a tile its bands made, placed in bands of
/// candidate rows: the recipe's revision, its priority and
/// placing, and each band's job with its objects once done.
pub(super) struct Placements {
    pub(super) made: u64,
    pub(super) priority: u32,
    pub(super) placing: Placing,
    pub(super) jobs: Vec<(JobId, Option<Vec<engine_terrain::Placed>>)>,
}

/// A tile generating in bands of rows: the recipe's revision
/// and the priority it generates at, the objects placed after, and its
/// bands' jobs, each with its band once done.
pub(super) struct Banding {
    pub(super) made: u64,
    pub(super) priority: u32,
    pub(super) placing: Placing,
    pub(super) jobs: Vec<(JobId, Option<TileBand>)>,
}

/// A tile kept in memory after it left the view: its bytes
/// there and the frame it left, or the last a place a game names wanted it.
pub(super) struct CachedTile {
    tile: Arc<Tile>,
    bytes: u64,
    pub(super) left: u64,
}

/// Bytes of one of a tile's arrays on the GPU, heights or control values:
/// what a tile arriving uploads, once or twice.
pub(super) const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;

/// Bytes `tile` holds in memory.
fn memory_bytes(tile: &Tile) -> u64 {
    let arrays = tile.heights().len() + tile.control().map_or(0, <[u32]>::len);
    let holes = tile.hole_bits().map_or(0, <[u64]>::len) * 8;
    (arrays * 4 + holes + std::mem::size_of_val(tile.placed())) as u64
}

/// What a terrain's surroundings were drawn from: the layers
/// its tiles generate from and its settings for them; and the rectangle of
/// their own ground (local) the terrain covers.
pub(super) struct Around {
    pub(super) base: Arc<Layers>,
    pub(super) surroundings: Surroundings,
    pub(super) cutout: (Vec2, Vec2),
}

impl TerrainState {
    pub(super) fn new(entity: Entity, terrain: Terrain) -> Self {
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
            baked: None,
            bake: None,
            hydrology: None,
            erosion: None,
            bridges: Arc::from(Vec::new()),
            falls: Arc::from(Vec::new()),
            edited: None,
            stale: HashSet::new(),
            shaped: HashSet::new(),
            around: None,
            missing: Vec::new(),
            target: HashSet::new(),
            keep: HashSet::new(),
            recipe: None,
            dispatched: HashMap::new(),
            far_wanted: HashMap::new(),
            far_jobs: HashMap::new(),
            bake_job: None,
            bake_inline: false,
            worker_failures: 0,
            cache: HashMap::new(),
            cached_bytes: 0,
            tile_bytes: 0,
            idle: HashMap::new(),
            ahead: Vec::new(),
            banded: HashMap::new(),
            placements: HashMap::new(),
            banded_tiles: 0,
            surface: None,
            materials: true,
        }
    }

    /// Whether its tiles hold control values, as `gpu` (the settings' GPU
    /// materials) and its surface allow: generated terrains
    /// whose surface the GPU's interpreter holds leave them to the GPU.
    /// Tiles resident when that changes regenerate, drawn until they do.
    pub(super) fn update_materials(&mut self, gpu: bool) {
        let TerrainSource::Generated(authored) = &self.terrain.source else {
            self.materials = true;
            return;
        };
        let layers = self.baked.clone().unwrap_or_else(|| authored.clone());
        let materials = if gpu && layers.unbaked().is_none() {
            let layers = self.generation(layers);
            let current = self
                .surface
                .as_ref()
                .is_some_and(|(held, _)| Arc::ptr_eq(held, &layers));
            if !current {
                let data =
                    engine_terrain::gpu::SurfaceData::new(&self.terrain.surface.rules, &layers)
                        .map(Arc::new);
                self.surface = Some((layers, data));
            }
            self.surface.as_ref().is_none_or(|(_, data)| data.is_none())
        } else {
            true
        };
        if materials != self.materials {
            self.materials = materials;
            let resident: Vec<TileKey> = self.tiles.keys().copied().collect();
            self.stale.extend(resident);
            let kept: Vec<TileKey> = self.cache.keys().copied().collect();
            for key in kept {
                self.uncache(key);
            }
            self.revision += 1;
        }
    }

    /// Its workers' jobs and recipe, for the next frame with workers to
    /// drop once the state is gone.
    pub(super) fn retire(&self, retired: &mut Retired) {
        retired
            .jobs
            .extend(self.dispatched.values().map(|&(id, _, _)| id));
        retired
            .jobs
            .extend(self.far_jobs.values().map(|&(id, _)| id));
        retired.jobs.extend(self.bake_job);
        retired.jobs.extend(self.idle.values().map(|&(id, _)| id));
        retired.jobs.extend(
            self.banded
                .values()
                .flat_map(|banding| banding.jobs.iter().map(|&(id, _)| id)),
        );
        retired.jobs.extend(
            self.placements
                .values()
                .flat_map(|placements| placements.jobs.iter().map(|&(id, _)| id)),
        );
        retired
            .recipes
            .extend(self.recipe.as_ref().map(|recipe| recipe.key.clone()));
    }

    /// The id renderers key its tiles by: its entity's bits, the top bit
    /// flipped for surroundings.
    pub(super) fn render_id(&self) -> u64 {
        self.entity.to_bits() ^ (u64::from(self.around.is_some()) << 63)
    }

    /// Whether terrain-local `local` lies in the ground surroundings
    /// leave to the terrain they surround.
    fn cut_out(&self, local: Vec2) -> bool {
        self.around.as_ref().is_some_and(|around| {
            let (low, high) = around.cutout;
            local.x > low.x && local.y > low.y && local.x < high.x && local.y < high.y
        })
    }

    /// Takes `edits` in place of its own: the resident tiles
    /// under any edit added or taken away, and two of their samples beyond,
    /// regenerate over the new stack, drawn as they were until then.
    pub(super) fn edit(&mut self, edits: Arc<[TerrainEdit]>) {
        let mut reach: Option<(Vec2, Vec2)> = None;
        let changed = edits
            .iter()
            .filter(|edit| !self.terrain.edits.contains(edit))
            .chain(
                self.terrain
                    .edits
                    .iter()
                    .filter(|edit| !edits.contains(edit)),
            );
        for edit in changed {
            let center = Vec2::from(edit.stamp.center);
            let radius = Vec2::splat(edit.stamp.radius + edit.stamp.falloff);
            let (low, high) = (center - radius, center + radius);
            reach = Some(reach.map_or((low, high), |(min, max)| (min.min(low), max.max(high))));
        }
        self.terrain.edits = edits;
        self.edited = None;
        let Some((low, high)) = reach else {
            return;
        };
        let pyramid = self.terrain.pyramid;
        let under = |key: &TileKey| {
            let spacing = pyramid.spacing(u32::from(key.level));
            let size = TILE_QUADS as f32 * spacing;
            let first =
                Vec2::new(f32::from(key.x), f32::from(key.z)) * size - Vec2::splat(2.0 * spacing);
            let last = first + Vec2::splat(size + 4.0 * spacing);
            first.x <= high.x && first.y <= high.y && last.x >= low.x && last.y >= low.y
        };
        let stale: Vec<TileKey> = self
            .tiles
            .keys()
            .filter(|key| under(key))
            .copied()
            .collect();
        self.stale.extend(stale);
        self.jobs.retain(|key, _| !under(key));
        // Tiles kept in memory under the edits are no longer the ground.
        let gone: Vec<TileKey> = self
            .cache
            .keys()
            .filter(|key| under(key))
            .copied()
            .collect();
        for key in gone {
            self.uncache(key);
        }
        self.revision += 1;
    }

    /// The layers its tiles generate from: `base`, its layers
    /// with their bakes, with its edits over them.
    pub(super) fn generation(&mut self, base: Arc<Layers>) -> Arc<Layers> {
        if self.terrain.edits.is_empty() {
            return base;
        }
        if let Some((over, edited)) = &self.edited
            && Arc::ptr_eq(over, &base)
        {
            return edited.clone();
        }
        let stamps: Vec<Stamp> = self.terrain.edits.iter().map(|edit| edit.stamp).collect();
        match base.with_stamps(&stamps) {
            Ok(edited) => {
                let edited = Arc::new(edited);
                self.edited = Some((base, edited.clone()));
                edited
            }
            // Edits are validated as they are made.
            Err(_) => base,
        }
    }

    /// Gives decoded `tile` its surface's control values, within
    /// `budget`; it becomes resident once they are done.
    pub(super) fn resurface(&mut self, tile: Tile, budget: &mut u64) {
        let key = tile.key();
        let mut job = Generator::resurface(
            &tile,
            &self.terrain.pyramid,
            self.terrain.surface.rules.clone(),
        );
        match job.advance(budget) {
            Some(tile) => self.insert(tile),
            None => {
                self.jobs.insert(key, job);
                // Its shape draws meanwhile.
                self.shape(tile);
            }
        }
    }

    /// Advances `key`'s surface job within `budget`, if it has one,
    /// making its tile resident once done; whether it had one.
    pub(super) fn surface_job(&mut self, key: TileKey, budget: &mut u64) -> bool {
        let Some(job) = self.jobs.get_mut(&key) else {
            return false;
        };
        if let Some(tile) = job.advance(budget) {
            self.jobs.remove(&key);
            self.shaped.remove(&key);
            self.insert(tile);
        }
        true
    }

    /// Whether `key` still loads: not resident, resident under edits since,
    /// or resident with its shape alone.
    pub(super) fn loads(&self, key: &TileKey) -> bool {
        !self.tiles.contains_key(key) || self.stale.contains(key) || self.shaped.contains(key)
    }

    /// Makes `shape`, a tile's heights and holes alone, resident until the
    /// whole tile arrives, unless its key is resident or no
    /// coarser resident tile holds materials to draw it with; whether it
    /// did.
    pub(super) fn shape(&mut self, shape: Tile) -> bool {
        let key = shape.key();
        if self.tiles.contains_key(&key) {
            return false;
        }
        let pyramid = self.terrain.pyramid;
        let mut ancestor = pyramid.parent(key);
        // The GPU gives shapes their materials.
        if !self.materials {
            self.shaped.insert(key);
            self.insert(shape);
            return true;
        }
        loop {
            let Some(at) = ancestor else {
                return false;
            };
            if self
                .tiles
                .get(&at)
                .is_some_and(|tile| tile.control().is_some())
            {
                break;
            }
            ancestor = pyramid.parent(at);
        }
        self.shaped.insert(key);
        self.insert(shape);
        true
    }

    /// Makes `tile` resident, measuring how far it lies from its resident
    /// parent and its resident children from it.
    pub(super) fn insert(&mut self, tile: Tile) {
        self.insert_shared(Arc::new(tile));
    }

    /// [`Self::insert`] for a tile already shared, as one kept in memory.
    fn insert_shared(&mut self, tile: Arc<Tile>) {
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
        self.tiles.insert(key, tile);
        self.revision += 1;
    }

    /// Makes generated `tile` resident in place of whatever its key held:
    /// its shape, a tile under edits since, or one waiting for its objects
    /// standing far away, whose job in `jobs` stops. One without the
    /// objects standing far away its level places wants them placed, at job
    /// priority `priority`, whichever way it arrived.
    pub(super) fn arrive(&mut self, tile: Tile, priority: u32, jobs: Option<&mut Jobs>) {
        let key = tile.key();
        self.stale.remove(&key);
        self.shaped.remove(&key);
        self.far_wanted.remove(&key);
        if let Some((id, _)) = self.far_jobs.remove(&key)
            && let Some(jobs) = jobs
        {
            jobs.cancel(id);
        }
        let far = self.lacks_far(&tile);
        self.insert(tile);
        if far {
            self.far_wanted.insert(key, priority);
        }
    }

    /// Whether `tile` lacks objects standing far away its level places.
    fn lacks_far(&self, tile: &Tile) -> bool {
        tile.placing() != Some(Placing::All)
            && self
                .terrain
                .surface
                .rules
                .places(tile.key().level, Placing::Far)
    }

    /// Resident tiles lacking objects their levels place with nothing under
    /// way to place them, which no frame would finish:
    /// [`TerrainStats::unplaced_tiles`].
    pub(super) fn unplaced_tiles(&self) -> u32 {
        let rules = &self.terrain.surface.rules;
        let levels: Vec<(bool, bool)> = (0..self.terrain.pyramid.levels())
            .map(|level| {
                let level = level as u8;
                (
                    rules.places(level, Placing::Near),
                    rules.places(level, Placing::Far),
                )
            })
            .collect();
        let unplaced = self.tiles.iter().filter(|&(key, tile)| {
            let Some(&(near, far)) = levels.get(usize::from(key.level)) else {
                return false;
            };
            if self.shaped.contains(key)
                || self.stale.contains(key)
                || self.placements.contains_key(key)
            {
                return false;
            }
            let near = near && tile.placing().is_none();
            let far = far
                && tile.placing() != Some(Placing::All)
                && !self.far_wanted.contains_key(key)
                && !self.far_jobs.contains_key(key);
            near || far
        });
        unplaced.count() as u32
    }

    /// Keeps `tile`, leaving residency at frame `frame`, in memory
    /// where it is whole: not its shape alone, not under
    /// edits since and not without its objects standing far away.
    pub(super) fn cache_leaving(&mut self, key: TileKey, tile: Arc<Tile>, frame: u64) {
        if !(self.shaped.contains(&key)
            || self.stale.contains(&key)
            || self.far_wanted.contains_key(&key)
            || self.far_jobs.contains_key(&key)
            || self.lacks_far(&tile))
        {
            self.cache_tile(key, tile, frame);
        }
    }

    /// Keeps `tile` in memory as of frame `frame`.
    pub(super) fn cache_tile(&mut self, key: TileKey, tile: Arc<Tile>, frame: u64) {
        let bytes = memory_bytes(&tile);
        self.uncache(key);
        self.cached_bytes += bytes;
        self.tile_bytes = self.tile_bytes.max(bytes);
        self.cache.insert(
            key,
            CachedTile {
                tile,
                bytes,
                left: frame,
            },
        );
    }

    /// Drops `key`'s tile from memory, if kept; it.
    pub(super) fn uncache(&mut self, key: TileKey) -> Option<Arc<Tile>> {
        let cached = self.cache.remove(&key)?;
        self.cached_bytes -= cached.bytes;
        Some(cached.tile)
    }

    /// Makes the tiles of `missing` kept in memory resident again,
    /// most important first, within `work`'s upload
    /// budget; how many.
    pub(super) fn restore(&mut self, missing: &[(TileKey, u32)], work: &Work) -> u64 {
        let mut restored = 0;
        for &(key, _) in missing {
            if self.cache.is_empty() {
                break;
            }
            if self.tiles.contains_key(&key) || !self.cache.contains_key(&key) {
                continue;
            }
            if restored > 0 && !work.uploads() {
                break;
            }
            if let Some(tile) = self.uncache(key) {
                work.upload(2 * TILE_ARRAY_BYTES);
                self.insert_shared(tile);
                restored += 1;
            }
        }
        restored
    }

    /// The finest resident tile holding terrain-local `local`, and its
    /// sample there.
    pub(super) fn sample(&self, local: Vec2) -> Option<(u32, TileSample)> {
        if self.cut_out(local) {
            return None;
        }
        let pyramid = &self.terrain.pyramid;
        (0..pyramid.levels()).find_map(|level| {
            let key = pyramid.tile_at(level, local)?;
            let tile = self.tiles.get(&key)?;
            let sample = tile.sample(local - pyramid.tile_origin(key))?;
            Some((level, sample))
        })
    }

    /// The meters tracks sink the ground at terrain-local `local` at most:
    /// its nearest sample's materials' blend, from the
    /// finest resident tile.
    pub(super) fn deform(&self, local: Vec2) -> Option<f32> {
        if self.cut_out(local) {
            return None;
        }
        let pyramid = &self.terrain.pyramid;
        (0..pyramid.levels()).find_map(|level| {
            let key = pyramid.tile_at(level, local)?;
            let tile = self.tiles.get(&key)?;
            let at = (local - pyramid.tile_origin(key)) / tile.spacing();
            let (x, z) = (at.x.round() as i32, at.y.round() as i32);
            // Tiles whose materials the GPU gives: the rules'
            // at the sample.
            let control = match (tile.control_at(x, z), &self.surface) {
                (Some(control), _) => control,
                (None, Some((layers, _))) => {
                    let stored =
                        |sample: i32| (sample.clamp(-1, TILE_STRIDE as i32 - 2) + 1) as usize;
                    engine_terrain::gpu::sample_control(
                        tile,
                        (layers, &self.terrain.surface.rules),
                        (stored(x), stored(z)),
                    )
                }
                (None, None) => return None,
            };
            let materials = &self.terrain.surface.materials;
            let deform = |index: u8| {
                materials
                    .get(usize::from(index))
                    .map_or(0.0, |material| material.deform)
            };
            let share = f32::from(control.blend) / 255.0;
            Some(deform(control.a) + (deform(control.b) - deform(control.a)) * share)
        })
    }
}

/// Every terrain's resident tiles, streaming state and queries.
/// Inserted by default; terrains stream during extraction.
#[derive(Resource, Default)]
pub struct Terrains {
    pub(super) entries: Vec<TerrainState>,
    pub(super) device_tiles: Option<u32>,
    pub(super) frame: u64,
    pub(super) stats: TerrainStats,
    pub(super) requests: Vec<TileRequest>,
    /// Milliseconds now, from the host; without one, a
    /// frame works a slice.
    pub(super) clock: Option<fn() -> f64>,
    /// The eye last streamed toward and its velocity, the
    /// last frame's seconds, and the seconds since the eye last moved faster
    /// than the settings' motion detail speed, `None` when
    /// it has not since they asked for it.
    pub(super) last_eye: Option<Vec3>,
    pub(super) velocity: Vec3,
    pub(super) seconds: f32,
    pub(super) fast_ago: Option<f32>,
    /// Whether the camera's view has loaded whole since terrains arrived.
    pub(super) loaded: bool,
    /// Workers' share: how many ran at the last frame, the
    /// stacks shared with them and keys handed out so far, and the jobs and
    /// recipes of terrains gone, to drop.
    pub(super) workers: u32,
    pub(super) stacks: Vec<(Arc<Layers>, String)>,
    pub(super) shares: u64,
    pub(super) retired: Retired,
    /// Tiles returned from memory since terrains started;
    /// and bytes of the frame's upload budget left for work between
    /// frames, and spent there since the last frame.
    pub(super) restored: u64,
    pub(super) upload_left: u64,
    pub(super) between: u64,
}

/// The workers' jobs and recipes of terrains gone.
#[derive(Default)]
pub(super) struct Retired {
    pub(super) jobs: Vec<JobId>,
    pub(super) recipes: Vec<String>,
}
