//! Each terrain's resident tiles and what streaming keeps for it, and the
//! Terrains resource holding them.

use super::*;

/// Frames a failed record waits before it is asked for again.
pub(super) const RETRY_AFTER: u64 = 120;

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

/// What a terrain's last hydrology bake found, its rivers' flow once a
/// solid stood in them (their water as what stands there now holds it
/// back), and the square around the camera its water is simulated over,
/// where water asks for one. A new bake's rivers flow anew, and its
/// simulated water starts over from their water as it settles.
#[derive(Default)]
pub(super) struct HydrologyState {
    pub(super) stats: Option<HydrologyStats>,
    pub(super) flow: Option<engine_terrain::RiverFlow>,
    pub(super) window: Option<Arc<WaterWindow>>,
}

impl HydrologyState {
    /// A bake's, which found `stats`.
    pub(super) fn baked(stats: HydrologyStats) -> Self {
        Self {
            stats: Some(stats),
            flow: None,
            window: None,
        }
    }
}

/// One terrain's streaming state.
pub(super) struct TerrainState {
    pub(super) entity: Entity,
    pub(super) terrain: Terrain,
    /// Its resident tiles, and those kept in memory.
    pub(super) store: TileStore,
    pub(super) failed: TileMap<u64>,
    pub(super) selection: Selection,
    /// Level-0 tiles collision wants.
    pub(super) collision: Vec<TileKey>,
    /// Advances whenever what stands on its tiles changes: the placement
    /// assets, the bridge material, and the bridges and falls its bakes
    /// found ([`Self::revision`] adds the tiles' own).
    pub(super) standing: u64,
    /// The ranges the last selection used.
    pub(super) ranges: Vec<f32>,
    /// The layers tiles generate from once its bakes are done, and what
    /// the last hydrology and erosion bakes found.
    pub(super) baked: Option<Arc<Layers>>,
    pub(super) hydrology: HydrologyState,
    pub(super) erosion: Option<ErosionStats>,
    /// Its roads' bridges, terrain-local, once every bake
    /// is done, and its rivers' waterfalls.
    pub(super) bridges: Arc<[Bridge]>,
    pub(super) falls: Arc<[Fall]>,
    /// The layers its tiles generate from with its edits over them,
    /// and the baked layers they were made over.
    pub(super) edited: Option<(Arc<Layers>, Arc<Layers>)>,
    /// For a terrain's surroundings, what they were drawn
    /// from.
    pub(super) around: Option<Around>,
    /// The tiles the last frame found missing, most important first with
    /// their jobs' priorities, every tile it loads and every
    /// tile it keeps: what loads between frames.
    pub(super) missing: Vec<(TileKey, u32)>,
    pub(super) target: HashSet<TileKey>,
    pub(super) keep: HashSet<TileKey>,
    /// The recipe its tiles' jobs generate from, the records of its tiles
    /// asked for (each one's job, the generation it was asked for under
    /// and its priority), its bakes' record while asked for and the frame
    /// it last failed at, and its jobs that failed.
    pub(super) recipe: Option<SharedRecipe>,
    pub(super) dispatched: TileMap<(JobId, Generation, u32)>,
    pub(super) bake: BakeRecord,
    pub(super) failures: JobFailures,
    /// Records of tiles places games name, loading into memory: each one's
    /// job and the generation it was asked for under.
    pub(super) prefetched: TileMap<(JobId, Generation)>,
    /// Its tiles and their settled water cooking, while a session cooks,
    /// and the settled water its resident tiles take.
    pub(super) cook: Option<CookTiles>,
    pub(super) cook_water: Option<CookTiles>,
    pub(super) settled: Settled,
    /// Its lighting, and its asking for it.
    pub(super) lighting: Lighting,
    /// The layers its tiles generate from and its surface's data for the
    /// GPU from them, `None` where the program needs more
    /// than the GPU's interpreter holds; and whether its tiles hold control
    /// values, as without GPU materials.
    pub(super) surface: Option<(Arc<Layers>, Option<Arc<engine_terrain::gpu::SurfaceData>>)>,
    pub(super) materials: bool,
}

/// A terrain's bakes as one record: the job asked for while under way,
/// the frame it last failed at, and the record's key once asked for.
#[derive(Default)]
pub(super) struct BakeRecord {
    pub(super) job: Option<JobId>,
    pub(super) failed: Option<u64>,
    pub(super) key: Option<RecordKey>,
}

/// Bytes of one of a tile's arrays on the GPU, heights or control values:
/// what a tile arriving uploads, once or twice.
pub(super) const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;

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
        Self {
            entity,
            store: TileStore::new(terrain.pyramid, Generation::of(&terrain.edits, true)),
            terrain,
            failed: TileMap::default(),
            selection: Selection::default(),
            collision: Vec::new(),
            standing: 0,
            ranges: Vec::new(),
            baked: None,
            hydrology: HydrologyState::default(),
            erosion: None,
            bridges: Arc::from(Vec::new()),
            falls: Arc::from(Vec::new()),
            edited: None,
            around: None,
            missing: Vec::new(),
            target: HashSet::default(),
            keep: HashSet::default(),
            recipe: None,
            dispatched: TileMap::default(),
            bake: BakeRecord::default(),
            failures: JobFailures::default(),
            prefetched: TileMap::default(),
            cook: None,
            cook_water: None,
            settled: Settled::default(),
            lighting: Lighting::default(),
            surface: None,
            materials: true,
        }
    }

    /// Whether its tiles hold control values, as `gpu` (the settings' GPU
    /// materials) and its surface allow: generated terrains
    /// whose surface the GPU's interpreter holds leave them to the GPU.
    /// Tiles resident when that changes regenerate, drawn until they do.
    pub(super) fn update_materials(&mut self, gpu: bool) {
        let layers = self
            .baked
            .clone()
            .unwrap_or_else(|| self.terrain.layers.clone());
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
            self.store
                .invalidate(Generation::of(&self.terrain.edits, materials), |_| true);
        }
    }

    /// Its jobs and recipe, for the next frame to drop once the state is
    /// gone.
    pub(super) fn retire(&self, retired: &mut Retired) {
        retired
            .jobs
            .extend(self.dispatched.iter().map(|(_, &(id, _, _))| id));
        retired.jobs.extend(self.bake.job);
        retired
            .jobs
            .extend(self.prefetched.iter().map(|(_, &(id, _))| id));
        retired.jobs.extend(
            self.cook
                .iter()
                .chain(&self.cook_water)
                .flat_map(CookTiles::asked),
        );
        retired.jobs.extend(self.settled.asked());
        retired
            .jobs
            .extend(self.lighting.job.iter().map(|&(id, _)| id));
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
    /// any edit added or taken away reaches regenerate, drawn as they were
    /// until then.
    pub(super) fn edit(&mut self, edits: Arc<[TerrainEdit]>) {
        let changed: Vec<Stamp> = edits
            .iter()
            .filter(|edit| !self.terrain.edits.contains(edit))
            .chain(
                self.terrain
                    .edits
                    .iter()
                    .filter(|edit| !edits.contains(edit)),
            )
            .map(|edit| edit.stamp)
            .collect();
        self.terrain.edits = edits;
        self.edited = None;
        if changed.is_empty() {
            return;
        }
        let pyramid = self.terrain.pyramid;
        let under = |key: &TileKey| changed.iter().any(|stamp| reaches(stamp, &pyramid, *key));
        // Tiles kept in memory under the edits are no longer the ground.
        let generation = Generation::of(&self.terrain.edits, self.materials);
        self.store.invalidate(generation, under);
    }

    /// Tile `key`'s job over the recipe shared as `recipe`, carrying the
    /// edits reaching its ground in their order, and the key of its
    /// record.
    pub(super) fn tile_record(&self, recipe: &str, key: TileKey) -> (TileJob, RecordKey) {
        let pyramid = self.terrain.pyramid;
        let job = TileJob {
            recipe: recipe.to_owned(),
            key,
            stamps: self
                .terrain
                .edits
                .iter()
                .map(|edit| edit.stamp)
                .filter(|stamp| reaches(stamp, &pyramid, key))
                .collect(),
        };
        let record = RecordKey::new(
            &engine_terrain::SOURCE,
            TileJob::KIND.name(),
            &job.identity(),
        );
        (job, record)
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

    /// Whether `key` still loads: not resident, or resident under edits
    /// since.
    pub(super) fn loads(&self, key: &TileKey) -> bool {
        self.store.loads(key)
    }

    /// Takes tile `key` out of residency at frame `frame`, keeping it in
    /// memory where it is of the generation tiles are made from now.
    pub(super) fn leave(&mut self, key: &TileKey, frame: u64) {
        if let Some(left) = self.store.leave(key)
            && left.current
        {
            self.store.keep_in_memory(left.tile, left.watered, frame);
        }
    }

    /// Advances whenever its resident tiles, or what stands on them, change.
    pub(super) fn revision(&self) -> u64 {
        self.store.revision().wrapping_add(self.standing)
    }

    /// Checks the residency invariants, the store's (R1, R2 and R5) and
    /// R3: every tile record asked for is for a tile still loading. Panics
    /// naming the first broken.
    #[cfg(debug_assertions)]
    pub(super) fn check(&self) {
        self.store.check();
        for key in self.dispatched.keys() {
            assert!(
                self.store.loads(&key),
                "R3: the record of tile {key:?} is asked for, which no longer loads"
            );
        }
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
            let tile = self.store.get(&key)?;
            let sample = tile.sample(local - pyramid.tile_origin(key))?;
            Some((level, sample))
        })
    }

    /// The water at terrain-local `local` as the tile [`Self::sample`] reads
    /// the ground of holds it: its surface, its depth over that ground, its
    /// flow and whitewater; `None` where it holds none, or over a hole.
    pub(super) fn water(&self, local: Vec2) -> Option<WaterPoint> {
        if self.cut_out(local) {
            return None;
        }
        let pyramid = &self.terrain.pyramid;
        (0..pyramid.levels())
            .find_map(|level| {
                let key = pyramid.tile_at(level, local)?;
                let tile = self.store.get(&key)?;
                let local = local - pyramid.tile_origin(key);
                let sample = tile.sample(local)?;
                Some((!sample.hole).then(|| tile.water_at(local)).flatten())
            })
            .flatten()
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
            let tile = self.store.get(&key)?;
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
    /// Jobs' share: the stacks shared with them and their keys, and the
    /// jobs and recipes of terrains gone, to drop.
    pub(super) stacks: Vec<(Arc<Layers>, String)>,
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

/// Whether `stamp` reaches tile `key`'s ground, or its samples' two beyond
/// (its border and the normals there).
pub(super) fn reaches(stamp: &Stamp, pyramid: &Pyramid, key: TileKey) -> bool {
    let spacing = pyramid.spacing(u32::from(key.level));
    let size = TILE_QUADS as f32 * spacing;
    let first = Vec2::new(f32::from(key.x), f32::from(key.z)) * size - Vec2::splat(2.0 * spacing);
    let last = first + Vec2::splat(size + 4.0 * spacing);
    let center = Vec2::from(stamp.center);
    let radius = Vec2::splat(stamp.radius + stamp.falloff);
    let (low, high) = (center - radius, center + radius);
    first.x <= high.x && first.y <= high.y && last.x >= low.x && last.y >= low.y
}
