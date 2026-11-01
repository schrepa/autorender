use super::*;

pub(super) const RETRY_AFTER: u64 = 120;

pub(super) const LOAD_REACH: f32 = 2.0;

pub(super) const KEEP_REACH: f32 = 2.5;

pub(super) const AHEAD_SECONDS: f32 = 2.0;

pub(super) const AHEAD_MAX: f32 = 1000.0;

pub(super) const JUMP: f32 = 100.0;

#[derive(Default)]
pub(super) struct HydrologyState {
    pub(super) stats: Option<HydrologyStats>,
    pub(super) bake: Option<(engine_core::Stamp, Arc<crate::HydrologyBake>)>,
}

impl HydrologyState {
    pub(super) fn baked(water: &crate::HydrologyBake) -> Self {
        Self {
            stats: Some(water.stats),
            bake: Some((engine_core::Stamp::next(), Arc::new(water.clone()))),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JobFailures {
    pub count: u64,
    pub last: Option<(u64, String)>,
}

impl JobFailures {
    pub fn record(&mut self, frame: u64, error: String) {
        self.count += 1;
        self.last = Some((frame, error));
    }
}

pub(super) struct TerrainState {
    pub(super) entity: Entity,
    pub(super) terrain: Terrain,
    pub(super) store: TileStore,
    pub(super) failed: TileMap<u64>,
    pub(super) drawn: bool,
    pub(super) collision: Vec<TileKey>,
    pub(super) standing: u64,
    pub(super) solids: Arc<[Option<Solid>]>,
    pub(super) relief: u128,
    pub(super) ranges: Vec<f32>,
    pub(super) baked: Option<Arc<Layers>>,
    pub(super) baked_rivers: Arc<[TerrainRiver]>,
    pub(super) hydrology: HydrologyState,
    pub(super) bridges: Arc<[Bridge]>,
    pub(super) edited: Option<(Arc<Layers>, Arc<Layers>)>,
    pub(super) around: Option<Around>,
    pub(super) missing: Vec<(TileKey, u32)>,
    pub(super) target: HashSet<TileKey>,
    pub(super) keep: HashSet<TileKey>,
    pub(super) recipe: Option<SharedRecipe>,
    pub(super) dispatched: TileMap<(Ticket, Generation, u32)>,
    pub(super) bake: BakeRecord,
    pub(super) failures: JobFailures,
    pub(super) prefetched: TileMap<(Ticket, Generation)>,
    pub(super) cook: Option<CookTiles>,
    pub(super) surface: Option<(Arc<Layers>, Option<Arc<crate::gpu::SurfaceData>>)>,
    pub(super) materials: bool,
}

#[derive(Default)]
pub(super) struct BakeRecord {
    pub(super) job: Option<Ticket>,
    pub(super) failed: Option<u64>,
    pub(super) drawn: Option<Arc<[TerrainRiver]>>,
}

pub(super) const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;

pub(super) struct Around {
    pub(super) base: Arc<Layers>,
    pub(super) surroundings: Surroundings,
    pub(super) cutout: (Vec2, Vec2),
}

impl TerrainState {
    pub(super) fn new(entity: Entity, terrain: Terrain) -> Self {
        Self {
            entity,
            store: TileStore::new(terrain.pyramid, Generation::of(&terrain.edits, &[], true)),
            terrain,
            failed: TileMap::default(),
            drawn: false,
            collision: Vec::new(),
            standing: 0,
            solids: Arc::from(Vec::new()),
            relief: 0,
            ranges: Vec::new(),
            baked: None,
            baked_rivers: Arc::from(Vec::new()),
            hydrology: HydrologyState::default(),
            bridges: Arc::from(Vec::new()),
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
            surface: None,
            materials: true,
        }
    }

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
                let data = crate::gpu::SurfaceData::new(&self.terrain.surface.rules, &layers)
                    .map(Arc::new);
                self.surface = Some((layers, data));
            }
            self.surface.as_ref().is_none_or(|(_, data)| data.is_none())
        } else {
            true
        };
        if materials != self.materials {
            self.materials = materials;
            self.store.invalidate(
                Generation::of(&self.terrain.edits, &self.baked_rivers, materials),
                |_| true,
            );
        }
    }

    pub(super) fn retire(self, retired: &mut Retired) {
        retired
            .jobs
            .extend(self.dispatched.into_values().map(|(ticket, _, _)| ticket));
        retired.jobs.extend(self.bake.job);
        retired
            .jobs
            .extend(self.prefetched.into_values().map(|(ticket, _)| ticket));
        retired
            .jobs
            .extend(self.cook.into_iter().flat_map(CookTiles::asked));
    }

    fn cut_out(&self, local: Vec2) -> bool {
        self.around.as_ref().is_some_and(|around| {
            let (low, high) = around.cutout;
            local.x > low.x && local.y > low.y && local.x < high.x && local.y < high.y
        })
    }

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
        let generation = Generation::of(&self.terrain.edits, &self.baked_rivers, self.materials);
        self.store.invalidate(generation, under);
    }

    pub(super) fn bake_source(
        &self,
        authored: &Arc<Layers>,
        make: Option<engine_places::terrain::DrawRivers>,
    ) -> Arc<Layers> {
        let base = self.baked.clone().unwrap_or_else(|| authored.clone());
        if *self.baked_rivers == *self.terrain.rivers || !authored.has_hydrology() {
            return base;
        }
        let drawn: Vec<GivenRiver> = self
            .terrain
            .rivers
            .iter()
            .map(|drawn| drawn.river.clone())
            .collect();
        Arc::new(authored.with_drawn(&drawn, self.baked.as_deref(), make))
    }

    pub(super) fn redrawn(&mut self, drawn: Arc<[TerrainRiver]>, before: Option<&Layers>) {
        self.baked_rivers = drawn;
        let changed = match (before, self.baked.as_deref()) {
            (Some(before), Some(after)) => after.changes_from(before),
            _ => None,
        };
        let pyramid = self.terrain.pyramid;
        let under = |key: &TileKey| {
            changed
                .as_ref()
                .is_none_or(|regions| regions.iter().any(|&bounds| covers(bounds, &pyramid, *key)))
        };
        let generation = Generation::of(&self.terrain.edits, &self.baked_rivers, self.materials);
        self.store.invalidate(generation, under);
    }

    pub(super) fn tile_job(&self, recipe: [u8; 32], key: TileKey) -> TileJob {
        let pyramid = self.terrain.pyramid;
        TileJob {
            recipe,
            key,
            stamps: self
                .terrain
                .edits
                .iter()
                .map(|edit| edit.stamp)
                .filter(|stamp| reaches(stamp, &pyramid, key))
                .collect(),
        }
    }

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
        match crate::layers::with_stamps(&base, &stamps) {
            Ok(edited) => {
                let edited = Arc::new(edited);
                self.edited = Some((base, edited.clone()));
                edited
            }
            Err(_) => base,
        }
    }

    pub(super) fn loads(&self, key: &TileKey) -> bool {
        self.store.loads(key)
    }

    pub(super) fn leave(&mut self, key: &TileKey, frame: u64) {
        if let Some(left) = self.store.leave(key)
            && left.current
        {
            self.store.keep_in_memory(left.tile, frame);
        }
    }

    pub(super) fn revision(&self) -> u64 {
        self.store.revision().wrapping_add(self.standing)
    }

    #[cfg(debug_assertions)]
    pub(super) fn check(&self) {
        self.store.check();
        for key in self.dispatched.keys() {
            assert!(self.store.loads(&key), "dispatched_tile_loads {key:?}");
        }
    }

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
}

#[derive(Resource, Default)]
pub struct Terrains {
    pub(super) entries: Vec<TerrainState>,
    pub(super) frame: u64,
    pub(super) stats: TerrainStats,
    pub(super) last_eye: Option<Vec3>,
    pub(super) velocity: Vec3,
    pub(super) seconds: f32,
    pub(super) fast_ago: Option<f32>,
    pub(super) loaded: bool,
    pub(super) stacks: Vec<(Arc<Layers>, [u8; 32])>,
    pub(super) retired: Retired,
    pub(super) restored: u64,
    pub(super) upload_left: u64,
    pub(super) between: u64,
    pub(crate) refused: Vec<(Entity, Error)>,
    pub(super) epoch: u64,
}

#[derive(Default)]
pub(super) struct Retired {
    pub(super) jobs: Vec<Ticket>,
}

pub(crate) fn reaches(stamp: &Stamp, pyramid: &Pyramid, key: TileKey) -> bool {
    let center = Vec2::from(stamp.center);
    let radius = Vec2::splat(stamp.radius + stamp.falloff);
    covers((center - radius, center + radius), pyramid, key)
}

pub(crate) fn covers((low, high): (Vec2, Vec2), pyramid: &Pyramid, key: TileKey) -> bool {
    let spacing = pyramid.spacing(u32::from(key.level));
    let size = TILE_QUADS as f32 * spacing;
    let first = Vec2::new(f32::from(key.x), f32::from(key.z)) * size - Vec2::splat(2.0 * spacing);
    let last = first + Vec2::splat(size + 4.0 * spacing);
    first.x <= high.x && first.y <= high.y && last.x >= low.x && last.y >= low.y
}
