use super::*;
use engine_core::{Background, RecordKey};

pub(super) const LOAD_REACH: f32 = 2.0;

pub(super) const KEEP_REACH: f32 = 2.5;

pub(super) const AHEAD_SECONDS: f32 = 2.0;

pub(super) const AHEAD_MAX: f32 = 1000.0;

pub(super) const JUMP: f32 = 100.0;

#[derive(Default)]
pub(super) struct HydrologyState {
    pub(super) stats: Option<HydrologyStats>,
    pub(super) bake: Option<Arc<crate::HydrologyBake>>,
}

impl HydrologyState {
    pub(super) fn baked(water: &crate::HydrologyBake) -> Self {
        Self {
            stats: Some(water.stats),
            bake: Some(Arc::new(water.clone())),
        }
    }
}

fn with_edits(
    base: Arc<Layers>,
    edits: &[TerrainEdit],
    kinds: &[&'static LayerKind],
) -> Arc<Layers> {
    if edits.is_empty() {
        return base;
    }
    let stamps: Vec<Stamp> = edits.iter().map(|edit| edit.stamp).collect();
    base.with_stamps(&stamps, kinds).map_or(base, Arc::new)
}

pub(super) type TileWork = Background<Arc<Tile>>;

#[derive(Clone, Copy)]
enum TileUse {
    Needed { record: RecordKey, priority: u32 },
    Ahead,
}

pub(super) struct TileSlot {
    work: TileWork,
    made: Generation,
    asked: TileUse,
}

impl TileSlot {
    pub(super) fn needed(
        work: TileWork,
        (record, made, priority): (RecordKey, Generation, u32),
    ) -> Self {
        Self {
            work,
            made,
            asked: TileUse::Needed { record, priority },
        }
    }

    pub(super) fn ahead(work: TileWork, made: Generation) -> Self {
        Self {
            work,
            made,
            asked: TileUse::Ahead,
        }
    }

    pub(super) fn loading(&self) -> Option<(Generation, u32)> {
        match self.asked {
            TileUse::Needed { priority, .. } if self.work.failure().is_none() => {
                Some((self.made, priority))
            }
            TileUse::Needed { .. } | TileUse::Ahead => None,
        }
    }

    pub(super) fn prefetching(&self) -> Option<Generation> {
        match self.asked {
            TileUse::Ahead => Some(self.made),
            TileUse::Needed { .. } => None,
        }
    }

    pub(super) fn failed(&self) -> Option<RecordKey> {
        match self.asked {
            TileUse::Needed { record, .. } if self.work.failure().is_some() => Some(record),
            TileUse::Needed { .. } | TileUse::Ahead => None,
        }
    }

    pub(super) fn work_mut(&mut self) -> Option<&mut TileWork> {
        if self.failed().is_some() {
            return None;
        }
        Some(&mut self.work)
    }
}

pub(super) struct TerrainState {
    pub(super) entity: Entity,
    pub(super) terrain: Terrain,
    pub(super) store: TileStore,
    pub(super) slots: TileMap<TileSlot>,
    pub(super) drawn: bool,
    pub(super) collision: Vec<TileKey>,
    pub(super) ranges: Vec<f32>,
    pub(super) baked: Option<Arc<Layers>>,
    pub(super) baked_rivers: Arc<[TerrainRiver]>,
    pub(super) hydrology: HydrologyState,
    pub(super) bridges: Arc<[Bridge]>,
    pub(super) current: Arc<Layers>,
    pub(super) changed: bool,
    pub(super) ground_changed: bool,
    pub(super) hydrology_changed: bool,
    pub(super) reshaped: Option<Vec<(Vec2, Vec2)>>,
    pub(super) around: Option<Around>,
    pub(super) missing: Vec<(TileKey, u32)>,
    pub(super) target: HashSet<TileKey>,
    pub(super) keep: HashSet<TileKey>,
    pub(super) recipe: Option<SharedRecipe>,
    pub(super) bake: Option<TerrainBake>,
    pub(super) failures: JobFailures,
    pub(super) surface: Option<(Arc<Layers>, Option<Arc<SurfaceData>>)>,
    pub(super) materials: TileMaterials,
}

pub(super) struct TerrainBake {
    pub(super) work: Background<Arc<Layers>>,
    pub(super) drawn: Arc<[TerrainRiver]>,
}

pub(super) const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;

pub(super) struct Around {
    pub(super) base: Arc<Layers>,
    pub(super) surroundings: Surroundings,
    pub(super) cutout: (Vec2, Vec2),
    pub(super) entity: Option<Entity>,
}

impl TerrainState {
    pub(super) fn new(entity: Entity, terrain: Terrain, kinds: &[&'static LayerKind]) -> Self {
        Self {
            entity,
            store: TileStore::new(terrain.pyramid, Generation::of(&terrain.edits, None)),
            current: with_edits(terrain.layers.clone(), &terrain.edits, kinds),
            terrain,
            slots: TileMap::default(),
            drawn: false,
            collision: Vec::new(),
            ranges: Vec::new(),
            baked: None,
            baked_rivers: Arc::from(Vec::new()),
            hydrology: HydrologyState::default(),
            bridges: Arc::from(Vec::new()),
            changed: true,
            ground_changed: true,
            hydrology_changed: true,
            reshaped: None,
            around: None,
            missing: Vec::new(),
            target: HashSet::default(),
            keep: HashSet::default(),
            recipe: None,
            bake: None,
            failures: JobFailures::default(),
            surface: None,
            materials: TileMaterials::Baked,
        }
    }

    pub(super) fn baking(&self) -> bool {
        self.bake.as_ref().is_some_and(|bake| bake.work.waiting())
    }

    pub(super) fn loading(&self) -> usize {
        self.slots
            .keys_where(|_, slot| slot.loading().is_some())
            .len()
    }

    pub(super) fn prefetching(&self) -> usize {
        self.slots
            .keys_where(|_, slot| slot.prefetching().is_some())
            .len()
    }

    pub(super) fn failed(&self, key: &TileKey) -> bool {
        self.slots
            .get(key)
            .is_some_and(|slot| slot.failed().is_some())
    }

    pub(super) fn show(&mut self, drawn: bool) {
        if self.drawn != drawn {
            self.drawn = drawn;
            self.changed = true;
        }
    }

    fn refresh_layers(&mut self, kinds: &[&'static LayerKind]) {
        let base = self
            .baked
            .clone()
            .unwrap_or_else(|| self.terrain.layers.clone());
        self.current = with_edits(base, &self.terrain.edits, kinds);
    }

    pub(super) fn update_materials(&mut self, gpu: bool) {
        let layers = &self.current;
        let materials = if gpu && layers.unbaked().is_none() {
            let current = self
                .surface
                .as_ref()
                .is_some_and(|(held, _)| Arc::ptr_eq(held, layers));
            if !current {
                let data = SurfaceData::new(&self.terrain.surface.rules, layers).map(Arc::new);
                self.surface = Some((layers.clone(), data));
                self.changed = true;
            }
            match self.surface.as_ref() {
                Some((_, Some(_))) => TileMaterials::Program,
                Some((_, None)) | None => TileMaterials::Baked,
            }
        } else {
            TileMaterials::Baked
        };
        if materials != self.materials {
            self.materials = materials;
            self.changed = true;
        }
    }

    pub(super) fn programmed(&self) -> Option<&Arc<SurfaceData>> {
        let drawn =
            self.materials == TileMaterials::Program || self.store.holds(TileMaterials::Program);
        self.surface
            .as_ref()
            .and_then(|(_, data)| data.as_ref())
            .filter(|_| drawn)
    }

    pub(super) fn retire(self, retired: &mut Retired) {
        retired
            .shares
            .extend(self.recipe.as_ref().map(|recipe| recipe.key));
        retired
            .entities
            .extend(self.around.and_then(|around| around.entity));
    }

    pub(super) fn generation(&self) -> Generation {
        Generation::of(
            &self.terrain.edits,
            self.recipe.as_ref().map(|recipe| recipe.key),
        )
    }

    fn cut_out(&self, local: Vec2) -> bool {
        self.around.as_ref().is_some_and(|around| {
            let (low, high) = around.cutout;
            local.x > low.x && local.y > low.y && local.x < high.x && local.y < high.y
        })
    }

    pub(super) fn edit(&mut self, edits: Arc<[TerrainEdit]>, kinds: &[&'static LayerKind]) {
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
        self.refresh_layers(kinds);
        self.ground_changed = true;
        if changed.is_empty() {
            return;
        }
        let pyramid = self.terrain.pyramid;
        let under = |key: &TileKey| changed.iter().any(|stamp| reaches(stamp, &pyramid, *key));
        let generation = self.generation();
        self.store.invalidate(generation, under);
    }

    pub(super) fn bake_source(
        &self,
        authored: &Arc<Layers>,
        kinds: &[&'static LayerKind],
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
        Arc::new(authored.with_drawn(&drawn, self.baked.as_deref(), kinds))
    }

    pub(super) fn redrawn(&mut self, drawn: Arc<[TerrainRiver]>, before: Option<&Layers>) {
        self.baked_rivers = drawn;
        self.reshaped = match (before, self.baked.as_deref()) {
            (Some(before), Some(after)) => after.changes_from(before),
            _ => None,
        };
    }

    pub(super) fn recipe_changed(&mut self, materials: Option<TileMaterials>) {
        let regions = self
            .reshaped
            .take()
            .filter(|_| materials == Some(self.materials));
        let pyramid = self.terrain.pyramid;
        let under = |key: &TileKey| {
            regions
                .as_ref()
                .is_none_or(|regions| regions.iter().any(|&bounds| covers(bounds, &pyramid, *key)))
        };
        let generation = self.generation();
        self.store.invalidate(generation, under);
        self.ground_changed = true;
    }

    fn reaching_stamps(&self, key: TileKey) -> impl Iterator<Item = Stamp> + '_ {
        let pyramid = self.terrain.pyramid;
        self.terrain
            .edits
            .iter()
            .map(|edit| edit.stamp)
            .filter(move |stamp| reaches(stamp, &pyramid, key))
    }

    pub(super) fn reaching(&self, key: TileKey) -> Arc<[Stamp]> {
        self.reaching_stamps(key).collect()
    }

    pub(super) fn tile_job(&self, recipe: [u8; 32], key: TileKey) -> TileJob {
        TileJob {
            recipe,
            key,
            stamps: self.reaching_stamps(key).collect(),
        }
    }

    pub(super) fn finish_bake(&mut self, baked: Arc<Layers>, kinds: &[&'static LayerKind]) {
        if baked.unbaked().is_none() {
            self.bridges = baked.bridges().into();
        }
        self.baked = Some(baked);
        self.refresh_layers(kinds);
        self.changed = true;
    }

    pub(super) fn loads(&self, key: &TileKey) -> bool {
        self.store.loads(key)
    }

    pub(super) fn leave(&mut self, key: &TileKey, frame: u64) {
        if let Some(left) = self.store.leave(key)
            && left.current
        {
            self.store.keep_in_memory(left, frame);
        }
    }

    #[cfg(debug_assertions)]
    pub(super) fn check(&self) {
        self.store.check();
        for key in self.slots.keys_where(|_, slot| slot.loading().is_some()) {
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
    pub(super) stacks: Vec<(Arc<Layers>, RecordKey)>,
    pub(super) retired: Retired,
    pub(super) restored: u64,
    pub(super) changed: bool,
    pub(super) settling: Settling,
}

#[derive(Default)]
pub(super) struct Retired {
    pub(super) shares: Vec<RecordKey>,
    pub(super) entities: Vec<Entity>,
}
