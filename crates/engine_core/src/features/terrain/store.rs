use super::*;

pub(super) type Watered = Option<Arc<engine_terrain::Rivers>>;

struct Resident {
    tile: Arc<Tile>,
    made: Generation,
    watered: Watered,
}

struct CachedTile {
    tile: Arc<Tile>,
    watered: Watered,
    bytes: u64,
    left: u64,
}

pub(super) struct Left {
    pub(super) tile: Arc<Tile>,
    pub(super) watered: Watered,
    pub(super) current: bool,
}

fn memory_bytes(tile: &Tile) -> u64 {
    let arrays = tile.heights().len() + tile.control().map_or(0, <[u32]>::len);
    let holes = tile.hole_bits().map_or(0, <[u64]>::len) * 8;
    let shading = tile.shading().map_or(0, |shading| shading.bytes());
    (arrays * 4 + holes + std::mem::size_of_val(tile.placed())) as u64 + shading
}

pub(super) struct TileStore {
    pyramid: Pyramid,
    tiles: TileMap<Resident>,
    generation: Generation,
    revision: u64,
    errors: Vec<f32>,
    cache: TileMap<CachedTile>,
    cached_bytes: u64,
}

impl TileStore {
    pub(super) fn new(pyramid: Pyramid, generation: Generation) -> Self {
        Self {
            pyramid,
            tiles: TileMap::default(),
            generation,
            revision: 1,
            errors: vec![0.0; pyramid.levels() as usize],
            cache: TileMap::default(),
            cached_bytes: 0,
        }
    }

    pub(super) fn get(&self, key: &TileKey) -> Option<&Arc<Tile>> {
        self.tiles.get(key).map(|resident| &resident.tile)
    }

    pub(super) fn watered(&self, key: &TileKey) -> Option<&Watered> {
        self.tiles.get(key).map(|resident| &resident.watered)
    }

    pub(super) fn contains(&self, key: &TileKey) -> bool {
        self.tiles.contains_key(key)
    }

    pub(super) fn generation(&self) -> Generation {
        self.generation
    }

    pub(super) fn loads(&self, key: &TileKey) -> bool {
        self.tiles
            .get(key)
            .is_none_or(|resident| resident.made != self.generation)
    }

    pub(super) fn len(&self) -> usize {
        self.tiles.len()
    }

    pub(super) fn keys_where(&self, wanted: impl Fn(&TileKey) -> bool) -> Vec<TileKey> {
        self.tiles.keys_where(|key, _| wanted(key))
    }

    pub(super) fn tiles(&self) -> impl Iterator<Item = &Arc<Tile>> {
        self.tiles.iter().map(|(_, resident)| &resident.tile)
    }

    pub(super) fn revision(&self) -> u64 {
        self.revision
    }

    pub(super) fn errors(&self) -> &[f32] {
        &self.errors
    }

    pub(super) fn arrive(&mut self, tile: Arc<Tile>, watered: Watered) {
        let made = self.generation;
        self.put(tile, made, watered);
    }

    pub(super) fn rewater(&mut self, key: &TileKey, tile: Option<Arc<Tile>>, watered: Watered) {
        let Some(resident) = self.tiles.get_mut(key) else {
            return;
        };
        resident.watered = watered;
        if let Some(tile) = tile {
            resident.tile = tile;
            self.revision += 1;
        }
    }

    pub(super) fn reshade(&mut self, key: &TileKey, tile: Arc<Tile>) {
        let Some(resident) = self.tiles.get_mut(key) else {
            return;
        };
        resident.tile = tile;
        self.revision += 1;
    }

    fn put(&mut self, tile: Arc<Tile>, made: Generation, watered: Watered) {
        let key = tile.key();
        let level = usize::from(key.level);
        if let Some(parent) = self
            .pyramid
            .parent(key)
            .and_then(|parent| self.tiles.get(&parent))
        {
            self.errors[level] =
                crate::math::max(self.errors[level], tile.morph_error(&parent.tile));
        }
        if let Some(finer) = level.checked_sub(1) {
            for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let child = TileKey::new(finer as u8, key.x * 2 + dx, key.z * 2 + dz);
                if let Some(child) = self.tiles.get(&child) {
                    self.errors[finer] =
                        crate::math::max(self.errors[finer], child.tile.morph_error(&tile));
                }
            }
        }
        self.tiles.insert(
            key,
            Resident {
                tile,
                made,
                watered,
            },
        );
        self.revision += 1;
    }

    pub(super) fn leave(&mut self, key: &TileKey) -> Option<Left> {
        let resident = self.tiles.remove(key)?;
        self.revision += 1;
        Some(Left {
            current: resident.made == self.generation,
            tile: resident.tile,
            watered: resident.watered,
        })
    }

    pub(super) fn invalidate(&mut self, generation: Generation, under: impl Fn(&TileKey) -> bool) {
        let previous = std::mem::replace(&mut self.generation, generation);
        for key in self
            .tiles
            .keys_where(|key, resident| resident.made == previous && !under(key))
        {
            if let Some(resident) = self.tiles.get_mut(&key) {
                resident.made = generation;
            }
        }
        for key in self.cache.keys_where(|key, _| under(key)) {
            self.forget(key);
        }
        self.revision += 1;
    }

    pub(super) fn keep_in_memory(&mut self, tile: Arc<Tile>, watered: Watered, frame: u64) {
        let key = tile.key();
        let bytes = memory_bytes(&tile);
        self.forget(key);
        self.cached_bytes += bytes;
        self.cache.insert(
            key,
            CachedTile {
                tile,
                watered,
                bytes,
                left: frame,
            },
        );
    }

    pub(super) fn forget(&mut self, key: TileKey) {
        if let Some(cached) = self.cache.remove(&key) {
            self.cached_bytes -= cached.bytes;
        }
    }

    pub(super) fn in_memory(&self, key: &TileKey) -> bool {
        self.cache.contains_key(key)
    }

    pub(super) fn wanted_in_memory(&mut self, key: &TileKey, frame: u64) {
        if let Some(cached) = self.cache.get_mut(key) {
            cached.left = frame;
        }
    }

    pub(super) fn kept(&self) -> impl Iterator<Item = (u64, TileKey)> + '_ {
        self.cache.iter().map(|(key, kept)| (kept.left, key))
    }

    pub(super) fn kept_tiles(&self) -> usize {
        self.cache.len()
    }

    pub(super) fn cached_bytes(&self) -> u64 {
        self.cached_bytes
    }

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
            if let Some(cached) = self.cache.remove(&key) {
                self.cached_bytes -= cached.bytes;
                work.upload(2 * TILE_ARRAY_BYTES);
                self.arrive(cached.tile, cached.watered);
                restored += 1;
            }
        }
        restored
    }

    #[cfg(debug_assertions)]
    pub(super) fn check(&self) {
        let levels = self.pyramid.levels();
        for (key, resident) in self.tiles.iter() {
            assert!(
                resident.tile.key() == key && u32::from(key.level) < levels,
                "R1: resident tile {key:?} holds tile {:?} ({levels} levels)",
                resident.tile.key()
            );
        }
        let mut bytes = 0;
        for (key, cached) in self.cache.iter() {
            assert!(
                cached.tile.key() == key,
                "R1: tile {key:?} kept in memory holds tile {:?}",
                cached.tile.key()
            );
            bytes += cached.bytes;
        }
        assert!(
            bytes == self.cached_bytes,
            "R5: tiles kept in memory hold {bytes} bytes, the store counts {}",
            self.cached_bytes
        );
    }
}
