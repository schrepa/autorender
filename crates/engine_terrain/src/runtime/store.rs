use super::*;
use engine_places::ground::TileFace;

struct Resident {
    tile: Arc<Tile>,
    stamps: Arc<[Stamp]>,
    made: Generation,
    face: TileFace,
}

struct CachedTile {
    tile: Arc<Tile>,
    stamps: Arc<[Stamp]>,
    face: TileFace,
    bytes: u64,
    left: u64,
}

pub(super) struct Left {
    pub(super) tile: Arc<Tile>,
    pub(super) stamps: Arc<[Stamp]>,
    pub(super) face: TileFace,
    pub(super) current: bool,
}

fn memory_bytes(tile: &Tile) -> u64 {
    let arrays = tile.heights().len() + tile.control().map_or(0, <[u32]>::len);
    let holes = tile.hole_bits().map_or(0, <[u64]>::len) * 8;
    (arrays * 4 + holes + std::mem::size_of_val(tile.placed())) as u64
}

fn face_of(
    tile: &Tile,
    (stamps, stamp): (&Arc<[Stamp]>, engine_core::Stamp),
    pyramid: &Pyramid,
) -> TileFace {
    let key = tile.key();
    let corner = pyramid.tile_origin(key);
    let size = TILE_QUADS as f32 * tile.spacing();
    let [low, high] = tile.bounds();
    TileFace {
        key,
        stamp,
        bounds: (
            Vec3::new(corner.x, low, corner.y),
            Vec3::new(corner.x + size, high, corner.y + size),
        ),
        level: key.level,
        spacing: tile.spacing(),
        heights: tile.shared_heights(),
        holes: tile.shared_holes(),
        control: tile.shared_control(),
        river_edges: tile.shared_river_edges(),
        river_beds: tile.shared_river_beds(),
        rivers: tile.river_texels(),
        patch_bounds: tile.shared_patch_bounds(),
        patch_holes: *tile.patch_holes(),
        ground_errors: tile.ground_errors().shared_halves(),
        stamps: Arc::clone(stamps),
        placed: tile.shared_placed(),
        current: true,
    }
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

    pub(super) fn known(&self, key: &TileKey) -> Option<&Arc<Tile>> {
        self.get(key)
            .or_else(|| self.cache.get(key).map(|cached| &cached.tile))
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

    pub(super) fn faces(&self) -> impl Iterator<Item = &TileFace> {
        self.tiles.iter().map(|(_, resident)| &resident.face)
    }

    pub(super) fn revision(&self) -> u64 {
        self.revision
    }

    pub(super) fn errors(&self) -> &[f32] {
        &self.errors
    }

    pub(super) fn arrive(&mut self, tile: Arc<Tile>, stamps: Arc<[Stamp]>) {
        let made = self.generation;
        let face = face_of(&tile, (&stamps, engine_core::Stamp::next()), &self.pyramid);
        self.put(tile, stamps, face, made);
    }

    pub(super) fn keep_unseen(&mut self, tile: Arc<Tile>, stamps: Arc<[Stamp]>, frame: u64) {
        let face = face_of(&tile, (&stamps, engine_core::Stamp::next()), &self.pyramid);
        let left = Left {
            tile,
            stamps,
            face,
            current: true,
        };
        self.keep_in_memory(left, frame);
    }

    fn put(&mut self, tile: Arc<Tile>, stamps: Arc<[Stamp]>, face: TileFace, made: Generation) {
        let key = tile.key();
        let level = usize::from(key.level);
        if let Some(parent) = self
            .pyramid
            .parent(key)
            .and_then(|parent| self.tiles.get(&parent))
        {
            self.errors[level] =
                engine_math::max(self.errors[level], tile.morph_error(&parent.tile));
        }
        if let Some(finer) = level.checked_sub(1) {
            for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let child = TileKey::new(finer as u8, key.x * 2 + dx, key.z * 2 + dz);
                if let Some(child) = self.tiles.get(&child) {
                    self.errors[finer] =
                        engine_math::max(self.errors[finer], child.tile.morph_error(&tile));
                }
            }
        }
        self.tiles.insert(
            key,
            Resident {
                tile,
                stamps,
                made,
                face,
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
            stamps: resident.stamps,
            face: resident.face,
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
        for resident in self.tiles.values_mut() {
            resident.face.current = resident.made == generation;
        }
        self.revision += 1;
    }

    pub(super) fn keep_in_memory(&mut self, left: Left, frame: u64) {
        let key = left.tile.key();
        let bytes = memory_bytes(&left.tile);
        self.forget(key);
        self.cached_bytes += bytes;
        self.cache.insert(
            key,
            CachedTile {
                tile: left.tile,
                stamps: left.stamps,
                face: left.face,
                bytes,
                left: frame,
            },
        );
    }

    pub(super) fn kept_faces(&self) -> impl Iterator<Item = (TileKey, engine_core::Stamp)> + '_ {
        self.cache.iter().map(|(key, kept)| (key, kept.face.stamp))
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
                let made = self.generation;
                let face = TileFace {
                    current: true,
                    ..cached.face
                };
                self.put(cached.tile, cached.stamps, face, made);
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
                "resident_tile_key {key:?} {:?} {levels}",
                resident.tile.key()
            );
        }
        let mut bytes = 0;
        for (key, cached) in self.cache.iter() {
            assert!(
                cached.tile.key() == key,
                "cached_tile_key {key:?} {:?}",
                cached.tile.key()
            );
            bytes += cached.bytes;
        }
        assert!(
            bytes == self.cached_bytes,
            "cached_bytes_counted {bytes} {}",
            self.cached_bytes
        );
    }
}
