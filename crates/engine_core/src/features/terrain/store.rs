//! A terrain's resident tiles, and the tiles kept in memory after they
//! left: the one owner of what each resident tile is. Each holds how much
//! of it is here (its [`Phase`]) and the [`Generation`] it was made from;
//! a tile of another generation than its terrain makes now lies under an
//! edit since (or was made before its terrain's materials changed) and
//! regenerates, drawn until then. The
//! store measures the levels' morph errors as tiles arrive and advances its
//! revision with every change, which renderers and collision follow. Tiles
//! become resident through [`TileStore::arrive`] alone.

use super::*;

/// How much of a resident tile is here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    /// Its heights and holes alone: drawn with coarser levels' materials
    /// and collided with until the whole tile arrives, which it loads for.
    Shape,
    /// Whole but for its objects, which jobs place in bands of rows; it
    /// loads until they join (and generates whole where they stop).
    Placing,
    /// Whole but for the objects standing far away its level places, which
    /// a job places after.
    Far,
    /// Everything it holds.
    Whole,
}

/// A resident tile: the tile, its phase, the generation it was made from
/// and the arrival that made it resident, which work on it (a far job)
/// belongs to.
struct Resident {
    tile: Arc<Tile>,
    phase: Phase,
    made: Generation,
    arrival: u64,
}

/// A tile kept in memory after it left the view: its bytes there and the
/// frame it left, or the last a place a game names wanted it.
struct CachedTile {
    tile: Arc<Tile>,
    bytes: u64,
    left: u64,
}

/// A resident tile that left: the tile, and whether it was whole and of
/// the generation tiles are made from now.
pub(super) struct Left {
    pub(super) tile: Arc<Tile>,
    pub(super) current: bool,
}

/// Bytes `tile` holds in memory.
fn memory_bytes(tile: &Tile) -> u64 {
    let arrays = tile.heights().len() + tile.control().map_or(0, <[u32]>::len);
    let holes = tile.hole_bits().map_or(0, <[u64]>::len) * 8;
    (arrays * 4 + holes + std::mem::size_of_val(tile.placed())) as u64
}

pub(super) struct TileStore {
    pyramid: Pyramid,
    tiles: HashMap<TileKey, Resident>,
    /// The generation tiles arriving now are made from.
    generation: Generation,
    /// Arrivals so far.
    arrivals: u64,
    revision: u64,
    /// The largest height difference measured between each level and the
    /// next coarser one ([`Tile::morph_error`]), what its patches morph by;
    /// zero until measured.
    errors: Vec<f32>,
    /// Tiles that left whole, kept in memory to return at once while the
    /// layers they generated from stay, and their bytes.
    cache: HashMap<TileKey, CachedTile>,
    cached_bytes: u64,
    /// The most memory one tile kept took: the room each tile generated
    /// ahead of need is given.
    tile_bytes: u64,
}

impl TileStore {
    /// No tiles of `pyramid`, which tiles of `generation` arrive at.
    pub(super) fn new(pyramid: Pyramid, generation: Generation) -> Self {
        Self {
            pyramid,
            tiles: HashMap::default(),
            generation,
            arrivals: 0,
            revision: 1,
            errors: vec![0.0; pyramid.levels() as usize],
            cache: HashMap::default(),
            cached_bytes: 0,
            tile_bytes: 0,
        }
    }

    /// The resident tile `key`.
    pub(super) fn get(&self, key: &TileKey) -> Option<&Arc<Tile>> {
        self.tiles.get(key).map(|resident| &resident.tile)
    }

    pub(super) fn contains(&self, key: &TileKey) -> bool {
        self.tiles.contains_key(key)
    }

    /// The phase of the resident tile `key`.
    pub(super) fn phase(&self, key: &TileKey) -> Option<Phase> {
        self.tiles.get(key).map(|resident| resident.phase)
    }

    /// The arrival that made tile `key` resident.
    pub(super) fn arrival(&self, key: &TileKey) -> Option<u64> {
        self.tiles.get(key).map(|resident| resident.arrival)
    }

    /// The generation tiles are made from now.
    pub(super) fn generation(&self) -> Generation {
        self.generation
    }

    /// Whether resident tile `key` was made from another generation than
    /// tiles are made from now.
    pub(super) fn stale(&self, key: &TileKey) -> bool {
        self.tiles
            .get(key)
            .is_some_and(|resident| resident.made != self.generation)
    }

    /// Whether `key` still loads: not resident, resident of another
    /// generation, or resident without its materials or objects in bands.
    pub(super) fn loads(&self, key: &TileKey) -> bool {
        self.tiles.get(key).is_none_or(|resident| {
            resident.made != self.generation
                || matches!(resident.phase, Phase::Shape | Phase::Placing)
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Resident tiles in `phases`.
    pub(super) fn count(&self, phases: &[Phase]) -> usize {
        self.tiles
            .values()
            .filter(|resident| phases.contains(&resident.phase))
            .count()
    }

    /// The resident tiles' keys.
    pub(super) fn keys(&self) -> impl Iterator<Item = &TileKey> {
        self.tiles.keys()
    }

    /// The resident tiles.
    pub(super) fn tiles(&self) -> impl Iterator<Item = &Arc<Tile>> {
        self.tiles.values().map(|resident| &resident.tile)
    }

    /// Advances whenever the resident tiles change, or which of them
    /// regenerate.
    pub(super) fn revision(&self) -> u64 {
        self.revision
    }

    /// Each level's largest measured morph error.
    pub(super) fn errors(&self) -> &[f32] {
        &self.errors
    }

    /// Makes `tile` resident in `phase`, made from the generation tiles are
    /// made from now, in place of whatever its key held, measuring how far
    /// it lies from its resident parent and its resident children from it.
    /// The only way a tile becomes resident.
    pub(super) fn arrive(&mut self, tile: Arc<Tile>, phase: Phase) {
        let made = self.generation;
        self.put(tile, phase, made);
    }

    /// Gives resident tile `key`, waiting on its objects standing far away,
    /// `tile`: itself with them. It stays of the generation it was made
    /// from: objects placed on ground edited since regenerate with it.
    pub(super) fn complete(&mut self, tile: Arc<Tile>) {
        let key = tile.key();
        let Some(made) = self.tiles.get(&key).map(|resident| resident.made) else {
            return;
        };
        self.put(tile, Phase::Whole, made);
    }

    fn put(&mut self, tile: Arc<Tile>, phase: Phase, made: Generation) {
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
        self.arrivals += 1;
        self.tiles.insert(
            key,
            Resident {
                tile,
                phase,
                made,
                arrival: self.arrivals,
            },
        );
        self.revision += 1;
    }

    /// Takes tile `key` out of residency.
    pub(super) fn leave(&mut self, key: &TileKey) -> Option<Left> {
        let resident = self.tiles.remove(key)?;
        self.revision += 1;
        Some(Left {
            current: resident.phase == Phase::Whole && resident.made == self.generation,
            tile: resident.tile,
        })
    }

    /// Makes `generation` the one tiles are made from, which the resident
    /// tiles `under` takes are not: they regenerate in their turn and draw
    /// until they do; every other resident tile of the generation before
    /// is of the new one too. Tiles kept in memory under it are no longer
    /// the ground.
    pub(super) fn invalidate(&mut self, generation: Generation, under: impl Fn(&TileKey) -> bool) {
        let previous = std::mem::replace(&mut self.generation, generation);
        for (key, resident) in &mut self.tiles {
            if resident.made == previous && !under(key) {
                resident.made = generation;
            }
        }
        let gone: Vec<TileKey> = self
            .cache
            .keys()
            .filter(|key| under(key))
            .copied()
            .collect();
        for key in gone {
            self.forget(key);
        }
        self.revision += 1;
    }

    // Tiles kept in memory.

    /// Keeps `tile` in memory as of frame `frame`.
    pub(super) fn keep_in_memory(&mut self, tile: Arc<Tile>, frame: u64) {
        let key = tile.key();
        let bytes = memory_bytes(&tile);
        self.forget(key);
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

    /// Drops `key`'s tile from memory, if kept.
    pub(super) fn forget(&mut self, key: TileKey) {
        if let Some(cached) = self.cache.remove(&key) {
            self.cached_bytes -= cached.bytes;
        }
    }

    /// Whether `key`'s tile is kept in memory.
    pub(super) fn in_memory(&self, key: &TileKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Marks `key`'s tile kept in memory as wanted at frame `frame`.
    pub(super) fn wanted_in_memory(&mut self, key: &TileKey, frame: u64) {
        if let Some(cached) = self.cache.get_mut(key) {
            cached.left = frame;
        }
    }

    /// The tiles kept in memory, by the frame each left.
    pub(super) fn kept(&self) -> impl Iterator<Item = (u64, TileKey)> + '_ {
        self.cache.iter().map(|(key, kept)| (kept.left, *key))
    }

    pub(super) fn kept_tiles(&self) -> usize {
        self.cache.len()
    }

    pub(super) fn cached_bytes(&self) -> u64 {
        self.cached_bytes
    }

    pub(super) fn tile_bytes(&self) -> u64 {
        self.tile_bytes
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
            if let Some(cached) = self.cache.remove(&key) {
                self.cached_bytes -= cached.bytes;
                work.upload(2 * TILE_ARRAY_BYTES);
                self.arrive(cached.tile, Phase::Whole);
                restored += 1;
            }
        }
        restored
    }

    /// Checks the residency invariants that need the store alone: R1, each
    /// resident and kept tile is the tile its key names; R2, a tile in
    /// [`Phase::Shape`] holds neither materials nor objects; R5, the bytes
    /// kept in memory are their tiles'. Panics naming the first broken.
    #[cfg(debug_assertions)]
    pub(super) fn check(&self) {
        let levels = self.pyramid.levels();
        for (key, resident) in &self.tiles {
            assert!(
                resident.tile.key() == *key && u32::from(key.level) < levels,
                "R1: resident tile {key:?} holds tile {:?} ({levels} levels)",
                resident.tile.key()
            );
            assert!(
                resident.phase != Phase::Shape
                    || (resident.tile.control().is_none() && resident.tile.placing().is_none()),
                "R2: tile {key:?}, its shape alone, holds materials or objects"
            );
        }
        let mut bytes = 0;
        for (key, cached) in &self.cache {
            assert!(
                cached.tile.key() == *key,
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
