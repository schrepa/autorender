//! What the runtime virtual texture's far pages are baked from
//! (`terrain_vt.rs`): each terrain's finest pinned level, once every tile of
//! it stands resident as it is and the terrain's macro colors are here, and
//! the hash of everything their texels read there: the terrain's place and
//! pyramid, that level's tiles (their heights, and their control values or
//! the surface the GPU gives them from), its surface materials' palette
//! entries and its macro colors' identity. Far pages read no material map:
//! at their texels, eight meters and more, the maps of any material
//! repeating within 64 meters show only their means, which the palette
//! holds, so pages cooked on one family of compressed formats serve every
//! other.
//!
//! Each tile's hashes are made once it stands settled after it arrived or
//! changed, and the surface's once as it arrives; a terrain's hash is made
//! again from them only when what it reads changed. Which tiles of the
//! level changed or began fading in each frame is what the atlas tells, so
//! a frame looks at the level's tiles only when one of them did.

use super::atlas::TileAtlas;
use super::tables::TerrainTables;
use super::*;
use crate::terrain_vt::FarSource;
use engine_core::terrain_data::Pyramid;

/// A tile's hashes as far pages read it: its heights', and its control
/// values' where it holds them.
type TileHashes = (u128, Option<u128>);

/// What a terrain's far pages read, as last seen: the level and the
/// pyramid it lies in, whether every tile of the level stood settled
/// (`None` until looked at again), each tile's hashes (by column, then row;
/// `None` until made after it changed), the place, the surface the GPU
/// gives control values from with its hash, the palette entries and the
/// macro colors' identity; and the hash of them all, `None` once one of
/// them changed.
struct FarTerrain {
    level: u32,
    pyramid: Pyramid,
    settled: Option<bool>,
    tiles: Vec<Option<TileHashes>>,
    origin: [u32; 3],
    surface: Option<(Arc<SurfaceData>, u128)>,
    palette: Vec<PaletteEntry>,
    colors: u128,
    identity: Option<u128>,
}

impl FarTerrain {
    /// What the far pages of a terrain over `pyramid` read at `level`,
    /// before anything of it was seen.
    fn new(pyramid: Pyramid, level: u32) -> Self {
        let [tiles_x, tiles_z] = pyramid.tiles(level);
        Self {
            level,
            pyramid,
            settled: None,
            tiles: vec![None; (tiles_x * tiles_z) as usize],
            origin: [0; 3],
            surface: None,
            palette: Vec::new(),
            colors: 0,
            identity: None,
        }
    }

    /// Notes what of the level changed this frame in `atlas` for terrain
    /// `id`: tiles that arrived, left, were replaced or finished fading in
    /// are hashed again once settled, and with those that began fading in
    /// have the level's settledness looked at again.
    fn follow(&mut self, id: u64, atlas: &TileAtlas) {
        let level = self.level;
        let [_, tiles_z] = self.pyramid.tiles(level);
        let ours = |&(owner, key): &(u64, TileKey)| owner == id && u32::from(key.level) == level;
        for &(_, key) in atlas.changed.iter().filter(|change| ours(*change)) {
            let at = usize::from(key.x) * tiles_z as usize + usize::from(key.z);
            if let Some(hashes) = self.tiles.get_mut(at) {
                *hashes = None;
            }
            (self.settled, self.identity) = (None, None);
        }
        if atlas.entered.iter().any(ours) {
            self.settled = None;
        }
    }

    /// The hash of what `terrain`'s far pages read, the hashes of the
    /// level's tiles that changed made from `atlas`' tiles; `None` where a
    /// tile of the level is not resident.
    fn hash(&mut self, terrain: &RenderTerrain, atlas: &TileAtlas) -> Option<u128> {
        let level = self.level;
        let [_, tiles_z] = self.pyramid.tiles(level);
        let mut bytes: Vec<u8> = Vec::new();
        for bits in self.origin {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        for tiles in self.pyramid.base_tiles() {
            bytes.extend_from_slice(&tiles.to_le_bytes());
        }
        bytes.extend_from_slice(&self.pyramid.spacing(0).to_bits().to_le_bytes());
        bytes.extend_from_slice(&level.to_le_bytes());
        // The level's tiles, in key order.
        for (at, hashes) in self.tiles.iter_mut().enumerate() {
            let key = TileKey::new(
                level as u8,
                (at / tiles_z as usize) as u16,
                (at % tiles_z as usize) as u16,
            );
            let (heights, control) = match *hashes {
                Some(made) => made,
                None => {
                    let tile = atlas.held.get(&(terrain.id, key))?;
                    let made = (
                        engine_core::content_hash(bytemuck::cast_slice(tile.heights())),
                        tile.control().map(|control| {
                            engine_core::content_hash(bytemuck::cast_slice(control))
                        }),
                    );
                    *hashes = Some(made);
                    made
                }
            };
            bytes.push(key.level);
            bytes.extend_from_slice(&key.x.to_le_bytes());
            bytes.extend_from_slice(&key.z.to_le_bytes());
            bytes.extend_from_slice(&heights.to_le_bytes());
            if let Some(control) = control {
                bytes.extend_from_slice(&control.to_le_bytes());
            }
        }
        if let Some((_, surface)) = &self.surface {
            bytes.extend_from_slice(&surface.to_le_bytes());
        }
        bytes.extend_from_slice(bytemuck::cast_slice(&self.palette));
        bytes.extend_from_slice(&self.colors.to_le_bytes());
        Some(engine_core::content_hash(&bytes))
    }
}

/// Each terrain's far pages' inputs as last seen, by id.
#[derive(Default)]
pub(super) struct FarInputs {
    terrains: HashMap<u64, FarTerrain>,
}

impl FarInputs {
    /// What each of `terrains`' far pages are baked from, by index, at the
    /// finest of `settings`' pinned levels: `None` while a tile of that
    /// level is missing or shows a coarser level's ground (`atlas`), or its
    /// macro colors (`colors`) have yet to arrive. Its palette entries are
    /// as `tables` wrote them.
    pub(super) fn sources(
        &mut self,
        terrains: &RenderTerrains,
        settings: &engine_core::TerrainSettings,
        (atlas, tables, colors): (&TileAtlas, &TerrainTables, &MacroMaps),
    ) -> Vec<Option<FarSource>> {
        self.terrains
            .retain(|id, _| terrains.terrains.iter().any(|terrain| terrain.id == *id));
        terrains
            .terrains
            .iter()
            .enumerate()
            .map(|(index, terrain)| {
                let pyramid = terrain.pyramid;
                let level = pyramid
                    .levels()
                    .saturating_sub(settings.pinned_levels.max(1));
                let far = (self.terrains.entry(terrain.id))
                    .or_insert_with(|| FarTerrain::new(pyramid, level));
                if far.level != level || far.pyramid != pyramid {
                    *far = FarTerrain::new(pyramid, level);
                }
                far.follow(terrain.id, atlas);
                let settled = *far.settled.get_or_insert_with(|| {
                    pyramid
                        .level_tiles(level)
                        .all(|key| atlas.settled((terrain.id, key)))
                });
                let map = terrain.macro_map.as_ref()?;
                if !settled || colors.layer(index) == 0 {
                    return None;
                }
                // What else the pages read, each against what was seen.
                let surface = terrain.gpu_surface.as_ref();
                let seen = far.surface.as_ref().map(|(held, _)| Arc::as_ptr(held));
                if surface.map(Arc::as_ptr) != seen {
                    far.surface = surface.map(|surface| {
                        let words = bytemuck::cast_slice(&surface.words);
                        (Arc::clone(surface), engine_core::content_hash(words))
                    });
                    far.identity = None;
                }
                let palette = tables.palette_of(index, terrain.surface.len());
                if far.palette.as_slice() != palette {
                    far.palette.clear();
                    far.palette.extend_from_slice(palette);
                    far.identity = None;
                }
                let origin = terrain.origin.to_array().map(f32::to_bits);
                if (far.origin, far.colors) != (origin, map.identity) {
                    (far.origin, far.colors) = (origin, map.identity);
                    far.identity = None;
                }
                let identity = match far.identity {
                    Some(identity) => identity,
                    None => {
                        let identity = far.hash(terrain, atlas)?;
                        far.identity = Some(identity);
                        identity
                    }
                };
                Some(FarSource { level, identity })
            })
            .collect()
    }
}
