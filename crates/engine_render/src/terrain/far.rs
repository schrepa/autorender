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
//! other. A terrain's hash is made again only when what it reads changed.

use super::atlas::TileAtlas;
use super::tables::TerrainTables;
use super::*;
use crate::terrain_vt::FarSource;

/// What a terrain's far pages were last hashed from: its tiles' revision,
/// its macro colors' identity, its palette entries and the level read; and
/// the hash.
struct Hashed {
    revision: u64,
    colors: u128,
    palette: Vec<PaletteEntry>,
    level: u32,
    identity: u128,
}

/// Each terrain's far pages' identity as last hashed, by id.
#[derive(Default)]
pub(super) struct FarInputs {
    hashed: HashMap<u64, Hashed>,
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
        self.hashed
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
                let settled = pyramid
                    .level_tiles(level)
                    .all(|key| atlas.settled((terrain.id, key)));
                let map = terrain.macro_map.as_ref()?;
                if !settled || colors.layer(index) == 0 {
                    return None;
                }
                let palette = tables.palette_of(index, terrain.surface.len());
                if let Some(hashed) = self.hashed.get(&terrain.id)
                    && hashed.revision == terrain.tiles_revision
                    && hashed.colors == map.identity
                    && hashed.level == level
                    && hashed.palette.as_slice() == palette
                {
                    return Some(FarSource {
                        level,
                        identity: hashed.identity,
                    });
                }
                let identity = identity(terrain, level, palette, map.identity);
                self.hashed.insert(
                    terrain.id,
                    Hashed {
                        revision: terrain.tiles_revision,
                        colors: map.identity,
                        palette: palette.to_vec(),
                        level,
                        identity,
                    },
                );
                Some(FarSource { level, identity })
            })
            .collect()
    }
}

/// The hash of what `terrain`'s far pages read at `level`, with its
/// `palette` entries and its macro colors' identity `colors`.
fn identity(terrain: &RenderTerrain, level: u32, palette: &[PaletteEntry], colors: u128) -> u128 {
    let mut bytes: Vec<u8> = Vec::new();
    for value in terrain.origin.to_array() {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    for tiles in terrain.pyramid.base_tiles() {
        bytes.extend_from_slice(&tiles.to_le_bytes());
    }
    bytes.extend_from_slice(&terrain.pyramid.spacing(0).to_bits().to_le_bytes());
    bytes.extend_from_slice(&level.to_le_bytes());
    // The level's tiles, in key order.
    for tile in terrain
        .tiles
        .iter()
        .filter(|tile| u32::from(tile.key().level) == level)
    {
        let key = tile.key();
        bytes.push(key.level);
        bytes.extend_from_slice(&key.x.to_le_bytes());
        bytes.extend_from_slice(&key.z.to_le_bytes());
        let heights = engine_core::content_hash(bytemuck::cast_slice(tile.heights()));
        bytes.extend_from_slice(&heights.to_le_bytes());
        if let Some(control) = tile.control() {
            let control = engine_core::content_hash(bytemuck::cast_slice(control));
            bytes.extend_from_slice(&control.to_le_bytes());
        }
    }
    if let Some(surface) = &terrain.gpu_surface {
        let surface = engine_core::content_hash(bytemuck::cast_slice(&surface.words));
        bytes.extend_from_slice(&surface.to_le_bytes());
    }
    bytes.extend_from_slice(bytemuck::cast_slice(palette));
    bytes.extend_from_slice(&colors.to_le_bytes());
    engine_core::content_hash(&bytes)
}
