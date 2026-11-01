use std::ops::Range;

use bevy_ecs::prelude::Entity;
use engine_core::AssetLibrary;
use engine_core::collections::HashMap;
use engine_places::ground::{
    PALETTE_ENTRIES, Pyramid, SurfaceMaps, SurfaceMaterial, SurfaceProjection, TerrainEntry,
    TerrainTiles, TileKey,
};
use engine_render::Prepare;

use super::part::{Handles, TerrainGpu};
use super::{HOLE_WORDS, NO_HOLES, widen};
use crate::gpu::{NOT_RESIDENT, PaletteEntry};
use crate::state::{Atlas, TileId};
use crate::terrain_maps::TerrainMaps;

#[derive(Default)]
pub(crate) struct TerrainTables {
    pub(super) hole_offsets: HashMap<TileId, u32>,
    hole_layers: HashMap<TileId, u32>,
    holes_bytes: u64,
    fades_stale: bool,
    pub(super) fade_words: u32,
    pub(super) directory_written: Vec<u32>,
    directory_layout: Vec<(Entity, Pyramid)>,
    directory_spare: Vec<u32>,
    directory_bytes: u64,
    pub(super) directory_bases: Vec<u32>,
    pub(super) palette_bases: Vec<u32>,
    palette_written: Vec<PaletteEntry>,
    palette_spare: Vec<PaletteEntry>,
    pub(crate) dropped_materials: u32,
}

impl TerrainTables {
    pub(super) fn fade_words(atlas: &Atlas) -> u32 {
        atlas.capacity.div_ceil(2).max(1)
    }

    pub(super) fn write_holes(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        atlas: &Atlas,
    ) {
        let max_layers = atlas.capacity;
        let fade_words = Self::fade_words(atlas);
        let edited = atlas
            .uploads
            .iter()
            .any(|(id, _, _)| atlas.holed.contains(id));
        let unchanged = !edited
            && fade_words == self.fade_words
            && atlas.holed.len() == self.hole_offsets.len()
            && atlas.holed.iter().all(|key| {
                self.hole_offsets.contains_key(key)
                    && atlas.slots.get(key) == self.hole_layers.get(key)
            });
        if unchanged {
            return;
        }
        self.fade_words = fade_words;
        self.hole_offsets.clear();
        self.hole_layers.clear();
        let mut words: Vec<u32> = vec![NO_HOLES; max_layers as usize];
        for &key in &atlas.holed {
            let Some(bits) = atlas.held.get(&key).and_then(|tile| tile.holes.as_ref()) else {
                continue;
            };
            let offset = fade_words + words.len() as u32;
            self.hole_offsets.insert(key, offset);
            if let Some(&layer) = atlas.slots.get(&key) {
                self.hole_layers.insert(key, layer);
                if let Some(word) = words.get_mut(layer as usize) {
                    *word = offset;
                }
            }
            words.extend(
                bits.iter()
                    .flat_map(|word| [*word as u32, (word >> 32) as u32])
                    .take(HOLE_WORDS),
            );
        }
        let start = u64::from(fade_words) * 4;
        let bytes = start + (words.len() * 4) as u64;
        if bytes > self.holes_bytes {
            self.holes_bytes = bytes;
            self.fades_stale = true;
        }
        cx.allocate_buffer(handles.holes, self.holes_bytes.max(16));
        if !words.is_empty() {
            cx.write(handles.holes, start, &words);
        }
    }

    pub(super) fn write_fades(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (words, dirty): (&[u32], Range<usize>),
    ) {
        if words.is_empty() {
            return;
        }
        let run = if self.fades_stale {
            0..words.len()
        } else {
            dirty
        };
        self.fades_stale = false;
        if let Some(changed) = words.get(run.clone()).filter(|changed| !changed.is_empty()) {
            cx.write(handles.holes, (run.start * 4) as u64, changed);
        }
    }

    pub(super) fn write_directory(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
        atlas: &Atlas,
    ) {
        let same = self.directory_layout.len() == tiles.terrains.len()
            && self.directory_layout.iter().zip(tiles.terrains.iter()).all(
                |(&(id, pyramid), terrain)| id == terrain.terrain && pyramid == terrain.pyramid,
            );
        if !same {
            self.lay_out_directory(cx, handles, tiles, atlas);
            return;
        }
        let mut dirty = 0..0;
        for (terrain, &base) in tiles.terrains.iter().zip(&self.directory_bases) {
            for level in 0..terrain.pyramid.levels() {
                let at = base as usize + 4 * level as usize + 3;
                let bits = reach(terrain, level).to_bits();
                set_word(&mut self.directory_written, at, bits, &mut dirty);
            }
        }
        for &(id, key) in &atlas.changed {
            let Some(index) = tiles
                .terrains
                .iter()
                .position(|terrain| terrain.terrain == id)
            else {
                continue;
            };
            let (pyramid, level) = (&tiles.terrains[index].pyramid, u32::from(key.level));
            if level >= pyramid.levels() {
                continue;
            }
            let [tiles_x, tiles_z] = pyramid.tiles(level);
            if u32::from(key.x) >= tiles_x || u32::from(key.z) >= tiles_z {
                continue;
            }
            let header = self
                .directory_bases
                .get(index)
                .map(|&base| base as usize + 4 * level as usize);
            let Some(&first) = header.and_then(|header| self.directory_written.get(header)) else {
                continue;
            };
            let at = first as usize + (u32::from(key.z) * tiles_x + u32::from(key.x)) as usize;
            let layer = atlas.layer_of(id, key).unwrap_or(NOT_RESIDENT);
            set_word(&mut self.directory_written, at, layer, &mut dirty);
        }
        let bytes = (self.directory_written.len() * 4) as u64;
        let grown = bytes > self.directory_bytes;
        if grown {
            self.directory_bytes = bytes;
        }
        cx.allocate_buffer(handles.directory, self.directory_bytes.max(16));
        let run = if grown {
            0..self.directory_written.len()
        } else {
            dirty
        };
        if self.directory_written.is_empty() || run.is_empty() {
            return;
        }
        cx.write(
            handles.directory,
            (run.start * 4) as u64,
            &self.directory_written[run],
        );
    }

    fn lay_out_directory(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
        atlas: &Atlas,
    ) {
        let mut words = std::mem::take(&mut self.directory_spare);
        words.clear();
        self.directory_bases.clear();
        for terrain in tiles.terrains.iter() {
            let pyramid = &terrain.pyramid;
            let base = words.len();
            self.directory_bases.push(base as u32);
            let levels = pyramid.levels() as usize;
            words.resize(base + 4 * levels, 0);
            for level in 0..pyramid.levels() {
                let [tiles_x, tiles_z] = pyramid.tiles(level);
                let header = base + 4 * level as usize;
                words[header] = words.len() as u32;
                words[header + 1] = tiles_x;
                words[header + 2] = tiles_z;
                words[header + 3] = reach(terrain, level).to_bits();
                for z in 0..tiles_z {
                    for x in 0..tiles_x {
                        let key = TileKey::new(level as u8, x as u16, z as u16);
                        words.push(atlas.layer_of(terrain.terrain, key).unwrap_or(NOT_RESIDENT));
                    }
                }
            }
        }
        self.directory_layout.clear();
        self.directory_layout.extend(
            tiles
                .terrains
                .iter()
                .map(|terrain| (terrain.terrain, terrain.pyramid)),
        );
        let bytes = (words.len() * 4) as u64;
        let grown = bytes > self.directory_bytes;
        if grown {
            self.directory_bytes = bytes;
        }
        cx.allocate_buffer(handles.directory, self.directory_bytes.max(16));
        if words.is_empty() || (words == self.directory_written && !grown) {
            self.directory_spare = words;
            return;
        }
        cx.write(handles.directory, 0, &words);
        self.directory_spare = std::mem::replace(&mut self.directory_written, words);
    }

    pub(super) fn write_palette(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (tiles, maps, assets): (&TerrainTiles, &TerrainMaps, &AssetLibrary),
    ) {
        let mut entries = std::mem::take(&mut self.palette_spare);
        entries.clear();
        self.palette_bases.clear();
        self.dropped_materials = maps.dropped;
        for terrain in tiles.terrains.iter() {
            let room = PALETTE_ENTRIES as usize - entries.len();
            let kept = terrain.materials.len().min(room);
            self.dropped_materials += (terrain.materials.len() - kept) as u32;
            self.palette_bases.push(if kept == terrain.materials.len() {
                entries.len() as u32
            } else {
                0
            });
            let terrain_maps = terrain.maps.iter().chain(std::iter::repeat(&None));
            entries.extend(
                terrain.materials[..kept]
                    .iter()
                    .zip(terrain_maps)
                    .map(|(look, terrain_maps)| palette_entry(look, terrain_maps, (maps, assets))),
            );
        }
        cx.allocate_buffer(handles.palette, crate::gpu::PALETTE_BYTES as u64);
        if entries.is_empty() || entries == self.palette_written {
            self.palette_spare = entries;
            return;
        }
        cx.write(handles.palette, 0, &entries);
        self.palette_spare = std::mem::replace(&mut self.palette_written, entries);
    }

    pub(super) fn palette_of(&self, index: usize, materials: usize) -> &[PaletteEntry] {
        let base = self.palette_bases.get(index).copied().unwrap_or(0) as usize;
        self.palette_written
            .get(base..base + materials)
            .unwrap_or_default()
    }

    pub(super) fn holes_of(&self, id: TileId) -> u32 {
        self.hole_offsets.get(&id).copied().unwrap_or(NO_HOLES)
    }
}

fn reach(terrain: &TerrainEntry, level: u32) -> f32 {
    match terrain.reach.get(level as usize) {
        Some(&reach) => reach,
        None if level + 1 == terrain.pyramid.levels() => f32::MAX,
        None => 0.0,
    }
}

fn set_word(words: &mut [u32], at: usize, value: u32, dirty: &mut Range<usize>) {
    if let Some(word) = words.get_mut(at)
        && *word != value
    {
        *word = value;
        *dirty = widen(dirty.clone(), at);
    }
}

fn palette_entry(
    look: &SurfaceMaterial,
    terrain_maps: &Option<SurfaceMaps>,
    (maps, assets): (&TerrainMaps, &AssetLibrary),
) -> PaletteEntry {
    let projection = match look.projection {
        SurfaceProjection::Top => 0.0,
        SurfaceProjection::Biplanar => 1.0,
    };
    let material = terrain_maps.as_ref().and_then(|terrain_maps| {
        Some((terrain_maps, TerrainMaps::material(terrain_maps, assets)?))
    });
    let (color, roughness, layer, scale, occlusion, metalness) = match material {
        Some((terrain_maps, material)) => (
            material.color,
            material.roughness,
            maps.layer(terrain_maps, assets),
            material.normal_scale,
            if material.packed {
                material.occlusion_strength
            } else {
                0.0
            },
            material.metalness,
        ),
        None => (look.color, look.roughness, None, 1.0, 0.0, 0.0),
    };
    let means = layer.map(|layer| maps.means(layer));
    let far = means.map_or([color[0], color[1], color[2], roughness], |means| {
        [
            color[0] * means.color[0],
            color[1] * means.color[1],
            color[2] * means.color[2],
            roughness * means.roughness,
        ]
    });
    let far_occlusion = means.map_or(1.0, |means| 1.0 + occlusion * (means.occlusion - 1.0));
    let slopes = means.map_or(0.0, |means| means.slopes * scale * scale);
    let heights = means.map_or([0.0; 4], |means| [means.height, 0.0, 0.0, 0.0]);
    PaletteEntry {
        color: [color[0], color[1], color[2], roughness],
        params: [
            look.tiling,
            look.height_contrast,
            projection,
            layer.map_or(0.0, |layer| (layer + 1) as f32),
        ],
        maps: [scale, occlusion, metalness, far_occlusion],
        far,
        relief: [look.depth, 0.0, slopes, 0.0],
        heights,
    }
}
