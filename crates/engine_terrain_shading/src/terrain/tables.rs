use bevy_ecs::prelude::Entity;
use engine_core::AssetLibrary;
use engine_heightfield::tile::shaded_elsewhere;
use engine_places::ground::{
    PALETTE_ENTRIES, Pyramid, SurfaceMaps, SurfaceMaterial, SurfaceProjection, TerrainEntry,
    TerrainTiles, TileFace, TileKey,
};
use engine_render::Prepare;

use super::part::{Handles, TerrainGpu};
use super::{DIRECTORY_AT, HOLE_WORDS, NO_HOLES, VIEWS};
use crate::gpu::{
    NOT_RESIDENT, PaletteEntry, TABLE_FADES, TABLE_NOW, TABLE_TERRAINS, TABLE_VIEWS, TERRAIN_WORDS,
    TerrainUniform, VIEW_WORDS, ViewUniform,
};
use crate::state::{Atlas, TileId, Upload, listed_tile};
use crate::terrain_maps::TerrainMaps;

const FADE_HELD: f32 = 1.0e30;
const FADE_DONE: f32 = -1.0e30;

#[derive(Default)]
pub(crate) struct TerrainTables {
    directory_layout: Vec<(Entity, Pyramid)>,
    directory_bases: Vec<u32>,
    directory_room: u32,
    terrains_room: u32,
    views_room: u32,
    layers_room: u32,
    holes_room: u32,
    total: u32,
    terrains_at: u32,
    views_at: u32,
    fades_at: u32,
    pub(super) hole_table: u32,
    pub(super) palette_bases: Vec<u32>,
    palette: Vec<PaletteEntry>,
    pub(crate) dropped_materials: u32,
}

impl TerrainTables {
    pub(super) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (tiles, atlas, moved): (&TerrainTiles, &Atlas, bool),
        views: usize,
    ) {
        let laid_out = self.directory_layout.len() == tiles.terrains.len()
            && self.directory_layout.iter().zip(tiles.terrains.iter()).all(
                |(&(id, pyramid), terrain)| id == terrain.terrain && pyramid == terrain.pyramid,
            );
        if !laid_out {
            self.lay_out_directory(tiles);
        }
        self.terrains_room = self
            .terrains_room
            .max(tiles.terrains.len().max(1) as u32 * TERRAIN_WORDS);
        self.views_room = self
            .views_room
            .max(views.max(VIEWS as usize) as u32 * VIEW_WORDS);
        self.layers_room = self.layers_room.max(atlas.capacity.max(1));
        for &(id, layer, upload) in &atlas.uploads {
            if upload == Upload::Whole
                && tile_of(tiles, id).is_some_and(|tile| tile.holes.is_some())
            {
                let room = (layer + 1).next_power_of_two().min(self.layers_room);
                self.holes_room = self.holes_room.max(room);
            }
        }
        self.terrains_at = DIRECTORY_AT + self.directory_room;
        self.views_at = self.terrains_at + self.terrains_room;
        self.fades_at = self.views_at + self.views_room;
        self.hole_table = self.fades_at + 2 * self.layers_room;
        let total = self.hole_table + self.layers_room + self.holes_room * HOLE_WORDS as u32;
        let fresh = total != self.total;
        self.total = total;
        cx.allocate_buffer(handles.tables, u64::from(total) * 4);
        if fresh || !laid_out {
            self.write_directory(cx, handles, tiles, atlas);
        } else {
            if moved {
                self.write_reaches(cx, handles, tiles);
            }
            self.write_changed_layers(cx, handles, tiles, atlas);
        }
        if fresh {
            self.write_all_fades(cx, handles, tiles, atlas);
            self.write_all_holes(cx, handles, tiles, atlas);
        } else {
            self.write_moved_fades(cx, handles, tiles, atlas);
            self.write_moved_holes(cx, handles, tiles, atlas);
        }
    }

    fn lay_out_directory(&mut self, tiles: &TerrainTiles) {
        self.directory_layout.clear();
        self.directory_layout.extend(
            tiles
                .terrains
                .iter()
                .map(|terrain| (terrain.terrain, terrain.pyramid)),
        );
        self.directory_bases.clear();
        let mut words = 0;
        for terrain in tiles.terrains.iter() {
            self.directory_bases.push(words);
            let pyramid = &terrain.pyramid;
            words += 4 * pyramid.levels() + tiles_below(pyramid, pyramid.levels());
        }
        self.directory_room = self.directory_room.max(words);
    }

    fn write_directory(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
        atlas: &Atlas,
    ) {
        let mut words: Vec<u32> = Vec::new();
        for terrain in tiles.terrains.iter() {
            let pyramid = &terrain.pyramid;
            let base = words.len();
            words.resize(base + 4 * pyramid.levels() as usize, 0);
            for level in 0..pyramid.levels() {
                let [tiles_x, tiles_z] = pyramid.tiles(level);
                let header = base + 4 * level as usize;
                words[header] = DIRECTORY_AT + words.len() as u32;
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
        if !words.is_empty() {
            cx.write(handles.tables, u64::from(DIRECTORY_AT) * 4, &words);
        }
    }

    fn write_reaches(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
    ) {
        for (terrain, &base) in tiles.terrains.iter().zip(&self.directory_bases) {
            for level in 0..terrain.pyramid.levels() {
                let at = DIRECTORY_AT + base + 4 * level + 3;
                let bits = reach(terrain, level).to_bits();
                cx.write(handles.tables, u64::from(at) * 4, &[bits]);
            }
        }
    }

    fn write_changed_layers(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
        atlas: &Atlas,
    ) {
        for &(id, key) in &atlas.changed {
            let Some(index) = tiles
                .terrains
                .iter()
                .position(|terrain| terrain.terrain == id)
            else {
                continue;
            };
            let pyramid = &tiles.terrains[index].pyramid;
            let level = u32::from(key.level);
            if level >= pyramid.levels() {
                continue;
            }
            let [tiles_x, tiles_z] = pyramid.tiles(level);
            if u32::from(key.x) >= tiles_x || u32::from(key.z) >= tiles_z {
                continue;
            }
            let first =
                self.directory_base(index) + 4 * pyramid.levels() + tiles_below(pyramid, level);
            let at = first + u32::from(key.z) * tiles_x + u32::from(key.x);
            let layer = atlas.layer_of(id, key).unwrap_or(NOT_RESIDENT);
            cx.write(handles.tables, u64::from(at) * 4, &[layer]);
        }
    }

    fn write_all_fades(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
        atlas: &Atlas,
    ) {
        let mut fades = vec![[FADE_DONE; 2]; self.layers_room as usize];
        for (&id, &layer) in &atlas.slots {
            if let Some(fade) = fades.get_mut(layer as usize) {
                *fade = fades_of(tiles, atlas, id);
            }
        }
        cx.write(handles.tables, u64::from(self.fades_at) * 4, &fades);
    }

    fn write_moved_fades(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
        atlas: &Atlas,
    ) {
        for &layer in &atlas.vacated {
            self.write_fade(cx, handles, layer, [FADE_DONE; 2]);
        }
        for &id in &atlas.changed {
            if let Some(layer) = atlas.layer_of(id.0, id.1) {
                self.write_fade(cx, handles, layer, fades_of(tiles, atlas, id));
            }
        }
    }

    fn write_fade(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        layer: u32,
        fade: [f32; 2],
    ) {
        let at = self.fades_at + 2 * layer;
        cx.write(handles.tables, u64::from(at) * 4, &fade);
    }

    fn write_all_holes(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
        atlas: &Atlas,
    ) {
        let table = vec![NO_HOLES; self.layers_room as usize];
        cx.write(handles.tables, u64::from(self.hole_table) * 4, &table);
        for (&id, &layer) in &atlas.slots {
            if let Some(tile) = tile_of(tiles, id) {
                self.write_tile_holes(cx, handles, layer, tile);
            }
        }
    }

    fn write_moved_holes(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        tiles: &TerrainTiles,
        atlas: &Atlas,
    ) {
        for &layer in &atlas.vacated {
            let at = self.hole_table + layer;
            cx.write(handles.tables, u64::from(at) * 4, &[NO_HOLES]);
        }
        for &(id, layer, upload) in &atlas.uploads {
            if upload != Upload::Whole {
                continue;
            }
            if let Some(tile) = tile_of(tiles, id) {
                self.write_tile_holes(cx, handles, layer, tile);
            }
        }
    }

    fn write_tile_holes(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        layer: u32,
        tile: &TileFace,
    ) {
        let at = self.hole_table + layer;
        cx.write(
            handles.tables,
            u64::from(at) * 4,
            &[self.holes_of(layer, tile)],
        );
        if let Some(bits) = &tile.holes {
            let words: Vec<u32> = bits
                .iter()
                .flat_map(|word| [*word as u32, (word >> 32) as u32])
                .take(HOLE_WORDS)
                .collect();
            cx.write(
                handles.tables,
                u64::from(self.hole_bits_at(layer)) * 4,
                &words,
            );
        }
    }

    fn hole_bits_at(&self, layer: u32) -> u32 {
        self.hole_table + self.layers_room + layer * HOLE_WORDS as u32
    }

    pub(super) fn holes_of(&self, layer: u32, tile: &TileFace) -> u32 {
        if tile.holes.is_some() {
            self.hole_bits_at(layer)
        } else {
            NO_HOLES
        }
    }

    pub(super) fn write_header(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        now: f32,
    ) {
        let mut header = [0_u32; DIRECTORY_AT as usize];
        header[TABLE_TERRAINS as usize] = self.terrains_at;
        header[TABLE_VIEWS as usize] = self.views_at;
        header[TABLE_FADES as usize] = self.fades_at;
        header[TABLE_NOW as usize] = now.to_bits();
        cx.write(handles.tables, 0, &header);
    }

    pub(super) fn write_terrains(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        uniforms: &[TerrainUniform],
    ) {
        if !uniforms.is_empty() {
            cx.write(handles.tables, u64::from(self.terrains_at) * 4, uniforms);
        }
    }

    pub(super) fn write_views(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        views: &[ViewUniform],
    ) {
        cx.write(handles.tables, u64::from(self.views_at) * 4, views);
    }

    pub(super) fn directory_base(&self, index: usize) -> u32 {
        DIRECTORY_AT + self.directory_bases.get(index).copied().unwrap_or(0)
    }

    pub(super) fn write_palette(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (tiles, maps, assets): (&TerrainTiles, &TerrainMaps, &AssetLibrary),
    ) {
        self.palette.clear();
        self.palette_bases.clear();
        self.dropped_materials = maps.dropped;
        for terrain in tiles.terrains.iter() {
            let room = PALETTE_ENTRIES as usize - self.palette.len();
            let kept = terrain.materials.len().min(room);
            self.dropped_materials += (terrain.materials.len() - kept) as u32;
            self.palette_bases.push(if kept == terrain.materials.len() {
                self.palette.len() as u32
            } else {
                0
            });
            let terrain_maps = terrain.maps.iter().chain(std::iter::repeat(&None));
            self.palette.extend(
                terrain.materials[..kept]
                    .iter()
                    .zip(terrain_maps)
                    .map(|(look, terrain_maps)| palette_entry(look, terrain_maps, (maps, assets))),
            );
        }
        cx.allocate_buffer(handles.palette, crate::gpu::PALETTE_BYTES as u64);
        if !self.palette.is_empty() {
            cx.write(handles.palette, 0, &self.palette);
        }
    }

    pub(super) fn palette_of(&self, index: usize, materials: usize) -> &[PaletteEntry] {
        let base = self.palette_bases.get(index).copied().unwrap_or(0) as usize;
        self.palette.get(base..base + materials).unwrap_or_default()
    }
}

fn tile_of(tiles: &TerrainTiles, id: TileId) -> Option<&TileFace> {
    let entry = tiles.terrains.iter().find(|entry| entry.terrain == id.0)?;
    listed_tile(&entry.tiles, id.1)
}

fn tiles_below(pyramid: &Pyramid, level: u32) -> u32 {
    (0..level)
        .map(|finer| {
            let [tiles_x, tiles_z] = pyramid.tiles(finer);
            tiles_x * tiles_z
        })
        .sum()
}

fn fades_of(tiles: &TerrainTiles, atlas: &Atlas, id: TileId) -> [f32; 2] {
    let materials = atlas.surfaced.get(&id).copied().unwrap_or(FADE_HELD);
    let shaded = tiles
        .terrains
        .iter()
        .find(|entry| entry.terrain == id.0)
        .is_some_and(|entry| shaded_elsewhere(id.1.level, entry.surroundings));
    if shaded {
        return [materials, materials];
    }
    [
        atlas.arrivals.get(&id).copied().unwrap_or(FADE_DONE),
        materials,
    ]
}

fn reach(terrain: &TerrainEntry, level: u32) -> f32 {
    match terrain.reach.get(level as usize) {
        Some(&reach) => reach,
        None if level + 1 == terrain.pyramid.levels() => f32::MAX,
        None => 0.0,
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
