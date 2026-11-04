use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use engine_core::Stamp;
use engine_core::collections::HashMap;
use engine_heightfield::ErrorPyramid;
use engine_heightfield::tile::shaded_elsewhere;
use engine_places::ground::{Pyramid, TerrainEntry, TerrainTiles, TileFace, TileKey};
use engine_places::water::Waterline;
use engine_render::{Prepare, Texture, wgpu};
use glam::Vec2;

use super::part::{Handles, TerrainGpu};
use super::{ARRIVAL_SECONDS, LAYER_SIZE, widen};
use crate::gpu::{CONTROL_GROUP, ControlJob, NORMALS_GROUP, NormalJob, SURFACE_ERROR_MIPS};
use crate::shading::TileShading;
use crate::state::{Atlas, Bakes, TerrainBakes, TileId, Upload, same_shading};

const HEIGHT_MIPS: u32 = 1 + SURFACE_ERROR_MIPS;
const ERROR_CELLS: u32 = engine_heightfield::TILE_QUADS / 2;
const FEWEST_LAYERS: u32 = 16;

const _: () = assert!(
    LAYER_SIZE >= 2 * ERROR_CELLS && LAYER_SIZE >> SURFACE_ERROR_MIPS >= 1,
    "height_mips_hold_peak_mips"
);

struct Uploaded {
    tile: TileFace,
    shading: Option<Arc<TileShading>>,
    water: Option<Arc<[u16]>>,
}

#[derive(Default)]
pub(crate) struct TileAtlas {
    layers: u32,
    uploaded: HashMap<TileId, Uploaded>,
    normal_jobs: Vec<NormalJob>,
    control_jobs: Vec<ControlJob>,
    pub(crate) uploads: u32,
    pub(crate) controls: u32,
    fades: Vec<u32>,
    fades_dirty: Range<usize>,
    faded_with: Option<f32>,
    moving: BTreeSet<TileId>,
}

struct Placing<'a> {
    id: TileId,
    layer: u32,
    index: usize,
    entry: &'a TerrainEntry,
    tile: &'a TileFace,
}

impl TileAtlas {
    pub(super) fn normal_jobs(&self) -> usize {
        self.normal_jobs.len()
    }

    pub(super) fn control_jobs(&self) -> usize {
        self.control_jobs.len()
    }

    pub(super) fn normal_groups(&self) -> [u32; 3] {
        let side = LAYER_SIZE.div_ceil(NORMALS_GROUP);
        [side, side, self.normal_jobs.len() as u32]
    }

    pub(super) fn control_groups(&self) -> [u32; 3] {
        let side = LAYER_SIZE.div_ceil(CONTROL_GROUP);
        [side, side, self.control_jobs.len() as u32]
    }

    pub(super) fn rest(&mut self) {
        *self = Self::default();
    }

    pub(super) fn upload(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (tiles, atlas, bakes): (&TerrainTiles, &Atlas, &Bakes),
        (line, watered): (&Waterline, bool),
    ) {
        self.uploads = 0;
        self.controls = 0;
        self.normal_jobs.clear();
        self.control_jobs.clear();
        let capacity = atlas.capacity.max(1);
        let needed = atlas.next_layer.min(capacity);
        let whole = needed > self.layers;
        if whole {
            self.layers = needed.next_power_of_two().max(FEWEST_LAYERS).min(capacity);
            self.uploaded.clear();
        }
        let layers = self.layers.max(1);
        let gpu = tiles.terrains.iter().any(|entry| entry.surface_program);
        let rivers = if gpu { layers } else { 1 };
        let array = |mips: u32, layers: u32| {
            (
                wgpu::Extent3d {
                    width: LAYER_SIZE,
                    height: LAYER_SIZE,
                    depth_or_array_layers: layers,
                },
                mips,
            )
        };
        let (size, mips) = array(HEIGHT_MIPS, layers);
        cx.allocate(handles.heights, size, mips);
        let (size, mips) = array(1, layers);
        cx.allocate(handles.normals, size, mips);
        cx.allocate(handles.control, size, mips);
        let (size, mips) = array(1, rivers);
        cx.allocate(handles.rivers, size, mips);
        for errors in [handles.errors, handles.water_errors] {
            cx.allocate(
                errors,
                wgpu::Extent3d {
                    width: ERROR_CELLS,
                    height: ERROR_CELLS,
                    depth_or_array_layers: layers,
                },
                SURFACE_ERROR_MIPS,
            );
        }
        self.uploaded.retain(|id, _| atlas.slots.contains_key(id));
        let uploads: Vec<(TileId, u32, Upload)> = if whole {
            let mut all: Vec<(TileId, u32, Upload)> = atlas
                .slots
                .iter()
                .map(|(&id, &layer)| (id, layer, Upload::Arrived))
                .collect();
            all.sort_unstable_by_key(|((terrain, key), _, _)| {
                (std::cmp::Reverse(key.level), *terrain, key.x, key.z)
            });
            all
        } else {
            atlas.uploads.clone()
        };
        let found = water_errors(line, watered || !uploads.is_empty());
        for (id, layer, upload) in uploads {
            let Some(index) = tiles
                .terrains
                .iter()
                .position(|entry| entry.terrain == id.0)
            else {
                continue;
            };
            let entry = &tiles.terrains[index];
            let Some(tile) = atlas.tile(id.0, id.1) else {
                continue;
            };
            let placing = Placing {
                id,
                layer,
                index,
                entry,
                tile,
            };
            let before = match upload {
                Upload::Arrived => None,
                Upload::Replaced | Upload::Errors => self.uploaded.remove(&id),
            };
            let water = errors_of(&found, id, tile.stamp);
            let wet = before
                .as_ref()
                .is_none_or(|held| !same_errors(held.water.as_ref(), water));
            if upload == Upload::Errors {
                if wet {
                    write_water(cx, handles.water_errors, layer, water);
                }
                let shading = before.and_then(|held| held.shading);
                let (tile, water) = (tile.clone(), water.cloned());
                self.uploaded.insert(
                    id,
                    Uploaded {
                        tile,
                        shading,
                        water,
                    },
                );
                continue;
            }
            let shades = shaded_elsewhere(id.1.level, entry.surroundings);
            let heights = before
                .as_ref()
                .is_none_or(|held| held.tile.heights != tile.heights);
            if heights {
                write_layer(
                    cx,
                    handles.heights,
                    layer,
                    (bytemuck::cast_slice(&tile.heights[..]), 4),
                );
                write_errors(cx, handles.errors, layer, &tile.ground_errors);
                if let Some(peaks) = atlas.peaks.get(&id) {
                    write_peaks(cx, handles.heights, layer, peaks);
                }
                if !shades {
                    self.normal_jobs.push(NormalJob {
                        layer,
                        spacing: tile.spacing.to_bits(),
                        pad: [0; 2],
                    });
                }
            }
            if wet {
                write_water(cx, handles.water_errors, layer, water);
            }
            let (control, shading) = if shades {
                self.shade(cx, handles, &placing, before.as_ref(), bakes)
            } else {
                let control =
                    self.give_control(cx, handles, &placing, before.as_ref(), (heights, bakes));
                (control, None)
            };
            self.uploads += u32::from(heights || control);
            let (tile, water) = (tile.clone(), water.cloned());
            self.uploaded.insert(
                id,
                Uploaded {
                    tile,
                    shading,
                    water,
                },
            );
        }
        if watered {
            self.rewater(cx, handles, atlas, &found);
        }
        if !self.normal_jobs.is_empty() {
            let bytes = std::mem::size_of_val(self.normal_jobs.as_slice()) as u64;
            cx.grow_buffer(handles.normal_jobs, bytes);
            cx.write(handles.normal_jobs, 0, &self.normal_jobs);
        }
        if !self.control_jobs.is_empty() {
            let bytes = std::mem::size_of_val(self.control_jobs.as_slice()) as u64;
            cx.grow_buffer(handles.control_jobs, bytes);
            cx.write(handles.control_jobs, 0, &self.control_jobs);
            self.controls = self.control_jobs.len() as u32;
        }
    }

    fn rewater(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        atlas: &Atlas,
        found: &WaterErrors<'_>,
    ) {
        let mut resident: Vec<(TileId, u32)> = atlas
            .slots
            .iter()
            .map(|(&id, &layer)| (id, layer))
            .collect();
        resident.sort_unstable();
        for (id, layer) in resident {
            let held = (atlas.tile(id.0, id.1), self.uploaded.get_mut(&id));
            let (Some(tile), Some(uploaded)) = held else {
                continue;
            };
            let water = errors_of(found, id, tile.stamp);
            if !same_errors(uploaded.water.as_ref(), water) {
                write_water(cx, handles.water_errors, layer, water);
                uploaded.water = water.cloned();
            }
        }
    }

    fn give_control(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        placing: &Placing<'_>,
        before: Option<&Uploaded>,
        (heights, bakes): (bool, &Bakes),
    ) -> bool {
        let Placing {
            id,
            layer,
            index,
            entry,
            tile,
        } = *placing;
        let gpu = tile.control.is_none() && entry.surface_program;
        let rivers = before.is_none_or(|held| {
            held.tile.river_edges != tile.river_edges || held.tile.river_beds != tile.river_beds
        });
        if gpu && (heights || rivers) {
            if let Some(texels) = &tile.rivers {
                write_layer(
                    cx,
                    handles.rivers,
                    layer,
                    (bytemuck::cast_slice(&texels[..]), 8),
                );
            }
            let spacing = tile.spacing;
            self.control_jobs.push(ControlJob {
                layer,
                rivers: u32::from(tile.rivers.is_some()),
                first: first_sample(entry, id.1, spacing),
                spacing,
                terrain: index as u32,
                pad: [0; 2],
            });
        }
        let control = !gpu
            && (tile.control.is_none()
                || before.is_none_or(|held| held.tile.control != tile.control));
        if control {
            let values = match &tile.control {
                Some(values) => Box::from(&values[..]),
                None => {
                    let listing = (&entry.tiles[..], entry.pyramid);
                    let shadings = bakes.of(id.0);
                    ancestor_texels(listing, id.1, |held| held_control(held, shadings))
                }
            };
            write_layer(
                cx,
                handles.control,
                layer,
                (bytemuck::cast_slice(&values[..]), 4),
            );
        }
        control
    }

    fn shade(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        placing: &Placing<'_>,
        before: Option<&Uploaded>,
        bakes: &Bakes,
    ) -> (bool, Option<Arc<TileShading>>) {
        let Placing {
            id, layer, entry, ..
        } = *placing;
        let shadings = bakes.of(id.0);
        let shading = shadings.and_then(|held| held.shading(&id.1)).cloned();
        if before.is_some_and(|held| same_shading(held.shading.as_ref(), shading.as_ref())) {
            return (false, shading);
        }
        let (control, gradients) = match &shading {
            Some(shading) => (shading.control_texels(), shading.gradient_texels()),
            None => {
                let listing = (&entry.tiles[..], entry.pyramid);
                (
                    ancestor_texels(listing, id.1, |held| held_control(held, shadings)),
                    ancestor_texels(listing, id.1, |held| held_gradients(held, shadings)),
                )
            }
        };
        write_layer(
            cx,
            handles.control,
            layer,
            (bytemuck::cast_slice(&control[..]), 4),
        );
        write_layer(
            cx,
            handles.normals,
            layer,
            (bytemuck::cast_slice(&gradients[..]), 4),
        );
        (true, shading)
    }

    pub(super) fn fade(
        &mut self,
        (atlas, tiles): (&Atlas, &TerrainTiles),
        count: u32,
    ) -> (&[u32], Range<usize>) {
        let whole = self.fades.len() != count as usize
            || self.faded_with.is_none_or(|then| atlas.now < then);
        self.faded_with = Some(atlas.now);
        self.fades_dirty = 0..0;
        let mut touched: Vec<TileId> = Vec::new();
        if whole {
            self.fades.clear();
            self.fades.resize(count as usize, 0);
            self.fades_dirty = 0..self.fades.len();
            self.moving.clear();
            touched.extend(atlas.slots.keys().copied());
        } else {
            for &layer in atlas.vacated.iter().rev() {
                self.set_fade(layer, 0);
            }
            touched.extend(atlas.changed.iter().copied());
            touched.extend(self.moving.iter().copied());
            touched.sort_unstable();
            touched.dedup();
        }
        for key in touched {
            self.refade((atlas, tiles), key);
        }
        (&self.fades, self.fades_dirty.clone())
    }

    fn refade(&mut self, (atlas, tiles): (&Atlas, &TerrainTiles), key: TileId) {
        let Some(&layer) = atlas.slots.get(&key) else {
            self.moving.remove(&key);
            return;
        };
        let shaded = tiles
            .terrains
            .iter()
            .find(|entry| entry.terrain == key.0)
            .is_some_and(|entry| shaded_elsewhere(key.1.level, entry.surroundings));
        let (normals, materials) = fades_of(atlas, &key, shaded);
        let timed = atlas.surfaced.contains_key(&key);
        let normals_move = normals > 0.0 && (timed || !shaded);
        if normals_move || (materials > 0.0 && timed) {
            self.moving.insert(key);
        } else {
            self.moving.remove(&key);
        }
        self.set_fade(layer, fade_half(normals, materials));
    }

    fn set_fade(&mut self, layer: u32, half: u32) {
        let at = layer as usize / 2;
        let shift = (layer % 2) * 16;
        let Some(word) = self.fades.get_mut(at) else {
            return;
        };
        let value = (*word & !(0xFFFF << shift)) | (half << shift);
        if value != *word {
            *word = value;
            self.fades_dirty = widen(self.fades_dirty.clone(), at);
        }
    }
}

fn fade_since(now: f32, at: f32) -> f32 {
    let since = engine_math::max(now - at, 0.0);
    1.0 - engine_math::min(since / ARRIVAL_SECONDS, 1.0)
}

fn fades_of(atlas: &Atlas, key: &TileId, shaded: bool) -> (f32, f32) {
    let materials = atlas
        .surfaced
        .get(key)
        .map_or(1.0, |&at| fade_since(atlas.now, at));
    if shaded {
        return (materials, materials);
    }
    let normals = atlas
        .arrivals
        .get(key)
        .map_or(0.0, |&at| fade_since(atlas.now, at));
    (normals, materials)
}

pub(super) fn fade_half(normals: f32, materials: f32) -> u32 {
    let byte = |fade: f32| (engine_math::clamp(fade, 0.0, 1.0) * 255.0 + 0.5) as u32;
    byte(normals) | (byte(materials) << 8)
}

fn first_sample(entry: &TerrainEntry, key: TileKey, spacing: f32) -> [i32; 2] {
    let offset = if entry.surroundings {
        entry.cutout.map_or(Vec2::ZERO, |(low, _)| -low)
    } else {
        Vec2::ZERO
    };
    let quads = i64::from(engine_heightfield::TILE_QUADS);
    [
        (i64::from(key.x) * quads + (offset.x / spacing).round() as i64) as i32,
        (i64::from(key.z) * quads + (offset.y / spacing).round() as i64) as i32,
    ]
}

fn write_layer(
    cx: &mut Prepare<'_, TerrainGpu>,
    texture: Texture<TerrainGpu>,
    layer: u32,
    (texels, bytes): (&[u8], u32),
) {
    cx.write_texture(
        texture,
        wgpu::Origin3d {
            x: 0,
            y: 0,
            z: layer,
        },
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LAYER_SIZE * bytes),
            rows_per_image: Some(LAYER_SIZE),
        },
        wgpu::Extent3d {
            width: LAYER_SIZE,
            height: LAYER_SIZE,
            depth_or_array_layers: 1,
        },
    );
}

fn write_errors(
    cx: &mut Prepare<'_, TerrainGpu>,
    texture: Texture<TerrainGpu>,
    layer: u32,
    pyramid: &[u16],
) {
    let mut start = 0;
    for mip in 0..SURFACE_ERROR_MIPS {
        let side = ERROR_CELLS >> mip;
        let count = (side * side) as usize;
        let texels: &[u16] = pyramid.get(start..start + count).unwrap_or_default();
        start += count;
        cx.write_texture_level(
            texture,
            mip,
            wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            bytemuck::cast_slice(texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side * 2),
                rows_per_image: Some(side),
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn write_water(
    cx: &mut Prepare<'_, TerrainGpu>,
    texture: Texture<TerrainGpu>,
    layer: u32,
    water: Option<&Arc<[u16]>>,
) {
    write_errors(cx, texture, layer, water_halves(water));
}

fn write_peaks(
    cx: &mut Prepare<'_, TerrainGpu>,
    texture: Texture<TerrainGpu>,
    layer: u32,
    peaks: &[f32],
) {
    for mip in 0..SURFACE_ERROR_MIPS {
        let side = ERROR_CELLS >> mip;
        cx.write_texture_level(
            texture,
            mip + 1,
            wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            bytemuck::cast_slice(peaks_mip(peaks, mip)),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side * 4),
                rows_per_image: Some(side),
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn peaks_mip(peaks: &[f32], mip: u32) -> &[f32] {
    let side = |level: u32| (ERROR_CELLS >> level) as usize;
    let start: usize = (0..mip).map(|finer| side(finer) * side(finer)).sum();
    &peaks[start..start + side(mip) * side(mip)]
}

fn listed_tile(tiles: &[TileFace], key: TileKey) -> Option<&TileFace> {
    tiles.iter().find(|tile| tile.key == key)
}

type WaterErrors<'a> = HashMap<TileId, (Stamp, &'a Arc<[u16]>)>;

fn water_errors(line: &Waterline, wanted: bool) -> WaterErrors<'_> {
    if !wanted {
        return HashMap::default();
    }
    line.errors
        .iter()
        .map(|held| ((held.terrain, held.key), (held.stamp, &held.errors)))
        .collect()
}

fn errors_of<'a>(found: &WaterErrors<'a>, id: TileId, stamp: Stamp) -> Option<&'a Arc<[u16]>> {
    found
        .get(&id)
        .filter(|(at, _)| *at == stamp)
        .map(|&(_, errors)| errors)
}

fn same_errors(a: Option<&Arc<[u16]>>, b: Option<&Arc<[u16]>>) -> bool {
    water_halves(a) == water_halves(b)
}

fn water_halves(errors: Option<&Arc<[u16]>>) -> &[u16] {
    errors.map_or(ErrorPyramid::dry().halves(), |held| &held[..])
}

fn held_control(tile: &TileFace, shadings: Option<&TerrainBakes>) -> Option<Box<[u32]>> {
    tile.control.as_deref().map(Box::from).or_else(|| {
        shadings
            .and_then(|held| held.shading(&tile.key))
            .map(|shading| shading.control_texels())
    })
}

fn held_gradients(tile: &TileFace, shadings: Option<&TerrainBakes>) -> Option<Box<[u32]>> {
    shadings
        .and_then(|held| held.shading(&tile.key))
        .map(|shading| shading.gradient_texels())
}

fn ancestor_texels(
    (tiles, pyramid): (&[TileFace], Pyramid),
    key: TileKey,
    held: impl Fn(&TileFace) -> Option<Box<[u32]>>,
) -> Box<[u32]> {
    let size = LAYER_SIZE as usize;
    let mut out = vec![0_u32; size * size].into_boxed_slice();
    let (mut at, mut shift) = (key, 0_u32);
    let values = loop {
        let Some(parent) = pyramid.parent(at) else {
            return out;
        };
        (at, shift) = (parent, shift + 1);
        if let Some(values) = listed_tile(tiles, at).and_then(&held) {
            break values;
        }
    };
    let quads = i64::from(engine_heightfield::TILE_QUADS);
    let half = 1_i64 << (shift - 1);
    let nearest = |texel: usize, tile: u16, ancestor: u16| {
        let sample = i64::from(tile) * quads + texel as i64 - 1;
        let there = (sample + half).div_euclid(1 << shift) - i64::from(ancestor) * quads + 1;
        there.clamp(0, size as i64 - 1) as usize
    };
    let columns: Vec<usize> = (0..size).map(|x| nearest(x, key.x, at.x)).collect();
    for (z, row) in out.chunks_exact_mut(size).enumerate() {
        let from = nearest(z, key.z, at.z) * size;
        for (value, &x) in row.iter_mut().zip(&columns) {
            *value = values[from + x];
        }
    }
    out
}
