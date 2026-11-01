use std::ops::Range;

use engine_places::ground::{Pyramid, TerrainTiles, TileFace, TileKey};
use engine_render::{Prepare, wgpu};

use super::part::{Handles, TerrainGpu};
use super::{ARRIVAL_SECONDS, LAYER_SIZE, widen};
use crate::gpu::{CONTROL_GROUP, ControlJob, NORMALS_GROUP, NormalJob, SURFACE_ERROR_MIPS};
use crate::shading::TileShading;
use crate::state::{Atlas, Bakes, TileId, Upload};

const HEIGHT_MIPS: u32 = 1 + SURFACE_ERROR_MIPS;
const ERROR_CELLS: u32 = engine_heightfield::TILE_QUADS / 2;

#[derive(Default)]
pub(crate) struct TileAtlas {
    layers: u32,
    normal_jobs: Vec<NormalJob>,
    control_jobs: Vec<ControlJob>,
    pub(crate) uploads: u32,
    pub(crate) controls: u32,
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

    pub(super) fn upload(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (tiles, atlas, bakes): (&TerrainTiles, &Atlas, &Bakes),
    ) {
        self.uploads = 0;
        self.controls = 0;
        self.normal_jobs.clear();
        self.control_jobs.clear();
        let layers = atlas.capacity.max(1);
        let whole = layers != self.layers;
        self.layers = layers;
        let array = |mips: u32| {
            (
                wgpu::Extent3d {
                    width: LAYER_SIZE,
                    height: LAYER_SIZE,
                    depth_or_array_layers: layers,
                },
                mips,
            )
        };
        let (size, mips) = array(HEIGHT_MIPS);
        cx.allocate(handles.heights, size, mips);
        let (size, mips) = array(1);
        cx.allocate(handles.normals, size, mips);
        cx.allocate(handles.control, size, mips);
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
        let uploads: Vec<(TileId, u32, Upload)> = if whole {
            atlas
                .slots
                .iter()
                .map(|(&id, &layer)| (id, layer, Upload::Arrived))
                .collect()
        } else {
            atlas.uploads.clone()
        };
        for (id, layer, upload) in uploads {
            let Some(entry) = tiles.terrains.iter().find(|entry| entry.terrain == id.0) else {
                continue;
            };
            let Some(tile) = atlas.tile(id.0, id.1) else {
                continue;
            };
            write_errors(cx, handles, layer, tile);
            if upload == Upload::Errors {
                continue;
            }
            let shades = TileShading::shades(id.1.level, entry.surroundings);
            write_layer(cx, handles.heights, layer, bytemuck::cast_slice(&tile.heights));
            let peaks = peaks(&tile.heights);
            write_peaks(cx, handles.heights, layer, &peaks);
            let spacing = entry.pyramid.spacing(u32::from(id.1.level));
            if !shades {
                self.normal_jobs.push(NormalJob {
                    layer,
                    spacing: spacing.to_bits(),
                    pad: [0; 2],
                });
            }
            let shading = bakes.of(id.0).and_then(|held| held.shadings.get(&id.1));
            if shades {
                let listing = (&entry.tiles[..], entry.pyramid);
                let (control, gradients) = match shading {
                    Some(shading) => (shading.control_texels(), shading.gradient_texels()),
                    None => (
                        ancestor_texels(listing, id.1, |tile| held_control(tile, None)),
                        ancestor_texels(listing, id.1, |_| None),
                    ),
                };
                write_layer(cx, handles.control, layer, bytemuck::cast_slice(&control[..]));
                write_layer(cx, handles.normals, layer, bytemuck::cast_slice(&gradients[..]));
            } else if let Some(control) = &tile.control {
                write_layer(cx, handles.control, layer, bytemuck::cast_slice(&control[..]));
            } else if entry.surface_program {
                if let Some(texels) = &tile.rivers {
                    write_rivers(cx, handles.rivers, layer, bytemuck::cast_slice(&texels[..]));
                }
                let first = first_sample(&entry.pyramid, id.1, spacing);
                self.control_jobs.push(ControlJob {
                    layer,
                    level: u32::from(id.1.level),
                    first,
                    spacing,
                    terrain: tiles
                        .terrains
                        .iter()
                        .position(|held| held.terrain == id.0)
                        .unwrap_or(0) as u32,
                    pad: [0; 2],
                });
            } else {
                let listing = (&entry.tiles[..], entry.pyramid);
                let control = ancestor_texels(listing, id.1, |tile| held_control(tile, None));
                write_layer(cx, handles.control, layer, bytemuck::cast_slice(&control[..]));
            }
            self.uploads += 1;
        }
        if !self.normal_jobs.is_empty() {
            let bytes = std::mem::size_of_val(self.normal_jobs.as_slice()) as u64;
            cx.allocate_buffer(handles.normal_jobs, bytes);
            cx.write(handles.normal_jobs, 0, &self.normal_jobs);
        }
        if !self.control_jobs.is_empty() {
            let bytes = std::mem::size_of_val(self.control_jobs.as_slice()) as u64;
            cx.allocate_buffer(handles.control_jobs, bytes);
            cx.write(handles.control_jobs, 0, &self.control_jobs);
            self.controls = self.control_jobs.len() as u32;
        }
    }
}

pub(super) fn fades<'w>(atlas: &Atlas, words: &'w mut Vec<u32>, fade_words: u32) -> (&'w [u32], Range<usize>) {
    let mut dirty = 0..0;
    if words.len() != fade_words as usize {
        words.clear();
        words.resize(fade_words as usize, 0);
        dirty = 0..words.len();
    }
    let mut next = vec![0_u32; words.len()];
    for (id, &layer) in &atlas.slots {
        let (normals, materials) = fades_of(atlas, id);
        let at = layer as usize / 2;
        let shift = (layer % 2) * 16;
        if let Some(word) = next.get_mut(at) {
            *word |= fade_half(normals, materials) << shift;
        }
    }
    for (at, (held, now)) in words.iter_mut().zip(next).enumerate() {
        if *held != now {
            *held = now;
            dirty = widen(dirty, at);
        }
    }
    (words, dirty)
}

fn fade_since(atlas: &Atlas, at: f32) -> f32 {
    let since = engine_math::max(atlas.now - at, 0.0);
    1.0 - engine_math::min(since / ARRIVAL_SECONDS, 1.0)
}

fn fades_of(atlas: &Atlas, key: &TileId) -> (f32, f32) {
    let materials = match atlas.surfaced.get(key) {
        None => 1.0,
        Some(&at) => fade_since(atlas, at),
    };
    let normals = match atlas.arrivals.get(key) {
        Some(&at) => fade_since(atlas, at),
        None => 0.0,
    };
    (normals, materials)
}

pub(super) fn fade_half(normals: f32, materials: f32) -> u32 {
    let byte = |fade: f32| (engine_math::clamp(fade, 0.0, 1.0) * 255.0 + 0.5) as u32;
    byte(normals) | (byte(materials) << 8)
}

fn first_sample(pyramid: &Pyramid, key: TileKey, spacing: f32) -> [i32; 2] {
    let corner = pyramid.tile_origin(key);
    [
        (corner.x / spacing).round() as i32,
        (corner.y / spacing).round() as i32,
    ]
}

fn texel_layout(bytes: u32) -> wgpu::TexelCopyBufferLayout {
    wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(LAYER_SIZE * bytes),
        rows_per_image: Some(LAYER_SIZE),
    }
}

fn layer_extent() -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: LAYER_SIZE,
        height: LAYER_SIZE,
        depth_or_array_layers: 1,
    }
}

fn write_layer(cx: &mut Prepare<'_, TerrainGpu>, texture: engine_render::Texture<TerrainGpu>, layer: u32, texels: &[u8]) {
    cx.write_texture(texture, wgpu::Origin3d { x: 0, y: 0, z: layer }, texels, texel_layout(4), layer_extent());
}

fn write_rivers(cx: &mut Prepare<'_, TerrainGpu>, texture: engine_render::Texture<TerrainGpu>, layer: u32, texels: &[u8]) {
    cx.write_texture(texture, wgpu::Origin3d { x: 0, y: 0, z: layer }, texels, texel_layout(8), layer_extent());
}

fn write_errors(cx: &mut Prepare<'_, TerrainGpu>, handles: &Handles, layer: u32, tile: &TileFace) {
    for (pyramid, texture) in [(&tile.ground_errors, handles.errors), (&tile.water_errors, handles.water_errors)] {
        let mut start = 0;
        for mip in 0..SURFACE_ERROR_MIPS {
            let side = ERROR_CELLS >> mip;
            let count = (side * side) as usize;
            let texels: &[u16] = pyramid.get(start..start + count).unwrap_or_default();
            start += count;
            cx.write_texture(
                texture,
                wgpu::Origin3d { x: 0, y: 0, z: layer },
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
}

fn write_peaks(cx: &mut Prepare<'_, TerrainGpu>, texture: engine_render::Texture<TerrainGpu>, layer: u32, peaks: &[f32]) {
    for mip in 0..SURFACE_ERROR_MIPS {
        let side = ERROR_CELLS >> mip;
        cx.write_texture(
            texture,
            wgpu::Origin3d { x: 0, y: 0, z: layer },
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

fn listed_tile(tiles: &[TileFace], key: TileKey) -> Option<&TileFace> {
    tiles.iter().find(|tile| tile.key == key)
}

fn held_control(tile: &TileFace, shading: Option<&TileShading>) -> Option<Box<[u32]>> {
    tile.control
        .as_deref()
        .map(Box::from)
        .or_else(|| shading.map(TileShading::control_texels))
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

fn peaks(heights: &[f32]) -> Box<[f32]> {
    let stride = LAYER_SIZE as usize;
    let height = |x: usize, z: usize| heights.get((z + 1) * stride + x + 1).copied();
    let mut side = ERROR_CELLS as usize;
    let mut level: Vec<f32> = Vec::with_capacity(side * side);
    for cell_z in 0..side {
        for cell_x in 0..side {
            let mut peak = -f32::MAX;
            for z in 2 * cell_z..=2 * cell_z + 2 {
                for x in 2 * cell_x..=2 * cell_x + 2 {
                    peak = engine_math::max(peak, height(x, z).unwrap_or(peak));
                }
            }
            level.push(peak);
        }
    }
    let texels = ((4_usize.pow(SURFACE_ERROR_MIPS)) - 1) / 3;
    let mut all = Vec::with_capacity(texels);
    all.extend_from_slice(&level);
    while side > 1 {
        let half = side / 2;
        let peak = |x: usize, z: usize| level[z * side + x];
        let coarser: Vec<f32> = (0..half * half)
            .map(|at| {
                let (x, z) = (2 * (at % half), 2 * (at / half));
                engine_math::max(
                    engine_math::max(peak(x, z), peak(x + 1, z)),
                    engine_math::max(peak(x, z + 1), peak(x + 1, z + 1)),
                )
            })
            .collect();
        all.extend_from_slice(&coarser);
        level = coarser;
        side = half;
    }
    all.into_boxed_slice()
}

fn peaks_mip(peaks: &[f32], mip: u32) -> &[f32] {
    let side = |level: u32| (ERROR_CELLS >> level) as usize;
    let start: usize = (0..mip).map(|finer| side(finer) * side(finer)).sum();
    &peaks[start..start + side(mip) * side(mip)]
}
