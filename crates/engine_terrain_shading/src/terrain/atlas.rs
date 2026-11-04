use std::sync::Arc;

use engine_core::Stamp;
use engine_core::collections::HashMap;
use engine_heightfield::ErrorPyramid;
use engine_heightfield::tile::shaded_elsewhere;
use engine_places::ground::{TerrainEntry, TerrainTiles, TileFace, TileKey};
use engine_places::water::Waterline;
use engine_render::{Buffer, ComputePipeline, Count, Encode, Prepare, Texture, wgpu};
use glam::Vec2;

use super::LAYER_SIZE;
use super::part::{Handles, TerrainGpu};
use crate::gpu::{
    ANCESTORS_GROUP, AncestorJob, CONTROL_GROUP, ControlJob, NORMALS_GROUP, NOT_RESIDENT,
    NormalJob, PEAKS_GROUP, RIVER_BEDS, RIVER_EDGES, SURFACE_ERROR_MIPS,
};
use crate::state::{Atlas, Bakes, TerrainBakes, TileId, Upload, listed_tile};

const HEIGHT_MIPS: u32 = 1 + SURFACE_ERROR_MIPS;
const ERROR_CELLS: u32 = engine_heightfield::TILE_QUADS / 2;
const FEWEST_LAYERS: u32 = 16;
const RIVER_PLANES: u32 = 4;
const BED_SAMPLES: usize = (LAYER_SIZE * LAYER_SIZE) as usize;

const _: () = assert!(
    LAYER_SIZE >= 2 * ERROR_CELLS && LAYER_SIZE >> SURFACE_ERROR_MIPS >= 1,
    "height_mips_hold_peak_mips"
);

#[derive(Default)]
pub(crate) struct TileAtlas {
    layers: u32,
    water: HashMap<TileId, Arc<[u16]>>,
    normal_jobs: Vec<NormalJob>,
    control_jobs: Vec<ControlJob>,
    peak_layers: Vec<u32>,
    control_ancestors: Vec<AncestorJob>,
    normal_ancestors: Vec<AncestorJob>,
    pub(crate) controls: u32,
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

    pub(super) fn peak_jobs(&self) -> usize {
        self.peak_layers.len()
    }

    pub(super) fn ancestor_jobs(&self) -> usize {
        self.control_ancestors.len() + self.normal_ancestors.len()
    }

    pub(super) fn normal_groups(&self) -> [u32; 3] {
        let side = LAYER_SIZE.div_ceil(NORMALS_GROUP);
        [side, side, self.normal_jobs.len() as u32]
    }

    pub(super) fn control_groups(&self) -> [u32; 3] {
        let side = LAYER_SIZE.div_ceil(CONTROL_GROUP);
        [side, side, self.control_jobs.len() as u32]
    }

    pub(super) fn peaks(
        &self,
        cx: &mut Encode<'_, TerrainGpu>,
        pipelines: &[ComputePipeline<TerrainGpu>],
    ) -> u32 {
        let jobs = self.peak_layers.len() as u32;
        let mut dispatched = 0;
        for (mip, pipeline) in pipelines.iter().enumerate() {
            let groups = (ERROR_CELLS >> mip).div_ceil(PEAKS_GROUP);
            dispatched += cx.dispatch(pipeline, Count::Groups([groups, groups, jobs]));
        }
        dispatched
    }

    pub(super) fn ancestors(
        &self,
        cx: &mut Encode<'_, TerrainGpu>,
        pipelines: &[ComputePipeline<TerrainGpu>; 2],
    ) -> u32 {
        let side = LAYER_SIZE.div_ceil(ANCESTORS_GROUP);
        let lists = [self.control_ancestors.len(), self.normal_ancestors.len()];
        let mut dispatched = 0;
        for (pipeline, jobs) in pipelines.iter().zip(lists) {
            if jobs > 0 {
                dispatched += cx.dispatch(pipeline, Count::Groups([side, side, jobs as u32]));
            }
        }
        dispatched
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
        self.normal_jobs.clear();
        self.control_jobs.clear();
        self.peak_layers.clear();
        self.control_ancestors.clear();
        self.normal_ancestors.clear();
        let capacity = atlas.capacity.max(1);
        let needed = atlas.next_layer.min(capacity);
        let whole = needed > self.layers;
        if whole {
            self.layers = needed.next_power_of_two().max(FEWEST_LAYERS).min(capacity);
            self.water.clear();
        }
        let layers = self.layers.max(1);
        let gpu = tiles.terrains.iter().any(|entry| entry.surface_program);
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
        cx.allocate(
            handles.rivers,
            wgpu::Extent3d {
                width: RIVER_PLANES * LAYER_SIZE,
                height: LAYER_SIZE,
                depth_or_array_layers: if gpu { layers } else { 1 },
            },
            1,
        );
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
        if !atlas.vacated.is_empty() {
            self.water.retain(|id, _| atlas.slots.contains_key(id));
        }
        let uploads: Vec<(TileId, u32, Upload)> = if whole {
            let mut all: Vec<(TileId, u32, Upload)> = atlas
                .slots
                .iter()
                .map(|(&id, &layer)| (id, layer, Upload::Whole))
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
            let Some(tile) = listed_tile(&entry.tiles, id.1) else {
                continue;
            };
            let placing = Placing {
                id,
                layer,
                index,
                entry,
                tile,
            };
            let shades = shaded_elsewhere(id.1.level, entry.surroundings);
            if upload == Upload::Whole {
                self.give_ground(
                    cx,
                    handles,
                    &placing,
                    shades,
                    errors_of(&found, id, tile.stamp),
                );
            }
            if shades {
                self.shade(cx, handles, &placing, (atlas, bakes));
            } else if upload == Upload::Whole {
                self.give_control(cx, handles, &placing, (atlas, bakes));
            }
        }
        if watered {
            self.rewater(cx, handles, atlas, &found);
        }
        write_jobs(cx, handles.normal_jobs, &self.normal_jobs);
        write_jobs(cx, handles.control_jobs, &self.control_jobs);
        write_jobs(cx, handles.peak_jobs, &self.peak_layers);
        write_jobs(cx, handles.control_ancestors, &self.control_ancestors);
        write_jobs(cx, handles.normal_ancestors, &self.normal_ancestors);
        self.controls = self.control_jobs.len() as u32;
    }

    fn give_ground(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        placing: &Placing<'_>,
        shades: bool,
        water: Option<&Arc<[u16]>>,
    ) {
        let Placing {
            id, layer, tile, ..
        } = *placing;
        write_layer(
            cx,
            handles.heights,
            layer,
            (bytemuck::cast_slice(&tile.heights[..]), 4),
        );
        write_errors(cx, handles.errors, layer, &tile.ground_errors);
        self.peak_layers.push(layer);
        if !shades {
            self.normal_jobs.push(NormalJob {
                layer,
                spacing: tile.spacing.to_bits(),
                pad: [0; 2],
            });
        }
        write_water(cx, handles.water_errors, layer, water);
        match water {
            Some(water) => {
                self.water.insert(id, Arc::clone(water));
            }
            None => {
                self.water.remove(&id);
            }
        }
    }

    fn rewater(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        atlas: &Atlas,
        found: &WaterErrors<'_>,
    ) {
        for (&id, &(stamp, water)) in found {
            let Some(layer) = atlas.layer_of(id.0, id.1) else {
                continue;
            };
            if atlas.stamp_of(id) == Some(stamp) && !same_water(self.water.get(&id), Some(water)) {
                write_water(cx, handles.water_errors, layer, Some(water));
                self.water.insert(id, Arc::clone(water));
            }
        }
        self.water.retain(|&id, _| {
            let wet = found
                .get(&id)
                .is_some_and(|&(stamp, _)| atlas.stamp_of(id) == Some(stamp));
            if !wet && let Some(layer) = atlas.layer_of(id.0, id.1) {
                write_water(cx, handles.water_errors, layer, None);
            }
            wet
        });
    }

    fn give_control(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        placing: &Placing<'_>,
        (atlas, bakes): (&Atlas, &Bakes),
    ) {
        let Placing {
            id,
            layer,
            index,
            entry,
            tile,
        } = *placing;
        if let Some(values) = &tile.control {
            write_layer(
                cx,
                handles.control,
                layer,
                (bytemuck::cast_slice(&values[..]), 4),
            );
        } else if entry.surface_program {
            let mut rivers = 0;
            if let Some(edges) = &tile.river_edges {
                rivers |= RIVER_EDGES;
                write_river(cx, handles.rivers, (layer, 0), edges);
            }
            if let Some(beds) = &tile.river_beds {
                rivers |= RIVER_BEDS;
                for (plane, bed) in beds.as_chunks::<BED_SAMPLES>().0.iter().take(3).enumerate() {
                    write_river(cx, handles.rivers, (layer, plane as u32 + 1), bed);
                }
            }
            let spacing = tile.spacing;
            self.control_jobs.push(ControlJob {
                layer,
                rivers,
                first: first_sample(entry, id.1, spacing),
                spacing,
                terrain: index as u32,
                pad: [0; 2],
            });
        } else {
            self.control_ancestors.push(ancestor_job(
                atlas,
                (entry, bakes.of(id.0)),
                (layer, id.1),
                true,
            ));
        }
    }

    fn shade(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        placing: &Placing<'_>,
        (atlas, bakes): (&Atlas, &Bakes),
    ) {
        let Placing {
            id,
            layer,
            entry,
            tile,
            ..
        } = *placing;
        let held = bakes.of(id.0);
        let Some(shading) = held.and_then(|held| held.shading(tile)) else {
            let tile_at = (layer, id.1);
            self.control_ancestors
                .push(ancestor_job(atlas, (entry, held), tile_at, true));
            self.normal_ancestors
                .push(ancestor_job(atlas, (entry, held), tile_at, false));
            return;
        };
        write_layer(
            cx,
            handles.control,
            layer,
            (bytemuck::cast_slice(&shading.control_texels()[..]), 4),
        );
        write_layer(
            cx,
            handles.normals,
            layer,
            (bytemuck::cast_slice(&shading.gradient_texels()[..]), 4),
        );
    }
}

fn ancestor_job(
    atlas: &Atlas,
    (entry, held): (&TerrainEntry, Option<&TerrainBakes>),
    (layer, key): (u32, TileKey),
    control: bool,
) -> AncestorJob {
    let quads = engine_heightfield::TILE_QUADS as i32;
    let first_of = |key: TileKey| [i32::from(key.x) * quads - 1, i32::from(key.z) * quads - 1];
    let (mut at, mut shift) = (key, 0);
    while let Some(parent) = entry.pyramid.parent(at) {
        (at, shift) = (parent, shift + 1);
        let holds = listed_tile(&entry.tiles, at).is_some_and(|tile| {
            (control && tile.control.is_some())
                || held.is_some_and(|held| held.shading(tile).is_some())
        });
        if let Some(held_layer) = atlas.layer_of(entry.terrain, at).filter(|_| holds) {
            return AncestorJob {
                layer,
                held_layer,
                shift,
                pad: 0,
                first: first_of(key),
                held_first: first_of(at),
            };
        }
    }
    AncestorJob {
        layer,
        held_layer: NOT_RESIDENT,
        shift: 0,
        pad: 0,
        first: first_of(key),
        held_first: [0; 2],
    }
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

fn write_jobs<T: bytemuck::Pod>(
    cx: &mut Prepare<'_, TerrainGpu>,
    buffer: Buffer<TerrainGpu>,
    jobs: &[T],
) {
    if !jobs.is_empty() {
        cx.grow_buffer(buffer, std::mem::size_of_val(jobs) as u64);
        cx.write(buffer, 0, jobs);
    }
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

fn write_river(
    cx: &mut Prepare<'_, TerrainGpu>,
    texture: Texture<TerrainGpu>,
    (layer, plane): (u32, u32),
    texels: &[u16],
) {
    cx.write_texture(
        texture,
        wgpu::Origin3d {
            x: plane * LAYER_SIZE,
            y: 0,
            z: layer,
        },
        bytemuck::cast_slice(texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LAYER_SIZE * 2),
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

fn same_water(a: Option<&Arc<[u16]>>, b: Option<&Arc<[u16]>>) -> bool {
    std::ptr::eq(water_halves(a), water_halves(b))
}

fn water_halves(errors: Option<&Arc<[u16]>>) -> &[u16] {
    errors.map_or(ErrorPyramid::dry().halves(), |held| &held[..])
}
