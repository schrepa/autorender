use engine_heightfield::tile::shaded_elsewhere;
use engine_places::ground::{TerrainEntry, TerrainTiles, TileKey};
use engine_places::probes::ProbeCaptures;
use engine_places::shadows::ground_level;
use engine_render::{Prepare, wgpu};
use glam::{DVec3, Mat4, Vec2, Vec3, Vec4};

use super::part::{Handles, TerrainGpu, TerrainViews};
use super::surface::SurfaceTrees;
use super::tables::TerrainTables;
use super::{FrameTerrain, NO_HOLES, TerrainFrame, changed_run};
use crate::gpu::{
    PATCH_GRID as GRID, PATCH_GRID_VERTICES as GRID_VERTICES, PatchRecord, TerrainUniform,
    ViewUniform,
};
use crate::state::Atlas;
use crate::terrain_macro::MacroMaps;
use crate::terrain_sky::SkyMaps;

const PATCH_QUADS: u32 = GRID - 1;
const PATCHES_PER_TILE: u32 = engine_heightfield::TILE_QUADS / PATCH_QUADS;

#[derive(Default)]
pub(crate) struct PatchDraws {
    index_count: u32,
    indexed: bool,
    records: Vec<PatchRecord>,
    written: Vec<PatchRecord>,
    record_room: u64,
    fresh: bool,
    uniforms: Vec<TerrainUniform>,
    written_uniforms: Vec<TerrainUniform>,
    written_views: Vec<ViewUniform>,
    kept_stack: Vec<(u32, u32, u32)>,
}

pub(super) struct Layout<'a> {
    pub(super) atlas: &'a Atlas,
    pub(super) tables: &'a TerrainTables,
    pub(super) surface: &'a SurfaceTrees,
    pub(super) sky: &'a SkyMaps,
    pub(super) macro_maps: &'a MacroMaps,
    pub(super) index: u32,
}

impl PatchDraws {
    pub(super) fn index_count(&self) -> u32 {
        self.index_count
    }

    pub(super) fn lay_out(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (tiles, atlas, tables, surface, sky, macro_maps): (
            &TerrainTiles,
            &Atlas,
            &TerrainTables,
            &SurfaceTrees,
            &SkyMaps,
            &MacroMaps,
        ),
        views: &TerrainViews<'_>,
    ) -> TerrainFrame {
        if !self.indexed {
            let indices = grid_indices();
            self.index_count = indices.len() as u32;
            cx.allocate_buffer(
                handles.grid,
                std::mem::size_of_val(indices.as_slice()) as u64,
            );
            cx.write(handles.grid, 0, &indices);
            self.indexed = true;
        }
        let eye = views.eye.as_dvec3();
        std::mem::swap(&mut self.records, &mut self.written);
        std::mem::swap(&mut self.uniforms, &mut self.written_uniforms);
        self.records.clear();
        self.uniforms.clear();
        let mut frame = TerrainFrame::default();
        for (index, terrain) in tiles.terrains.iter().enumerate() {
            let parts = Layout {
                atlas,
                tables,
                surface,
                sky,
                macro_maps,
                index: index as u32,
            };
            let uniform = terrain_uniform(index, terrain, eye, views, &parts);
            self.uniforms.push(uniform);
            if !atlas.slots.keys().any(|(held, _)| *held == terrain.terrain) {
                continue;
            }
            let screen = ground_rect(views.camera, terrain, eye, uniform.relief[0]);
            frame.terrains.push(FrameTerrain {
                terrain: index,
                holes: atlas.holed.iter().any(|&(id, _)| id == terrain.terrain),
                maps: Vec::with_capacity(views.regions.len()),
                screen,
            });
        }
        for &(map, region) in &views.regions {
            let planes = region.planes.map(Vec4::from_array);
            for drawn in &mut frame.terrains {
                let terrain = &tiles.terrains[drawn.terrain];
                let parts = Layout {
                    atlas,
                    tables,
                    surface,
                    sky,
                    macro_maps,
                    index: drawn.terrain as u32,
                };
                let level = ground_level(&terrain.pyramid, region.texel);
                let first = self.records.len() as u32;
                kept_patches(
                    (&mut self.records, &mut self.kept_stack),
                    terrain,
                    (&planes[..], level),
                    Anchors::of(terrain, eye),
                    &parts,
                );
                let range = first..self.records.len() as u32;
                frame.shadow_patches += range.len() as u32;
                drawn.maps.push((map, range));
            }
        }
        let record_bytes = (self.records.len() * size_of::<PatchRecord>()) as u64;
        if record_bytes > self.record_room {
            self.record_room =
                (self.records.len() as u64).next_power_of_two() * size_of::<PatchRecord>() as u64;
            self.fresh = true;
        }
        cx.allocate_buffer(
            handles.patches,
            self.record_room.max(size_of::<PatchRecord>() as u64),
        );
        let run = if self.fresh {
            0..self.records.len()
        } else {
            changed_run(&self.written, &self.records)
        };
        self.fresh = false;
        if !run.is_empty() {
            let offset = (run.start * size_of::<PatchRecord>()) as u64;
            cx.write(handles.patches, offset, &self.records[run]);
        }
        cx.allocate_buffer(
            handles.terrains,
            (self.uniforms.len().max(1) * size_of::<TerrainUniform>()) as u64,
        );
        if self.uniforms != self.written_uniforms {
            cx.write(handles.terrains, 0, &self.uniforms);
        }
        frame
    }

    pub(super) fn write_views(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        views: &TerrainViews<'_>,
        captures: &ProbeCaptures,
    ) {
        let mut written = vec![ViewUniform {
            relative_view_projection: views.camera.to_cols_array_2d(),
            eye_offset: [0.0; 4],
            inverse_relative_view_projection: views
                .camera
                .as_dmat4()
                .inverse()
                .as_mat4()
                .to_cols_array_2d(),
        }];
        for face in &captures.faces {
            let relative = face.view;
            let at = 1 + face.face as usize;
            if written.len() <= at {
                written.resize(at + 1, ViewUniform::default());
            }
            written[at] = ViewUniform {
                inverse_relative_view_projection: relative
                    .as_dmat4()
                    .inverse()
                    .as_mat4()
                    .to_cols_array_2d(),
                relative_view_projection: relative.to_cols_array_2d(),
                eye_offset: (views.eye.as_dvec3() - face.center.as_dvec3())
                    .as_vec3()
                    .extend(0.0)
                    .to_array(),
            };
        }
        if written == self.written_views {
            return;
        }
        let views_bytes = written.len().max(super::VIEWS as usize) * size_of::<ViewUniform>();
        cx.allocate_buffer(handles.views, views_bytes as u64);
        cx.write(handles.views, 0, &written);
        self.written_views = written;
    }
}

#[derive(Clone, Copy)]
pub(super) struct TileRecord {
    key: TileKey,
    layer: u32,
    spacing: f32,
    holes: u32,
    terrain: u32,
}

impl TileRecord {
    pub(super) fn of(parts: &Layout<'_>, terrain: &TerrainEntry, key: TileKey) -> Option<Self> {
        let layer = parts.atlas.layer_of(terrain.terrain, key)?;
        let holes = parts.tables.holes_of((terrain.terrain, key));
        Some(Self {
            key,
            layer,
            spacing: terrain.pyramid.spacing(u32::from(key.level)),
            holes,
            terrain: parts.index,
        })
    }

    pub(super) fn kept_record(
        &self,
        (x, z): (u8, u8),
        holes: bool,
        anchors: Anchors,
    ) -> PatchRecord {
        let key = self.key;
        let anchor = anchors.at(u32::from(key.level));
        let first = [
            i64::from(key.x) * 256 + i64::from(x) * 16 - anchor[0],
            i64::from(key.z) * 256 + i64::from(z) * 16 - anchor[1],
        ];
        let texel = (u32::from(x) * 16) | ((u32::from(z) * 16) << 16);
        PatchRecord {
            first: [first[0] as i32, first[1] as i32],
            shape: [self.spacing, SKIRT_SPACINGS * self.spacing],
            atlas: [self.layer, texel],
            holes: if holes { self.holes } else { NO_HOLES },
            terrain: self.terrain,
        }
    }
}

const SKIRT_SPACINGS: f32 = 4.0;

fn kept_patches(
    (records, stack): (&mut Vec<PatchRecord>, &mut Vec<(u32, u32, u32)>),
    terrain: &TerrainEntry,
    (planes, level): (&[Vec4], u32),
    anchors: Anchors,
    parts: &Layout<'_>,
) {
    let pyramid = &terrain.pyramid;
    let top = pyramid.levels() - 1;
    let finest = level.min(top);
    let extent = pyramid.extent();
    let side = PATCHES_PER_TILE;
    let [tiles_x, tiles_z] = pyramid.tiles(top);
    stack.clear();
    stack.extend((0..tiles_z * side).flat_map(|z| (0..tiles_x * side).map(move |x| (top, x, z))));
    let stray = |from: u32| {
        let measured: f32 = (finest..from)
            .map(|at| {
                terrain
                    .level_errors
                    .get(at as usize)
                    .copied()
                    .unwrap_or(0.0)
            })
            .sum();
        measured + pyramid.spacing(from)
    };
    let held = |key: TileKey| parts.atlas.tile(terrain.terrain, key);
    let mut shared: Option<(TileKey, Option<TileRecord>)> = None;
    while let Some((at, x, z)) = stack.pop() {
        let spacing = pyramid.spacing(at);
        let size = PATCH_QUADS as f32 * spacing;
        let corner = Vec2::new(x as f32, z as f32) * size;
        if corner.x >= extent.x || corner.y >= extent.y || cut_out(terrain.cutout, corner, size) {
            continue;
        }
        let key = TileKey::new(at as u8, (x / side) as u16, (z / side) as u16);
        let Some(tile) = held(key) else {
            continue;
        };
        let (column, row) = (x % side, z % side);
        let [low, high] = tile.patch_bounds[(row * side + column) as usize];
        let min = terrain.origin + Vec3::new(corner.x, low - SKIRT_SPACINGS * spacing, corner.y);
        let max = terrain.origin + Vec3::new(corner.x + size, high, corner.y + size);
        if at > finest {
            let child = TileKey::new(at as u8 - 1, (2 * x / side) as u16, (2 * z / side) as u16);
            if held(child).is_some() {
                let widened = Vec3::Y * stray(at);
                if visible(planes, min - widened, max + widened) {
                    for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        stack.push((at - 1, 2 * x + dx, 2 * z + dz));
                    }
                }
                continue;
            }
        }
        if !visible(planes, min, max) {
            continue;
        }
        let record = match shared {
            Some((cached, record)) if cached == key => record,
            _ => {
                let record = TileRecord::of(parts, terrain, key);
                shared = Some((key, record));
                record
            }
        };
        if let Some(record) = record {
            let index = (row * side + column) as usize;
            let holes = tile.patch_holes[index / 64] & (1 << (index % 64)) != 0;
            records.push(record.kept_record((column as u8, row as u8), holes, anchors));
        }
    }
}

fn cut_out(cutout: Option<(Vec2, Vec2)>, corner: Vec2, size: f32) -> bool {
    cutout.is_some_and(|(first, last)| {
        corner.x >= first.x
            && corner.y >= first.y
            && corner.x + size <= last.x
            && corner.y + size <= last.y
    })
}

fn ground_rect(view: Mat4, terrain: &TerrainEntry, eye: DVec3, reach: f32) -> Option<[f32; 4]> {
    let (low, high) = terrain
        .tiles
        .iter()
        .map(|tile| (tile.bounds.0.y, tile.bounds.1.y))
        .fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(low, high), (bottom, top)| {
                (engine_math::min(low, bottom), engine_math::max(high, top))
            },
        );
    if low > high {
        return None;
    }
    let origin = (terrain.origin.as_dvec3() - eye).as_vec3();
    let extent = terrain.pyramid.extent();
    let min = origin + Vec3::new(0.0, low - reach, 0.0);
    let max = origin + Vec3::new(extent.x, high + reach, extent.y);
    projected_rect(
        [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
        view,
        min,
        max,
    )
}

fn projected_rect(rect: [f32; 4], view: Mat4, min: Vec3, max: Vec3) -> Option<[f32; 4]> {
    let base = view * min.extend(1.0);
    let size = max - min;
    let edges = [
        view.x_axis * size.x,
        view.y_axis * size.y,
        view.z_axis * size.z,
    ];
    let [mut left, mut bottom, mut right, mut top] = rect;
    for corner in 0..8 {
        let mut clip = base;
        for (axis, edge) in edges.iter().enumerate() {
            if corner & (1 << axis) != 0 {
                clip += *edge;
            }
        }
        if clip.w <= 0.0 {
            return None;
        }
        let (x, y) = (clip.x / clip.w, clip.y / clip.w);
        left = engine_math::min(left, x);
        bottom = engine_math::min(bottom, y);
        right = engine_math::max(right, x);
        top = engine_math::max(top, y);
    }
    Some([left, bottom, right, top])
}

const PATCH_ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0 => Sint32x2, 1 => Float32x2, 2 => Uint32x2, 3 => Uint32, 4 => Uint32];

pub(super) fn patch_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<PatchRecord>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &PATCH_ATTRIBUTES,
    }
}

pub(super) fn grid_indices() -> Vec<u16> {
    let at = |x: u32, z: u32| (z * GRID + x) as u16;
    let skirt = |edge: u32, along: u32| (GRID_VERTICES + edge * GRID + along) as u16;
    let mut indices = Vec::new();
    for z in 0..GRID - 1 {
        for x in 0..GRID - 1 {
            let (a, b, c, d) = (at(x, z), at(x + 1, z), at(x, z + 1), at(x + 1, z + 1));
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let last = GRID - 1;
    for along in 0..last {
        let (t0, t1, b0, b1) = (
            at(along, 0),
            at(along + 1, 0),
            skirt(0, along),
            skirt(0, along + 1),
        );
        indices.extend_from_slice(&[t0, t1, b0, t1, b1, b0]);
        let (t0, t1, b0, b1) = (
            at(along, last),
            at(along + 1, last),
            skirt(1, along),
            skirt(1, along + 1),
        );
        indices.extend_from_slice(&[t0, b0, t1, t1, b0, b1]);
        let (t0, t1, b0, b1) = (
            at(0, along),
            at(0, along + 1),
            skirt(2, along),
            skirt(2, along + 1),
        );
        indices.extend_from_slice(&[t0, b0, t1, t1, b0, b1]);
        let (t0, t1, b0, b1) = (
            at(last, along),
            at(last, along + 1),
            skirt(3, along),
            skirt(3, along + 1),
        );
        indices.extend_from_slice(&[t0, t1, b0, t1, b1, b0]);
    }
    indices
}

#[derive(Clone, Copy)]
pub(super) struct Anchors {
    cells: [i64; 2],
    levels: u32,
}

impl Anchors {
    pub(super) fn of(terrain: &TerrainEntry, eye: DVec3) -> Self {
        let pyramid = &terrain.pyramid;
        let levels = pyramid.levels();
        let top = f64::from(pyramid.spacing(levels - 1));
        let cells = ((eye - terrain.origin.as_dvec3()) / top).round();
        Self {
            cells: [cells.x as i64, cells.z as i64],
            levels,
        }
    }

    fn at(self, level: u32) -> [i64; 2] {
        let scale = 1_i64 << (self.levels - 1 - level);
        [self.cells[0] * scale, self.cells[1] * scale]
    }
}

fn terrain_uniform(
    _: usize,
    terrain: &TerrainEntry,
    eye: DVec3,
    views: &TerrainViews<'_>,
    parts: &Layout<'_>,
) -> TerrainUniform {
    let pyramid = &terrain.pyramid;
    let anchors = Anchors::of(terrain, eye);
    let top = f64::from(pyramid.spacing(pyramid.levels() - 1));
    let anchor = terrain.origin.as_dvec3()
        + DVec3::new(
            anchors.cells[0] as f64 * top,
            0.0,
            anchors.cells[1] as f64 * top,
        );
    let relative = (anchor - eye).as_vec3();
    let tables = parts.tables;
    let cover = match &views.cover {
        None => [0.0; 4],
        Some((cover, _)) if terrain.surroundings => [0.0, 0.0, cover.density, 0.0],
        Some((cover, pixels)) => [
            cover.width_of(terrain.terrain) * *pixels,
            0.0,
            cover.density,
            0.0,
        ],
    };
    let deepest = tables
        .palette_of(parts.index as usize, terrain.materials.len())
        .iter()
        .filter(|entry| entry.params[3] > 0.0)
        .map(|entry| entry.relief[0])
        .fold(0.0, engine_math::max);
    let (first, finest) = parts.surface.of(parts.index as usize);
    let water = parts.surface.water_slot(parts.index as usize);
    TerrainUniform {
        anchor: [relative.x, relative.y, relative.z, pyramid.spacing(0)],
        eye: eye.as_vec3().extend(0.0).to_array(),
        surface: [
            first,
            tables.fade_words,
            anchors.cells[0] as i32 as u32,
            anchors.cells[1] as i32 as u32,
        ],
        origin: terrain.origin.extend(0.0).to_array(),
        directory: [
            tables
                .directory_bases
                .get(parts.index as usize)
                .copied()
                .unwrap_or(0),
            pyramid.levels(),
            tables
                .palette_bases
                .get(parts.index as usize)
                .copied()
                .unwrap_or(0),
            parts.sky.layer(parts.index as usize),
        ],
        relief: [deepest, views.settings.relief_steps as f32, finest, 0.0],
        cover,
        pixel_error: [terrain.pixel_error, 0.0, 0.0, 0.0],
        macro_map: [parts.macro_maps.layer(parts.index as usize), 0, 0, 0],
        steep: {
            let [material, low, high] = terrain.steep.map_or([0.0; 3], |steep| {
                [
                    f32::from(steep.material) + 1.0,
                    steep.gradient[0],
                    steep.gradient[1],
                ]
            });
            let averaged = shaded_elsewhere(1, terrain.surroundings);
            [material, low, high, if averaged { 1.0 } else { 0.0 }]
        },
        water: [
            0,
            water.map_or(0, |(slot, _)| slot + 1),
            water.map_or(0, |(_, first)| first),
            0,
        ],
    }
}

pub(crate) fn planes(clip: Mat4, sides_only: bool) -> Vec<Vec4> {
    let rows = [clip.row(0), clip.row(1), clip.row(2), clip.row(3)];
    let mut planes = vec![
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
    ];
    if !sides_only {
        planes.push(rows[2]);
        planes.push(rows[3] - rows[2]);
    }
    planes
}

pub(crate) fn visible(planes: &[Vec4], min: Vec3, max: Vec3) -> bool {
    planes.iter().all(|plane| {
        let corner = Vec3::new(
            if plane.x >= 0.0 { max.x } else { min.x },
            if plane.y >= 0.0 { max.y } else { min.y },
            if plane.z >= 0.0 { max.z } else { min.z },
        );
        plane.truncate().dot(corner) + plane.w >= 0.0
    })
}
