use super::atlas::TileAtlas;
use super::tables::TerrainTables;
use super::*;
use engine_core::glam::Vec2;
use engine_core::terrain_data::{PATCH_QUADS, PATCHES_PER_TILE, Pyramid};

pub(crate) struct PatchDraws {
    indices: wgpu::Buffer,
    index_count: u32,
    instances: wgpu::Buffer,
    records: Vec<PatchRecord>,
    written: Vec<PatchRecord>,
    fresh: bool,
    uniforms: Vec<u8>,
    written_uniforms: Vec<u8>,
    uniforms_generation: Option<u64>,
    pub(super) views: wgpu::Buffer,
    written_views: Option<ViewUniform>,
    kept_stack: Vec<(u32, u32, u32)>,
    kept_ground: HashMap<(u64, TileKey), (Arc<engine_core::Tile>, GroundChange)>,
    ground_changes: Vec<GroundChange>,
    eye: Vec3,
    pub(crate) frame: TerrainFrame,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GroundChange {
    pub(crate) min: Vec3,
    pub(crate) max: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) level: u32,
}

pub(super) struct Layout<'a> {
    pub(super) atlas: &'a TileAtlas,
    pub(super) tables: &'a TerrainTables,
    pub(super) surface: &'a SurfaceTrees,
    pub(super) sky: &'a SkyMaps,
    pub(super) macro_maps: &'a MacroMaps,
}

impl PatchDraws {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let indices = grid_indices();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine/terrain-indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            indices: index_buffer,
            index_count: indices.len() as u32,
            instances: instance_buffer(device, 256),
            records: Vec::new(),
            written: Vec::new(),
            fresh: true,
            uniforms: Vec::new(),
            written_uniforms: Vec::new(),
            uniforms_generation: None,
            views: crate::buffers::uniform(device, "engine/terrain-views", VIEWS * SLOT),
            written_views: None,
            kept_stack: Vec::new(),
            kept_ground: HashMap::default(),
            ground_changes: Vec::new(),
            eye: Vec3::ZERO,
            frame: TerrainFrame::default(),
        }
    }

    pub(super) fn begin(&mut self, eye: Vec3) {
        let mut terrains = std::mem::take(&mut self.frame.terrains);
        terrains.clear();
        self.frame = TerrainFrame {
            terrains,
            ..TerrainFrame::default()
        };
        self.eye = eye;
    }

    pub(super) fn eye(&self) -> Vec3 {
        self.eye
    }

    #[cfg(debug_assertions)]
    pub(super) fn records(&self) -> &[PatchRecord] {
        &self.records
    }

    pub(super) fn lay_out(
        &mut self,
        device: &wgpu::Device,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
        parts: &Layout<'_>,
    ) {
        let eye = views.eye.as_dvec3();
        std::mem::swap(&mut self.records, &mut self.written);
        std::mem::swap(&mut self.uniforms, &mut self.written_uniforms);
        self.records.clear();
        self.uniforms.clear();
        self.uniforms
            .resize(terrains.terrains.len() * SLOT as usize, 0);
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            let uniform = terrain_uniform(index, terrain, eye, views, parts);
            let at = index * SLOT as usize;
            self.uniforms[at..at + size_of::<TerrainUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
            if !terrain.drawn {
                continue;
            }
            let screen = ground_rect(views.camera, terrain, eye, uniform.relief[0]);
            self.frame.terrains.push(FrameTerrain {
                terrain: index,
                uniform: (index as u64 * SLOT) as u32,
                holes: parts.atlas.holed.iter().any(|&(id, _)| id == terrain.id),
                maps: Vec::with_capacity(views.regions.len()),
                screen,
            });
        }
        for &(map, region) in views.regions {
            self.lay_out_kept(terrains, (map, region, eye), parts);
        }
        let record_bytes = (self.records.len() * size_of::<PatchRecord>()) as u64;
        if record_bytes > self.instances.size() {
            self.instances =
                instance_buffer(device, (self.records.len() as u64).next_power_of_two());
            self.fresh = true;
        }
    }

    fn lay_out_kept(
        &mut self,
        terrains: &RenderTerrains,
        (map, region, eye): (usize, crate::shadows::KeptRefresh, DVec3),
        parts: &Layout<'_>,
    ) {
        let planes = region.planes.map(Vec4::from_array);
        for frame in &mut self.frame.terrains {
            let terrain = &terrains.terrains[frame.terrain];
            let level = crate::shadow_fit::ground_level(&terrain.pyramid, region.texel);
            let first = self.records.len() as u32;
            kept_patches(
                (&mut self.records, &mut self.kept_stack),
                terrain,
                (&planes[..], level),
                Anchors::of(terrain, eye),
                parts,
            );
            let range = first..self.records.len() as u32;
            self.frame.shadow_patches += range.len() as u32;
            frame.maps.push((map, range));
        }
    }

    pub(super) fn follow_ground(&mut self, terrains: &RenderTerrains, atlas: &TileAtlas) {
        self.ground_changes.clear();
        for &slot in &atlas.held_changes {
            let before = self.kept_ground.remove(&slot);
            let now = atlas.held.get(&slot).and_then(|tile| {
                let terrain = terrains
                    .terrains
                    .iter()
                    .find(|terrain| terrain.id == slot.0)?;
                Some((Arc::clone(tile), ground_box(terrain, slot.1, tile.bounds())))
            });
            let change = match (&before, &now) {
                (Some((was, stood)), Some((is, stands)))
                    if was.heights() == is.heights()
                        && stood.min == stands.min
                        && stood.max == stands.max =>
                {
                    None
                }
                (Some((_, stood)), Some((_, stands))) => Some(GroundChange {
                    min: stood.min.min(stands.min),
                    max: stood.max.max(stands.max),
                    ..*stands
                }),
                (Some((_, stood)), None) => Some(*stood),
                (None, Some((_, stands))) => Some(*stands),
                (None, None) => None,
            };
            self.ground_changes.extend(change);
            if let Some(now) = now {
                self.kept_ground.insert(slot, now);
            }
        }
    }

    pub(super) fn ground_changes(&self) -> &[GroundChange] {
        &self.ground_changes
    }

    pub(super) fn write(
        &mut self,
        queue: &wgpu::Queue,
        views: &TerrainViews<'_>,
        (uniforms, generation): (&wgpu::Buffer, u64),
    ) {
        let run = if self.fresh {
            0..self.records.len()
        } else {
            changed_run(&self.written, &self.records)
        };
        self.fresh = false;
        if !run.is_empty() {
            let offset = (run.start * size_of::<PatchRecord>()) as u64;
            let records = bytemuck::cast_slice(&self.records[run]);
            queue.write_buffer(&self.instances, offset, records);
        }
        if self.uniforms_generation != Some(generation) || self.uniforms != self.written_uniforms {
            queue.write_buffer(uniforms, 0, &self.uniforms);
            self.uniforms_generation = Some(generation);
        }
        let camera = ViewUniform {
            relative_view_projection: views.camera.to_cols_array_2d(),
            eye_offset: [0.0; 4],
            inverse_relative_view_projection: views
                .camera
                .as_dmat4()
                .inverse()
                .as_mat4()
                .to_cols_array_2d(),
        };
        if self.written_views != Some(camera) {
            queue.write_buffer(&self.views, 0, bytemuck::bytes_of(&camera));
            self.written_views = Some(camera);
        }
    }

    pub(super) fn write_probe_view(
        &self,
        queue: &wgpu::Queue,
        face: usize,
        relative: Mat4,
        eye: Vec3,
        center: Vec3,
    ) -> u32 {
        let view = ViewUniform {
            inverse_relative_view_projection: relative
                .as_dmat4()
                .inverse()
                .as_mat4()
                .to_cols_array_2d(),
            relative_view_projection: relative.to_cols_array_2d(),
            eye_offset: (eye.as_dvec3() - center.as_dvec3())
                .as_vec3()
                .extend(0.0)
                .to_array(),
        };
        let offset = (1 + face as u64) * SLOT;
        queue.write_buffer(&self.views, offset, bytemuck::bytes_of(&view));
        offset as u32
    }

    pub(super) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        group: &wgpu::BindGroup,
        range: &Range<u32>,
        (view, uniform): (u32, u32),
    ) -> u32 {
        if range.is_empty() {
            return 0;
        }
        pass.set_bind_group(1, group, &[view, uniform]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, range.clone());
        1
    }
}

#[derive(Clone, Copy)]
pub(super) struct TileRecord {
    key: TileKey,
    layer: u32,
    spacing: f32,
    holes: u32,
}

impl TileRecord {
    pub(super) fn of(parts: &Layout<'_>, terrain: &RenderTerrain, key: TileKey) -> Option<Self> {
        let layer = *parts.atlas.slots.get(&(terrain.id, key))?;
        let holes = (parts.tables.hole_offsets.get(&(terrain.id, key)))
            .copied()
            .unwrap_or(NO_HOLES);
        Some(Self {
            key,
            layer,
            spacing: terrain.pyramid.spacing(u32::from(key.level)),
            holes,
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
        }
    }
}

const SKIRT_SPACINGS: f32 = 4.0;

fn kept_patches(
    (records, stack): (&mut Vec<PatchRecord>, &mut Vec<(u32, u32, u32)>),
    terrain: &RenderTerrain,
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
            .map(|at| terrain.errors.get(at as usize).copied().unwrap_or(0.0))
            .sum();
        measured + pyramid.spacing(from)
    };
    let held = |key: TileKey| parts.atlas.held.get(&(terrain.id, key));
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
        let [low, high] = tile.patch_bounds(column, row);
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
            let holes = tile.patch_has_holes(column, row);
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

fn ground_box(terrain: &RenderTerrain, key: TileKey, [low, high]: [f32; 2]) -> GroundChange {
    let pyramid = terrain.pyramid;
    let level = u32::from(key.level);
    let corner = pyramid.tile_origin(key);
    let size = pyramid.tile_size(level);
    let skirt = SKIRT_SPACINGS * pyramid.spacing(level);
    GroundChange {
        min: terrain.origin + Vec3::new(corner.x, low - skirt, corner.y),
        max: terrain.origin + Vec3::new(corner.x + size, high, corner.y + size),
        pyramid,
        level,
    }
}

fn ground_rect(view: Mat4, terrain: &RenderTerrain, eye: DVec3, reach: f32) -> Option<[f32; 4]> {
    let (low, high) = terrain.tiles.iter().map(|tile| tile.bounds()).fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(low, high), [bottom, top]| {
            (
                engine_core::math::min(low, bottom),
                engine_core::math::max(high, top),
            )
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
        left = engine_core::math::min(left, x);
        bottom = engine_core::math::min(bottom, y);
        right = engine_core::math::max(right, x);
        top = engine_core::math::max(top, y);
    }
    Some([left, bottom, right, top])
}

const PATCH_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Sint32x2, 1 => Float32x2, 2 => Uint32x2, 3 => Uint32];

pub(super) fn patch_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<PatchRecord>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &PATCH_ATTRIBUTES,
    }
}

fn instance_buffer(device: &wgpu::Device, records: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/terrain-patches"),
        size: records * size_of::<PatchRecord>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn grid_indices() -> Vec<u16> {
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
    pub(super) fn of(terrain: &RenderTerrain, eye: DVec3) -> Self {
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
    index: usize,
    terrain: &RenderTerrain,
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
    let cover = match views.cover {
        None => [0.0; 4],
        Some((cover, _)) if terrain.surroundings => [0.0, 0.0, cover.density, 0.0],
        Some((cover, pixels)) => {
            let terms = tables.cover_terms.get(index).copied().unwrap_or_default();
            let law = crate::ground_cover::CoverLaw::of(terms, pixels);
            [law.span, 0.0, cover.density, 0.0]
        }
    };
    let deepest = tables
        .palette_of(index, terrain.surface.len())
        .iter()
        .filter(|entry| entry.params[3] > 0.0)
        .map(|entry| entry.relief[0])
        .fold(0.0, engine_core::math::max);
    let (first, finest) = parts.surface.of(index);
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
            tables.directory_bases.get(index).copied().unwrap_or(0),
            pyramid.levels(),
            tables.palette_bases.get(index).copied().unwrap_or(0),
            parts.sky.layer(index),
        ],
        relief: [deepest, views.settings.relief_steps as f32, finest, 0.0],
        cover,
        pixel_error: [views.settings.pixel_error, 0.0, 0.0, 0.0],
        macro_map: [parts.macro_maps.layer(index), 0, 0, 0],
        steep: {
            let [material, low, high] = terrain.steep.map_or([0.0; 3], |steep| {
                [
                    f32::from(steep.material) + 1.0,
                    steep.gradient[0],
                    steep.gradient[1],
                ]
            });
            let averaged = engine_core::terrain_data::TileShading::shades(1, terrain.surroundings);
            [material, low, high, if averaged { 1.0 } else { 0.0 }]
        },
        water: [tables.water_base, 0, 0, 0],
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
