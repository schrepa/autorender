//! The frame's patches: each terrain's uniform and the instance records of
//! the patches the camera, reflection probe faces and each shadow map draw,
//! culled by their bounds, over the one shared grid with skirts; and the
//! views they are drawn from.

use super::atlas::TileAtlas;
use super::tables::TerrainTables;
use super::*;

pub(crate) struct PatchDraws {
    indices: wgpu::Buffer,
    index_count: u32,
    instances: wgpu::Buffer,
    records: Vec<PatchRecord>,
    // The camera's view, then a reflection probe's six faces, and the
    // camera's last written.
    pub(super) views: wgpu::Buffer,
    written_views: Option<ViewUniform>,
    /// The eye this frame's terrain positions are relative to.
    eye: Vec3,
    pub(crate) frame: TerrainFrame,
}

/// What laying out the frame's patches reads besides the terrains: the
/// tiles' layers and fades, the tables' bases, holes and covers, the sky
/// maps' layers, the horizon maps' cascade reach, deformation's shading
/// and each terrain's virtual texture uniform fields.
pub(super) struct Layout<'a> {
    pub(super) atlas: &'a TileAtlas,
    pub(super) tables: &'a TerrainTables,
    pub(super) sky: &'a SkyMaps,
    pub(super) horizons: &'a Horizons,
    pub(super) shading: crate::deformation::DeformationShading,
    pub(super) textures: &'a [([u32; 4], [f32; 4], [u32; 4])],
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
            views: crate::buffers::uniform(device, "engine/terrain-views", VIEWS * SLOT),
            written_views: None,
            eye: Vec3::ZERO,
            frame: TerrainFrame::default(),
        }
    }

    /// Starts a frame of `terrains` seen from `eye`.
    pub(super) fn begin(&mut self, terrains: &RenderTerrains, eye: Vec3) {
        self.frame = TerrainFrame {
            shaded: terrains.view == engine_core::TerrainView::Shaded,
            ..TerrainFrame::default()
        };
        self.eye = eye;
    }

    /// The eye this frame's terrain positions are relative to.
    pub(super) fn eye(&self) -> Vec3 {
        self.eye
    }

    /// The grid's indices: its triangles, then its skirts'.
    pub(super) fn indices(&self) -> &wgpu::Buffer {
        &self.indices
    }

    /// The records this frame draws, which debug builds check.
    #[cfg(debug_assertions)]
    pub(super) fn records(&self) -> &[PatchRecord] {
        &self.records
    }

    /// Lays out this frame's patches of `terrains` for the camera,
    /// reflection probe faces and the shadow maps of `views`, writing their
    /// records and the camera's view; returns every terrain's uniform, a
    /// slot each.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn lay_out(
        &mut self,
        device: &wgpu::Device,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
        parts: &Layout<'_>,
    ) -> Vec<u8> {
        let uniform_bytes = terrains.terrains.len() as u64 * SLOT;
        let eye = views.eye.as_dvec3();
        self.records.clear();
        let mut uniforms = vec![0_u8; uniform_bytes as usize];
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            // What the terrain shows: 0 shaded, then its development views.
            let view = match terrains.view {
                engine_core::TerrainView::Shaded => 0.0,
                engine_core::TerrainView::Levels => 1.0,
                engine_core::TerrainView::Normals => 2.0,
                engine_core::TerrainView::Tiles => 3.0,
                engine_core::TerrainView::Materials => 4.0,
                engine_core::TerrainView::Sky => 5.0,
                engine_core::TerrainView::Detail => 6.0,
            };
            let directory = [
                parts
                    .tables
                    .directory_bases
                    .get(index)
                    .copied()
                    .unwrap_or(0),
                terrain.pyramid.levels(),
                parts.tables.palette_bases.get(index).copied().unwrap_or(0),
                parts.sky.layer(index),
            ];
            // The ground's relief.
            let relief = [
                views.settings.relief_distance,
                views.settings.relief_steps as f32,
                0.0,
                0.0,
            ];
            let (mut uniform, anchors) =
                terrain_uniform(terrain, eye, (view, directory, relief), parts.shading);
            uniform.cover = match views.cover {
                None => [0.0; 4],
                // Surroundings grow no blades: their grassy
                // ground shows as canopy from the eye on.
                Some((cover, _)) if terrain.surroundings => [0.0, 0.0, cover.density, 0.0],
                // Elsewhere as blades thin out, by their law.
                Some((cover, pixels)) => {
                    let blade = parts
                        .tables
                        .cover_densities
                        .get(index)
                        .copied()
                        .unwrap_or([0.0; 2])[1];
                    let reach = crate::ground_cover::tier_reach(&cover, blade, pixels);
                    [cover.radius, reach, cover.density, 0.0]
                }
            };
            if let Some(&(vt, vt_scale, vt_layout)) = parts.textures.get(index) {
                uniform.vt = vt;
                uniform.vt_scale = vt_scale;
                uniform.vt_layout = vt_layout;
            }
            let at = index * SLOT as usize;
            uniforms[at..at + size_of::<TerrainUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
            let origin = terrain.origin;
            let relative_origin = (terrain.origin.as_dvec3() - eye).as_vec3();
            let camera_planes = planes(views.culling, false);
            let start = self.records.len() as u32;
            let mut all = Vec::new();
            for patch in &terrain.patches {
                let Some(record) = record(parts, terrain, patch, anchors, views.fades) else {
                    continue;
                };
                all.push((record, patch.min, patch.max));
            }
            // Patches with holes last, each group nearest first as selected.
            all.sort_by_key(|(record, ..)| record.holes != NO_HOLES);
            let all_holes = all
                .iter()
                .filter(|(record, ..)| record.holes != NO_HOLES)
                .count() as u32;
            let mut camera_holes = 0;
            let mut screen = Some([
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ]);
            for &(record, min, max) in &all {
                if visible(&camera_planes, relative_origin + min, relative_origin + max) {
                    self.records.push(record);
                    camera_holes += u32::from(record.holes != NO_HOLES);
                    // As drawn, skirts hanging below the box.
                    let low = relative_origin + min - Vec3::Y * record.shape[3];
                    screen = screen.and_then(|rect| {
                        projected_rect(rect, views.camera, low, relative_origin + max)
                    });
                }
            }
            let camera = start..self.records.len() as u32;
            let all_start = self.records.len() as u32;
            if views.probes {
                self.records.extend(all.iter().map(|&(record, ..)| record));
            }
            let all_range = all_start..self.records.len() as u32;
            let all_holes = if views.probes { all_holes } else { 0 };
            let mut maps = Vec::with_capacity(views.maps.len());
            // Terrains casting long shadows draw into the near cascades
            // only, within the reach their horizon maps leave them.
            let reach = parts.horizons.cascade_reach(terrain.id);
            for &(map, clip) in views.maps {
                if reach.is_some() && map == crate::shadows::MOVING_MAP {
                    continue;
                }
                let sides = caster_planes(clip, reach);
                let first = self.records.len() as u32;
                for &(record, min, max) in &all {
                    if visible(&sides, origin + min, origin + max) {
                        self.records.push(record);
                    }
                }
                let range = first..self.records.len() as u32;
                self.frame.shadow_patches += range.len() as u32;
                maps.push((map, range));
            }
            self.frame.patches += camera.len() as u32;
            self.frame.terrains.push(FrameTerrain {
                uniform: (index as u64 * SLOT) as u32,
                camera,
                camera_holes,
                all: all_range,
                all_holes,
                maps,
                screen,
            });
        }
        self.frame.triangles = self.frame.patches * PATCH_TRIANGLES;
        let record_bytes = (self.records.len() * size_of::<PatchRecord>()) as u64;
        if record_bytes > self.instances.size() {
            self.instances =
                instance_buffer(device, (self.records.len() as u64).next_power_of_two());
        }
        uniforms
    }

    /// Writes the records laid out and the camera's view, after the
    /// terrains' uniforms.
    pub(super) fn write(&mut self, queue: &wgpu::Queue, views: &TerrainViews<'_>) {
        if !self.records.is_empty() {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.records));
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

    /// Writes reflection probe face `face`'s view, seen from `center` by the
    /// eye-relative projection `relative`, for the terrain whose eye is
    /// `eye`; returns its uniform offset.
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

    /// Draws `range` of this frame's patches through the terrain group
    /// `group` for the terrain whose uniform sits at `uniform`, seen by the
    /// view at `view`, with the pipeline the caller set; returns the draws.
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

    /// Whether any patch drawn this frame has holes.
    pub(super) fn has_holes(&self) -> bool {
        self.frame
            .terrains
            .iter()
            .any(|terrain| terrain.camera_holes > 0 || terrain.all_holes > 0)
    }
}

/// The instance record of `patch`, or `None` when its tile or its
/// parent has no layer.
pub(super) fn record(
    parts: &Layout<'_>,
    terrain: &RenderTerrain,
    patch: &engine_core::Patch,
    anchors: Anchors,
    fades: bool,
) -> Option<PatchRecord> {
    let atlas = parts.atlas;
    let key = patch.tile;
    let level = u32::from(key.level);
    let layer = *atlas.slots.get(&(terrain.id, key))?;
    let spacing = terrain.pyramid.spacing(level);
    let anchor = anchors.at(level);
    let first = [
        i64::from(key.x) * 256 + i64::from(patch.x) * 16 - anchor[0],
        i64::from(key.z) * 256 + i64::from(patch.z) * 16 - anchor[1],
    ];
    let (parent_layer, parent_texel) = match terrain.pyramid.parent(key) {
        Some(parent) => {
            let parent_layer = *atlas.slots.get(&(terrain.id, parent))?;
            let x = u32::from(key.x % 2) * 128 + u32::from(patch.x) * 8;
            let z = u32::from(key.z % 2) * 128 + u32::from(patch.z) * 8;
            (parent_layer, x | (z << 16))
        }
        None => (layer, 0),
    };
    let texel = (u32::from(patch.x) * 16) | ((u32::from(patch.z) * 16) << 16);
    let holes = if patch.holes {
        parts
            .tables
            .hole_offsets
            .get(&(terrain.id, key))
            .copied()
            .unwrap_or(NO_HOLES)
    } else {
        NO_HOLES
    };
    // A tile that arrived grows out of its parent.
    let fade = match atlas.arrivals.get(&(terrain.id, key)) {
        Some(&at) if fades => atlas.fade_since(at),
        _ => 0.0,
    };
    Some(PatchRecord {
        first: [first[0] as i32, first[1] as i32],
        shape: [spacing, patch.morph[0], patch.morph[1], 4.0 * spacing],
        atlas: [layer, parent_layer, texel, parent_texel],
        holes,
        fade,
    })
}

/// The normalized device rectangle `rect` (least x and y, then greatest)
/// widened to the box from `min` to `max` projected by `view`; `None` where
/// the box reaches behind the eye, whose projection is unbounded.
fn projected_rect(rect: [f32; 4], view: Mat4, min: Vec3, max: Vec3) -> Option<[f32; 4]> {
    // Corners as the box's least one plus its edges along each axis.
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

const PATCH_ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0 => Sint32x2, 1 => Float32x4, 2 => Uint32x4, 3 => Uint32, 4 => Float32];

/// Patch instances: `PatchInput`'s locations 0 to 4.
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

/// The shared grid's triangles, each cell split from its (x, z+1) corner to
/// its (x+1, z) corner as collision splits it, facing up; then the skirts,
/// facing out of the patch.
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
        // Edge z = 0 faces -z; z = last faces +z; x = 0 faces -x; x = last
        // faces +x.
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

/// Each level's anchor sample (x, z): a point on every level's grid near
/// the eye, from which patches count their samples.
#[derive(Clone, Copy)]
pub(super) struct Anchors {
    // The eye's cell on the top level's grid.
    cells: [i64; 2],
    levels: u32,
}

impl Anchors {
    /// `terrain`'s anchors for a view from `eye`.
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

    /// Level `level`'s anchor sample.
    fn at(self, level: u32) -> [i64; 2] {
        let scale = 1_i64 << (self.levels - 1 - level);
        [self.cells[0] * scale, self.cells[1] * scale]
    }
}

/// A terrain's uniform for a view from `eye`, and its levels' anchors.
fn terrain_uniform(
    terrain: &RenderTerrain,
    eye: DVec3,
    (view, directory, relief): (f32, [u32; 4], [f32; 4]),
    deformation: crate::deformation::DeformationShading,
) -> (TerrainUniform, Anchors) {
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
    let uniform = TerrainUniform {
        anchor: [relative.x, relative.y, relative.z, pyramid.spacing(0)],
        eye: eye.as_vec3().extend(0.0).to_array(),
        origin: terrain.origin.extend(view).to_array(),
        directory,
        relief,
        deform_grid: deformation.grid,
        deform: deformation.scale,
        cover: [0.0; 4],
        vt: [0; 4],
        vt_scale: [0.0; 4],
        vt_layout: [0; 4],
    };
    (uniform, anchors)
}

/// The planes of `clip`'s frustum as `ax + by + cz + d >= 0` inside: its
/// sides, and unless `sides_only` its near and far planes (depth 0 to 1).
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

/// The planes of map `clip` where terrain casts into it: its sides and,
/// with a `reach`, its receivers' depths widened that far toward the sun.
fn caster_planes(clip: Mat4, reach: Option<f32>) -> Vec<Vec4> {
    let mut planes = planes(clip, true);
    if let Some(reach) = reach {
        let (depth, w) = (clip.row(2), clip.row(3));
        // Depth changes by the third row's length per world unit.
        let per_unit = depth.truncate().length();
        planes.push(w - depth);
        planes.push(depth + w * (reach * per_unit));
    }
    planes
}

/// Whether the box from `min` to `max` reaches inside every plane.
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
