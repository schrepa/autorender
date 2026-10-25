//! The frame's uniforms, views and kept patches: each terrain's uniform,
//! the views terrain is drawn from, and the instance records of the patches
//! each region of the shadow clipmap's tiles redrawn this frame takes of
//! the ground, over the one shared grid with skirts. What the records of a
//! tile's patches share is looked up once for each run of them, the vectors
//! laying them out are kept from frame to frame, and the records and
//! uniforms are written where they differ from the last frame's. The
//! camera's ground draws its subdivision's leaves instead (`surface`).
//!
//! Each region of the shadow clipmap's tiles a frame redraws takes the
//! ground there at a level of detail its level's texels choose, not the
//! camera's subdivision: each patch of that pyramid level whose tile is
//! resident, else of the finest coarser level whose tile is, unmorphed
//! (`kept_patches`). Where a tile's ground arrives, leaves or changes, the
//! box it stood and stands in goes to the clipmap, whose tiles redraw over
//! it (`follow_ground`).

use super::atlas::TileAtlas;
use super::tables::TerrainTables;
use super::*;
use engine_core::glam::Vec2;
use engine_core::terrain_data::{PATCH_QUADS, PATCHES_PER_TILE, Pyramid};

pub(crate) struct PatchDraws {
    indices: wgpu::Buffer,
    index_count: u32,
    instances: wgpu::Buffer,
    // This frame's records and the last frame's, which the instances hold
    // unless they were made since (`fresh`).
    records: Vec<PatchRecord>,
    written: Vec<PatchRecord>,
    fresh: bool,
    // Every terrain's uniform, a slot each; the last frame's, and the
    // generation of the buffer they went into.
    uniforms: Vec<u8>,
    written_uniforms: Vec<u8>,
    uniforms_generation: Option<u64>,
    // The camera's view, then a reflection probe's six faces, and the
    // camera's last written.
    pub(super) views: wgpu::Buffer,
    written_views: Option<ViewUniform>,
    // The patches a region's ground has yet to look at, each held tile
    // and the box its ground stands in as last followed, and the ground
    // that arrived, left or changed this frame.
    kept_stack: Vec<(u32, u32, u32)>,
    kept_ground: HashMap<(u64, TileKey), (Arc<engine_core::Tile>, GroundChange)>,
    ground_changes: Vec<GroundChange>,
    /// The eye this frame's terrain positions are relative to.
    eye: Vec3,
    pub(crate) frame: TerrainFrame,
}

/// Ground that arrived, left or changed: the world box a tile's ground
/// stood or stands in, skirts included, and its terrain's pyramid and the
/// tile's level, which say the shadow clipmap's levels that may draw it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GroundChange {
    pub(crate) min: Vec3,
    pub(crate) max: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) level: u32,
}

/// What laying out the frame reads besides the terrains: the tiles' layers
/// and holes, the tables' bases, holes and covers, the subdivisions' leaves,
/// the sky maps' and macro colors' layers, deformation's shading and each
/// terrain's virtual texture uniform fields.
pub(super) struct Layout<'a> {
    pub(super) atlas: &'a TileAtlas,
    pub(super) tables: &'a TerrainTables,
    pub(super) surface: &'a SurfaceTrees,
    pub(super) sky: &'a SkyMaps,
    pub(super) macro_maps: &'a MacroMaps,
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

    /// Starts a frame seen from `eye`.
    pub(super) fn begin(&mut self, eye: Vec3) {
        let mut terrains = std::mem::take(&mut self.frame.terrains);
        terrains.clear();
        self.frame = TerrainFrame {
            terrains,
            ..TerrainFrame::default()
        };
        self.eye = eye;
    }

    /// The eye this frame's terrain positions are relative to.
    pub(super) fn eye(&self) -> Vec3 {
        self.eye
    }

    /// The records this frame draws, which debug builds check.
    #[cfg(debug_assertions)]
    pub(super) fn records(&self) -> &[PatchRecord] {
        &self.records
    }

    /// Lays out every terrain's uniform of `terrains`, a slot each, which
    /// [`Self::write`] writes; each terrain drawing this frame; and the
    /// patches of the regions of shadow tiles `views` redraws.
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
            // Where deep materials displace it, its ground reaches their
            // depth past its tiles' bounds.
            let screen = ground_rect(views.camera, terrain, eye, uniform.relief[2]);
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

    /// Lays out the ground of each of `terrains` drawing this frame that a
    /// region of shadow tiles redrawn this frame, whose casters list as map
    /// `map`, takes within `region`: at the level of detail its texels
    /// choose, whatever the eye, counted from the anchors of the eye at
    /// `eye`, as `parts` hold it.
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

    /// Follows the tiles `atlas` holds of `terrains` where this frame's sync
    /// changed them: the box each one's ground stood and stands in where it
    /// arrived, left or changed, over which the shadow clipmap's tiles
    /// holding it redraw. A tile replaced over the same ground (its water
    /// made again) changes none.
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

    /// The ground that arrived, left or changed this frame.
    pub(super) fn ground_changes(&self) -> &[GroundChange] {
        &self.ground_changes
    }

    /// Writes the records laid out where they differ from the last frame's,
    /// every terrain's uniform into `uniforms` (of its generation) where it
    /// changed, and the camera's view.
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
}

/// What the instance records of a tile's patches share: the tile, its
/// layer, its level's spacing and its run of hole bits (`NO_HOLES`
/// without).
#[derive(Clone, Copy)]
pub(super) struct TileRecord {
    key: TileKey,
    layer: u32,
    spacing: f32,
    holes: u32,
}

impl TileRecord {
    /// What the records of `terrain`'s tile `key`'s patches share, as
    /// `parts` hold it; `None` when the tile has no layer.
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

    /// The instance record of the tile's patch `(x, z)` as the shadow
    /// clipmap's tiles draw it, with its holes where `holes`, counting its
    /// samples from `anchors`: unmorphed, whatever the eye.
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

/// Sample spacings a patch's skirt hangs below its edges.
const SKIRT_SPACINGS: f32 = 4.0;

/// Appends to `records` the records of `terrain`'s ground within the world
/// `planes` of a region of shadow tiles redrawn: its patches at pyramid
/// level `level` where their tiles are resident, else those of the finest
/// coarser level whose are, unmorphed, so what the tiles keep follows the
/// terrain's tiles alone, never the eye. Records count their samples from
/// `anchors` and take what `parts` hold; patches within the terrain's
/// cutout never draw, as its leaves do not. A patch's descendants stand
/// within its box widened by the errors measured between their levels and
/// a sample of its own; `stack` holds the patches yet to look at.
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
    // How far the ground of a level's descendants down to the finest may
    // stray from its own heights.
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
        // As drawn, skirts hanging below the box.
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

/// Whether the square of side `size` from terrain-local `corner` lies
/// within `cutout`, the ground a terrain's surroundings leave to the
/// terrain they surround.
fn cut_out(cutout: Option<(Vec2, Vec2)>, corner: Vec2, size: f32) -> bool {
    cutout.is_some_and(|(first, last)| {
        corner.x >= first.x
            && corner.y >= first.y
            && corner.x + size <= last.x
            && corner.y + size <= last.y
    })
}

/// The box `terrain`'s tile `key` stands in with heights from `low` to
/// `high`, world, its patches' skirts hanging below.
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

/// The rectangle of the screen `terrain`'s ground covers seen through
/// `view` from `eye` (normalized device coordinates, least x and y, then
/// greatest): its extent between its resident tiles' lowest and highest
/// ground, `reach` meters wider; `None` where its box reaches behind the
/// eye, whose projection is unbounded, or no tile is resident.
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

const PATCH_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Sint32x2, 1 => Float32x2, 2 => Uint32x2, 3 => Uint32];

/// Patch instances: `PatchInput`'s locations 0 to 3.
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
/// the eye, from which patches and leaves count their positions.
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

/// Terrain `index`'s uniform, `terrain`'s, for a view from `eye` as `views`
/// see it with what `parts` hold.
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
    let densest = (tables.cover_densities.get(index))
        .copied()
        .unwrap_or([0.0; 3]);
    let cover = match views.cover {
        None => [0.0; 4],
        // Surroundings grow no blades: their grassy
        // ground shows as canopy from the eye on.
        Some((cover, _)) if terrain.surroundings => [0.0, cover.density, densest[2], 0.0],
        // Elsewhere as blades retire, by their law.
        Some((cover, pixels)) => {
            let retire = crate::ground_cover::cover_retire(&cover, densest[2], pixels);
            [retire, cover.density, densest[2], 0.0]
        }
    };
    let (vt, vt_scale, vt_layout) = parts
        .textures
        .get(index)
        .copied()
        .unwrap_or(([0; 4], [0.0; 4], [0; 4]));
    // The deepest any of its materials with maps sinks and displaces the
    // ground, as the palette holds them.
    let palette = tables.palette_of(index, terrain.surface.len());
    let deepest = |part: usize| {
        palette
            .iter()
            .filter(|entry| entry.params[3] > 0.0)
            .map(|entry| entry.relief[part])
            .fold(0.0, engine_core::math::max)
    };
    let (first, finest) = parts.surface.of(index);
    TerrainUniform {
        anchor: [relative.x, relative.y, relative.z, pyramid.spacing(0)],
        eye: eye.as_vec3().extend(0.0).to_array(),
        // Its leaves, its layers' hole runs and its anchor's cell, from
        // which its leaves count their positions.
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
        // The ground's relief and displacement.
        relief: [
            deepest(0),
            views.settings.relief_steps as f32,
            deepest(3),
            finest,
        ],
        deform_grid: parts.shading.grid,
        deform: parts.shading.scale,
        cover,
        vt,
        vt_scale,
        vt_layout,
        // The ground's shading levels hand over as its tiles stream.
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
            // 1 where its coarser levels shade with level 0's averaged.
            let averaged = engine_core::terrain_data::TileShading::shades(1, terrain.surroundings);
            [material, low, high, if averaged { 1.0 } else { 0.0 }]
        },
    }
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
