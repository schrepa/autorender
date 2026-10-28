#![expect(
    clippy::expect_used,
    reason = "the pipelines are created before they draw"
)]

mod casters;
mod maps;
mod marks;
mod pipelines;
mod regions;
mod selection;
mod sky;

use std::cell::Cell;

use engine_core::{MaterialHandle, MeshHandle, RenderFrame, RenderSnapshot, glam::Vec3};
use engine_terrain::Pyramid;

use crate::batches::BatchPlan;
use crate::bindings::SHADOW_POSES;
use crate::cached::{Bound, GroupCache};
use crate::gpu_data::{ComposeData, DRAW_ARGS_BYTES, RouteData, ShadowCullUniform};
use crate::mesh_arena::{Bindings, Ids, MeshDraw};
use crate::profiling::GpuProfiler;
use crate::renderer::FramePoses;
use crate::shadow_fit::{ClipPlan, ClipSettings, ClipStats, Clipmap, LightBasis, ground_level};

use crate::gpu_layouts::constants::MOVING_MAP;
use crate::gpu_layouts::constants::SHADOW_GROUP as GROUP;
use crate::gpu_layouts::constants::SHADOW_REGIONS;
use crate::gpu_layouts::constants::SKY_MAP;
use crate::gpu_layouts::constants::SKY_REFRESH;
use casters::CasterGroups;
pub(crate) use casters::ShadowGroup;
pub(crate) use maps::SHADOW_FORMAT;
use maps::{ShadowMaps, map_pass};
use marks::Marks;
use pipelines::ShadowPipelines;
use regions::{FrameRegions, SelectionCounts};
use selection::{Selection, map_factor, map_start};
use sky::{SkyDraw, SkyMaps};

const MAPS: usize = SHADOW_REGIONS + 2;

const VIEW_STRIDE: u64 = 256;
const MOVING_VIEW: u64 = SHADOW_REGIONS as u64 * VIEW_STRIDE;
const COMPOSE_VIEW: u64 = MOVING_VIEW + VIEW_STRIDE;
const SKY_VIEW: u64 = COMPOSE_VIEW + VIEW_STRIDE;
const VIEWS_BYTES: u64 = SKY_VIEW + VIEW_STRIDE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    Presented,
    Listed,
}

pub(crate) struct Presented<'a> {
    pub candidates: &'a wgpu::Buffer,
    pub batches: &'a wgpu::Buffer,
    pub flags: &'a wgpu::Buffer,
    pub count: u32,
    pub generation: u64,
}

pub(crate) struct CasterDraw {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub lods: Vec<(u32, u32, f32)>,
}

impl CasterDraw {
    fn level_within(&self, error: f32) -> u8 {
        self.lods
            .iter()
            .take_while(|&&(_, _, lod_error)| lod_error <= error)
            .count() as u8
    }

    fn range(&self, level: u8) -> (u32, u32, i32) {
        match level.checked_sub(1) {
            Some(lod) => {
                let (first_index, index_count, _) = self.lods[lod as usize];
                (index_count, first_index, self.base_vertex)
            }
            None => (self.index_count, self.first_index, self.base_vertex),
        }
    }
}

pub(crate) struct ShadowInputs<'a> {
    pub snapshot: &'a RenderSnapshot,
    pub presented: Option<Presented<'a>>,
    pub poses: FramePoses<'a>,
    pub slot_capacity: usize,
    pub mesh_layout: u64,
    pub placement: &'a dyn Fn(MeshHandle) -> CasterDraw,
    pub terrain: bool,
    pub pixel_angle: f32,
}

pub(crate) type ScatterDraw<'a> = dyn Fn(
        &mut wgpu::RenderPass<'_>,
        usize,
        &[wgpu::RenderPipeline; 2],
        Option<&wgpu::RenderPipeline>,
    ) -> u32
    + 'a;

#[derive(Clone, Copy, Debug)]
pub(crate) struct KeptRefresh {
    pub planes: [[f32; 4]; 6],
    pub texel: f32,
}

pub(crate) struct MapSources<'a> {
    pub meshes: &'a dyn Fn(MeshHandle) -> Option<MeshDraw<'a>>,
    pub materials: &'a dyn Fn(MaterialHandle) -> Option<&'a wgpu::BindGroup>,
    pub terrain_draw: &'a dyn Fn(&mut wgpu::RenderPass<'_>, usize) -> u32,
    pub scatter_draw: &'a ScatterDraw<'a>,
}

#[derive(Clone, Copy)]
enum Casters {
    Terrain,
    Scatter,
    Meshes,
}

const CASTERS: [Casters; 3] = [Casters::Terrain, Casters::Scatter, Casters::Meshes];

struct Caster<'m> {
    choice: usize,
    draw: MeshDraw<'m>,
    material: Option<&'m wgpu::BindGroup>,
    index: usize,
}

fn recycle<T, U>(mut storage: Vec<T>) -> Vec<U> {
    storage.clear();
    storage.into_iter().filter_map(|_| None).collect()
}

pub(crate) struct ShadowFrame {
    candidates: u32,
    regions: u32,
    composes: u32,
    moving: u32,
    sky: Option<SkyDraw>,
    pose_index: usize,
    tiles: ClipStats,
}

impl ShadowFrame {
    pub(crate) fn tiles(&self) -> ClipStats {
        self.tiles
    }
}

pub(crate) struct ShadowPass {
    pipelines: Option<ShadowPipelines>,
    material_layout: crate::pipelines::GroupLayout,
    maps: ShadowMaps,
    horizon_records: f32,
    clipmap: Clipmap,
    plan: ClipPlan,
    region_views: Vec<(usize, KeptRefresh)>,
    casters: CasterGroups,
    selection: Selection,
    marks: Marks,
    pose_groups: [GroupCache<3>; 2],
    compose_groups: [GroupCache<3>; 2],
    groups_generation: u64,
    caster_draws: Vec<CasterDraw>,
    draws_key: Option<(u64, u64, bool)>,
    written_views: Option<Vec<f32>>,
    written_cull: Option<ShadowCullUniform>,
    written_routes: Option<Box<RouteData>>,
    composes: Vec<ComposeData>,
    compose_buffer: crate::buffers::Growable,
    caster_storage: Cell<Vec<Caster<'static>>>,
    sky: SkyMaps,
    sky_groups: u64,
    written_sky_view: Option<engine_core::glam::Mat4>,
}

impl ShadowPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        material_layout: &crate::pipelines::GroupLayout,
    ) -> Self {
        Self {
            pipelines: None,
            material_layout: material_layout.clone(),
            maps: ShadowMaps::new(device),
            horizon_records: 0.0,
            clipmap: Clipmap::default(),
            plan: ClipPlan::default(),
            region_views: Vec::new(),
            casters: CasterGroups::new(),
            selection: selection::Selection::new(device),
            marks: Marks::new(device),
            pose_groups: Default::default(),
            compose_groups: Default::default(),
            groups_generation: 0,
            caster_draws: Vec::new(),
            draws_key: None,
            written_views: None,
            written_cull: None,
            written_routes: None,
            composes: Vec::new(),
            compose_buffer: crate::buffers::Growable::new(
                device,
                "engine/shadow-composes",
                size_of::<ComposeData>() as u64,
                wgpu::BufferUsages::VERTEX,
            ),
            caster_storage: Cell::default(),
            sky: SkyMaps::new(device),
            sky_groups: 0,
            written_sky_view: None,
        }
    }

    pub(crate) fn sky_bindings(&self) -> [(&'static str, crate::cached::Bound<'_>); 3] {
        self.sky.bindings()
    }

    pub(crate) fn sky_counts(&self) -> (u32, u32) {
        self.sky.counts()
    }

    pub(crate) fn groups_generation(&self) -> u64 {
        self.groups_generation
    }

    pub(crate) fn entries(&self) -> [(&'static str, crate::cached::Bound<'_>); 5] {
        self.maps.entries()
    }

    pub(crate) fn groups(&self) -> &[ShadowGroup] {
        self.casters.groups()
    }

    pub(crate) fn batch_group(&self, index: usize) -> u32 {
        self.casters.batch_group(index)
    }

    pub(crate) fn region_views(&self) -> &[(usize, KeptRefresh)] {
        &self.region_views
    }

    pub(crate) fn dirty_kept(&mut self, min: Vec3, max: Vec3) {
        self.clipmap.dirty_box(min, max);
    }

    pub(crate) fn dirty_kept_all(&mut self) {
        self.clipmap.dirty_all();
    }

    pub(crate) fn dirty_ground(
        &mut self,
        (min, max): (Vec3, Vec3),
        (pyramid, level): (&Pyramid, u32),
    ) {
        for index in 0..self.clipmap.len() {
            if level >= ground_level(pyramid, self.clipmap.texel(index)) {
                self.clipmap.dirty_level_box(index, min, max);
            }
        }
    }

    pub(crate) fn caster_counts(&self) -> Option<[u32; 3]> {
        self.selection.caster_counts()
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.capacity_bytes() + self.selection.capacity_bytes() + self.marks.capacity_bytes()
    }

    pub(crate) fn plan_groups(&mut self, plan: &BatchPlan, frame: RenderFrame<'_>, gpu: bool) {
        if !frame.snapshot().environment.shadows.enabled {
            if !self.casters.groups().is_empty() {
                self.groups_generation += 1;
            }
            self.casters.clear();
            return;
        }
        if self.casters.plan(plan, frame, gpu) {
            self.groups_generation += 1;
            self.selection.groups_changed();
            self.clipmap.dirty_all();
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &ShadowInputs<'_>,
    ) -> Option<ShadowFrame> {
        let ShadowInputs {
            snapshot,
            presented,
            poses,
            ..
        } = inputs;
        self.region_views.clear();
        let settings = snapshot.environment.shadows.sanitized();
        if !settings.enabled || (self.casters.groups().is_empty() && !inputs.terrain) {
            if !settings.enabled {
                self.maps.release();
            }
            self.sky.release(queue);
            self.maps.write_uniform(queue, bytemuck::Zeroable::zeroed());
            return None;
        }
        let source = self.casters.source();
        let classify = source == Source::Presented;
        self.pipelines
            .get_or_insert_with(|| ShadowPipelines::new(device, &self.material_layout));
        let limit = device.limits().max_texture_dimension_2d;
        let (layout, made) = self.maps.ensure(device, &settings, limit);
        let basis = LightBasis::new(snapshot.environment.atmosphere.sanitized().sun_direction);
        let restarted = self.clipmap.place(
            basis,
            snapshot.detail_eye,
            ClipSettings {
                levels: settings.levels,
                finest_texel: settings.finest_texel,
                tile_texels: settings.tile_texels,
                slots: layout.still,
            },
        );
        if made && !restarted {
            self.clipmap.invalidate();
        }
        let epoch = self.clipmap.epoch();
        let marked = self.marks.take(epoch);
        let (dirty, moving) = self.selection.take_masks(epoch);
        let still_changed = dirty.iter().any(|&word| word != 0);
        self.clipmap.merge_marks(&marked);
        self.clipmap.merge_dirty(&dirty);
        self.clipmap.merge_moving(&moving);
        let mask_words = self.clipmap.mask_words();
        self.marks.ensure(device, mask_words);
        if classify
            && self
                .selection
                .ensure_stillness(device, inputs.slot_capacity)
        {
            self.clipmap.dirty_all();
        }
        if self.selection.ensure_outputs(
            device,
            (self.casters.groups().len(), self.casters.total()),
            mask_words,
        ) {
            self.clipmap.dirty_all();
        }
        let moving_budget = if self.casters.groups().is_empty() {
            0
        } else {
            layout.moving
        };
        self.clipmap.plan(
            (
                settings.redraw_budget,
                crate::shadow_fit::region_tiles(settings.tile_texels),
            ),
            moving_budget,
            &mut self.plan,
        );
        let frame_regions = FrameRegions {
            basis,
            clipmap: &self.clipmap,
            plan: &self.plan,
            layout,
        };
        frame_regions.region_views(&mut self.region_views);
        let draws_key = (
            self.groups_generation,
            inputs.mesh_layout,
            snapshot.environment.level_of_detail.enabled,
        );
        if self.draws_key != Some(draws_key) {
            self.draws_key = Some(draws_key);
            self.caster_draws = self
                .casters
                .groups()
                .iter()
                .map(|group| (inputs.placement)(group.mesh))
                .collect();
        }
        let draws = &self.caster_draws;
        let levels: Vec<u8> = frame_regions
            .errors()
            .iter()
            .flat_map(|&error| draws.iter().map(move |draw| draw.level_within(error)))
            .collect();
        self.selection.update_template(
            device,
            queue,
            &self.casters,
            levels,
            |index, level| draws[index].range(level),
            inputs.mesh_layout,
        );
        let pipelines = self.pipelines.as_ref().expect("created above");
        let wind = engine_terrain::TerrainEnvironment::of(snapshot).wind;
        let gusting = [
            wind.velocity[0],
            wind.velocity[1],
            wind.gusts,
            snapshot.elapsed_seconds,
        ];
        let viewer = snapshot.detail_eye.extend(1.0 / inputs.pixel_angle);
        let views = frame_regions.view_matrices((wind.velocity, gusting), viewer);
        if self.written_views.as_deref() != Some(&views[..]) {
            queue.write_buffer(&pipelines.view_buffer, 0, bytemuck::cast_slice(&views));
            self.written_views = Some(views);
        }
        let routes = frame_regions.routes();
        if self
            .written_routes
            .as_deref()
            .is_none_or(|written| *written != routes)
        {
            queue.write_buffer(&pipelines.routes, 0, bytemuck::bytes_of(&routes));
            self.written_routes = Some(Box::new(routes));
        }
        frame_regions.composes(settings.tile_texels, &mut self.composes);
        if !self.composes.is_empty() {
            let bytes: &[u8] = bytemuck::cast_slice(&self.composes);
            self.compose_buffer.grow(device, bytes.len() as u64);
            queue.write_buffer(&self.compose_buffer, 0, bytes);
        }
        let candidates = match (presented, source) {
            (Some(presented), Source::Presented) => presented.count,
            (_, Source::Listed) => self.casters.listed.len() as u32,
            _ => 0,
        };
        let mut cull = frame_regions.selection(
            &SelectionCounts {
                candidates,
                groups: self.casters.groups().len() as u32,
                capacity: self.selection.id_capacity as u32,
                classify,
            },
            mask_words as u32,
        );
        let lookup = frame_regions.lookup_uniform(
            settings.tile_texels,
            snapshot.environment.foliage.sanitized().crown_depth,
        );
        let groups_changed = self.sky_groups != self.groups_generation;
        self.sky_groups = self.groups_generation;
        let sky = self.sky.prepare(
            device,
            queue,
            &snapshot.environment.sky_occlusion.sanitized(),
            snapshot.detail_eye,
            still_changed || groups_changed,
        );
        let pipelines = self.pipelines.as_ref().expect("created above");
        if let Some(draw) = sky {
            cull.sky = draw.rows;
            cull.sky_rect = draw.rect;
            cull.sky_depths = [draw.depth, 0.0, 0.0, 0.0];
            cull.mode |= SKY_REFRESH;
            if self.written_sky_view != Some(draw.clip) {
                let mut view = [0.0_f32; 20];
                view[..16].copy_from_slice(&draw.clip.to_cols_array());
                view[16..].copy_from_slice(&gusting);
                queue.write_buffer(
                    &pipelines.view_buffer,
                    SKY_VIEW,
                    bytemuck::cast_slice(&view),
                );
                self.written_sky_view = Some(draw.clip);
            }
        }
        if self
            .written_cull
            .is_none_or(|written| bytemuck::bytes_of(&written) != bytemuck::bytes_of(&cull))
        {
            queue.write_buffer(&pipelines.cull, 0, bytemuck::bytes_of(&cull));
            self.written_cull = Some(cull);
        }
        let mut lookup = lookup;
        lookup.level_state[3] = self.horizon_records;
        self.maps.write_uniform(queue, lookup);
        let (table, changed) = self.clipmap.take_table();
        self.maps.write_table(queue, table, &changed);
        self.selection.bind(
            device,
            pipelines,
            source,
            presented.as_ref(),
            poses,
            self.maps.table(),
        );
        self.bind_poses(device, poses);
        Some(ShadowFrame {
            candidates,
            regions: self.plan.regions.len() as u32,
            composes: self.plan.composes.len() as u32,
            moving: self.plan.moving.len() as u32,
            sky,
            pose_index: poses.parity,
            tiles: self.clipmap.stats(),
        })
    }

    fn bind_poses(&mut self, device: &wgpu::Device, poses: &FramePoses<'_>) {
        let Some(pipelines) = &self.pipelines else {
            return;
        };
        let groups = self.pose_groups.iter_mut().zip(&mut self.compose_groups);
        for (parity, (group, compose)) in groups.enumerate() {
            let Some(buffer) = poses.bound(parity) else {
                group.clear();
                compose.clear();
                continue;
            };
            let entries = |regions| {
                [
                    ("slot_poses", buffer.clone()),
                    ("slot_colors", poses.colors.bound()),
                    ("shadow_regions", regions),
                ]
            };
            let layout = (&SHADOW_POSES, &pipelines.pose_layout);
            group.bind(
                device,
                "engine/shadow-pose-group",
                layout,
                entries(self.maps.regions_placeholder()),
            );
            match self.maps.regions_bound() {
                Some(regions) => {
                    compose.bind(
                        device,
                        "engine/shadow-compose-group",
                        layout,
                        entries(regions),
                    );
                }
                None => compose.clear(),
            }
        }
    }

    pub(crate) fn encode_selection(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &ShadowFrame,
        profiler: &GpuProfiler,
    ) {
        let (Some(pipelines), Some(_)) = (&self.pipelines, self.maps.textures()) else {
            return;
        };
        if self.casters.groups().is_empty() {
            return;
        }
        let epoch = self.clipmap.epoch();
        self.selection.encode(
            device,
            encoder,
            pipelines,
            (frame, self.casters.groups().len()),
            epoch,
            profiler,
        );
    }

    pub(crate) fn encode_marks(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (camera, depth): (&wgpu::Buffer, Bound<'_>),
        size: (u32, u32),
        profiler: &GpuProfiler,
    ) {
        let (Some(pipelines), Some(_)) = (&self.pipelines, self.maps.textures()) else {
            return;
        };
        let epoch = self.clipmap.epoch();
        self.marks.encode(
            device,
            encoder,
            pipelines,
            (camera, self.maps.uniform()),
            depth,
            size,
            epoch,
            profiler,
        );
    }

    pub(crate) fn prepare_impostor_casters(
        &mut self,
        device: &wgpu::Device,
        impostor_layout: &crate::pipelines::GroupLayout,
    ) {
        if let Some(pipelines) = &mut self.pipelines
            && pipelines.impostor_casters.is_none()
        {
            pipelines.impostor_casters = Some(pipelines.impostor_caster(device, impostor_layout));
        }
    }

    pub(crate) fn set_horizon_records(&mut self, queue: &wgpu::Queue, count: u32) {
        self.horizon_records = count as f32;
        self.maps.write_horizon_records(queue, self.horizon_records);
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn encode_maps(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &ShadowFrame,
        sources: &MapSources<'_>,
        profiler: &GpuProfiler,
    ) -> u32 {
        let (Some(pipelines), Some(maps)) = (&self.pipelines, self.maps.textures()) else {
            return 0;
        };
        let MapSources {
            meshes,
            materials,
            terrain_draw,
            scatter_draw,
        } = sources;
        let draws = Cell::new(0_u32);
        let groups = self.casters.groups();
        let group_count = groups.len() as u64;
        let (ids, args, id_capacity) = (
            &self.selection.ids,
            &self.selection.args,
            self.selection.id_capacity,
        );
        let pose_group = self.pose_groups[frame.pose_index].get();
        let compose_group = self.compose_groups[frame.pose_index].get();
        let mut casters: Vec<Caster<'_>> = recycle(self.caster_storage.take());
        if pose_group.is_some() {
            casters.extend(groups.iter().enumerate().filter_map(|(index, group)| {
                let draw = meshes(group.mesh)?;
                let material = match group.material.map(materials) {
                    Some(None) => return None,
                    Some(Some(material)) => Some(material),
                    None => None,
                };
                Some(Caster {
                    choice: usize::from(material.is_some()),
                    draw,
                    material,
                    index,
                })
            }));
        }
        casters.sort_unstable_by_key(|caster| {
            (
                caster.choice,
                std::ptr::from_ref(caster.draw.vertices) as usize,
                caster
                    .material
                    .map_or(0, |group| std::ptr::from_ref(group) as usize),
                caster.index,
            )
        });
        let draw_meshes =
            |pass: &mut wgpu::RenderPass<'_>, map: usize, set: &[wgpu::RenderPipeline; 2]| {
                let Some(pose_group) = pose_group else {
                    return;
                };
                pass.set_pipeline(&set[0]);
                pass.set_bind_group(1, pose_group, &[]);
                let mut bound = 0;
                let mut bindings = Bindings::default();
                let start = map_start(map, id_capacity);
                for caster in &casters {
                    if caster.choice != bound {
                        pass.set_pipeline(&set[caster.choice]);
                        bound = caster.choice;
                    }
                    if let Some(material) = caster.material {
                        bindings.group(pass, 2, material);
                    }
                    bindings.vertex(pass, 0, caster.draw.vertices);
                    bindings.index(pass, caster.draw.indices);
                    let caster_ids = if self.selection.regions_in_args {
                        Ids::Whole(ids)
                    } else {
                        let first =
                            start + map_factor(map) as usize * groups[caster.index].start as usize;
                        Ids::Region(ids.slice(first as u64 * 4..))
                    };
                    bindings.ids(pass, 1, caster_ids);
                    pass.draw_indexed_indirect(
                        args,
                        (map as u64 * group_count + caster.index as u64) * DRAW_ARGS_BYTES,
                    );
                    draws.set(draws.get() + 1);
                }
            };
        let draw_selected = |pass: &mut wgpu::RenderPass<'_>, map: usize| {
            for casters in CASTERS {
                match casters {
                    Casters::Terrain => draws.set(draws.get() + terrain_draw(pass, map)),
                    Casters::Scatter => draws.set(
                        draws.get()
                            + scatter_draw(
                                pass,
                                map,
                                &pipelines.casters,
                                pipelines.impostor_casters.as_ref(),
                            ),
                    ),
                    Casters::Meshes => draw_meshes(pass, map, &pipelines.casters),
                }
            }
        };
        let last_region = (frame.regions as usize).saturating_sub(1);
        let writes = |span: &'static str, begin: bool, end: bool| {
            (begin || end)
                .then(|| profiler.span_in("shadow_maps", span))
                .flatten()
                .map(|span| span.render_part(begin, end))
        };
        for (region, layer) in maps.region_layers.iter().enumerate() {
            if region >= frame.regions as usize {
                break;
            }
            let timestamps = writes("shadow_regions", region == 0, region == last_region);
            let mut pass = map_pass(encoder, layer, wgpu::LoadOp::Clear(1.0), timestamps);
            pass.set_bind_group(
                0,
                &pipelines.view_group,
                &[(region as u64 * VIEW_STRIDE) as u32],
            );
            draw_selected(&mut pass, region);
        }
        let composing = frame.composes > 0 && compose_group.is_some();
        let moving = frame.moving > 0 && pose_group.is_some();
        if composing || moving {
            let mut pass = map_pass(
                encoder,
                &maps.atlas_view,
                wgpu::LoadOp::Load,
                writes("shadow_atlas", true, true),
            );
            if let (true, Some(compose)) = (composing, compose_group) {
                pass.set_pipeline(&pipelines.compose);
                pass.set_bind_group(0, &pipelines.view_group, &[COMPOSE_VIEW as u32]);
                pass.set_bind_group(1, compose, &[]);
                pass.set_vertex_buffer(0, self.compose_buffer.slice(..));
                pass.draw(0..6, 0..frame.composes);
            }
            if let (true, Some(pose_group)) = (moving, pose_group) {
                pass.set_pipeline(&pipelines.clear_moving);
                pass.set_bind_group(0, &pipelines.view_group, &[MOVING_VIEW as u32]);
                pass.set_bind_group(1, pose_group, &[]);
                pass.draw(0..6, 0..frame.moving);
                draw_meshes(&mut pass, MOVING_MAP, &pipelines.moving);
            }
        }
        if let Some(view) = frame.sky.and_then(|draw| self.sky.layer_view(draw.layer)) {
            let mut pass = map_pass(
                encoder,
                view,
                wgpu::LoadOp::Clear(1.0),
                writes("shadow_sky", true, true),
            );
            pass.set_bind_group(0, &pipelines.view_group, &[SKY_VIEW as u32]);
            draw_meshes(&mut pass, SKY_MAP, &pipelines.casters);
        }
        self.caster_storage.set(recycle(casters));
        draws.get()
    }

    pub(crate) fn after_submit(&mut self) {
        self.selection.after_submit();
        self.marks.after_submit();
    }
}
