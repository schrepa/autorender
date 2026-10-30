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

use engine_core::{
    MeshHandle, Pyramid, RenderFrame, RenderSnapshot, RenderStats, ShadowStats, glam::Vec3,
};
use engine_faces::heightfield::HeightfieldFace;
use engine_faces::placed::PlacedFace;
use engine_faces::shadows::{KeptRefresh, SHADOW_POSES, ShadowCastersFace, ground_level};
use engine_render::batches::BatchPlan;
use engine_render::bindings::{
    Contribution, Sample, Sampling, Stages, sampler, texture, uniform_of,
};
use engine_render::cached::{Bound, GroupCache};
use engine_render::gpu_data::DRAW_ARGS_BYTES;
use engine_render::gpu_presentation::Presented;
use engine_render::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
};
use engine_render::mesh_arena::{Bindings, Ids, MeshDraw};
use engine_render::parts::{
    Begin, BeginOut, Create, DrawAt, Encode, GpuPart, PartAssets, PartDecl, PartEntry, Prelude,
    Shading, Step, View,
};
use engine_render::profiling::GpuProfiler;
use engine_render::renderer::{CasterDraw, FramePoses};

use crate::gpu_layouts::{ComposeData, RouteData, ShadowCullUniform, ShadowUniform, SkyUniform};
use crate::shadow_fit::{ClipPlan, ClipSettings, ClipStats, Clipmap, LightBasis};

use crate::gpu_layouts::MOVING_MAP;
use crate::gpu_layouts::SHADOW_GROUP as GROUP;
use crate::gpu_layouts::SHADOW_REGIONS;
use crate::gpu_layouts::SKY_MAP;
use crate::gpu_layouts::SKY_REFRESH;
use casters::CasterGroups;
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

pub(crate) const FRAME_SHADOWS: Contribution = Contribution {
    group: "frame",
    set: "shadows",
    bindings: &[
        uniform_of(
            "shadow",
            size_of::<ShadowUniform>(),
            Stages::VERTEX_FRAGMENT,
        ),
        texture(
            "shadow_atlas",
            Sample::Depth,
            engine_render::bindings::View::D2,
            Stages::VERTEX_FRAGMENT,
        ),
        texture(
            "shadow_tiles",
            Sample::Uint,
            engine_render::bindings::View::D2,
            Stages::VERTEX_FRAGMENT,
        ),
        sampler("shadow_sampler", Sampling::Comparison, Stages::FRAGMENT),
        sampler(
            "shadow_texels",
            Sampling::NonFiltering,
            Stages::VERTEX_FRAGMENT,
        ),
    ],
};

pub(crate) const AMBIENT_SKY_OCCLUSION: Contribution = Contribution {
    group: "ambient",
    set: "sky_occlusion",
    bindings: &[
        texture(
            "sky_maps",
            Sample::Depth,
            engine_render::bindings::View::D2Array,
            Stages::FRAGMENT,
        ),
        sampler("sky_sampler", Sampling::Comparison, Stages::FRAGMENT),
        uniform_of("sky", size_of::<SkyUniform>(), Stages::FRAGMENT),
    ],
};

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "shadows",
        brick: "shadows",
        passes: &[
            PassDecl {
                name: "shadow_select",
                stage: Stage::Prepare,
            },
            PassDecl {
                name: "shadow_maps",
                stage: Stage::Prepare,
            },
            PassDecl {
                name: "shadow_marks",
                stage: Stage::AfterOpaque,
            },
        ],
        resources: &[
            ResourceDecl {
                name: "casters",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "shadow_maps",
                lifetime: Lifetime::Kept,
            },
            ResourceDecl {
                name: "shadow_marks",
                lifetime: Lifetime::Kept,
            },
        ],
        hooks: &[],
        contributes: &[FRAME_SHADOWS, AMBIENT_SKY_OCCLUSION],
        effects: &[],
        labels: &[],
        lazy: false,
        faces: &[engine_render::offer_face!(ShadowPass => dyn ShadowCastersFace)],
        needs: &[],
        piece: None,
    },
    create,
};

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(ShadowPass::new(&cx.core.device))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    Presented,
    Listed,
}

struct ShadowInputs<'a> {
    snapshot: &'a RenderSnapshot,
    presented: Option<Presented<'a>>,
    poses: FramePoses<'a>,
    slot_capacity: usize,
    mesh_layout: u64,
    placement: &'a dyn Fn(MeshHandle) -> CasterDraw,
    terrain: bool,
    pixel_angle: f32,
}

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

#[derive(Clone, Copy, Default)]
struct GraphIds {
    select: Option<PassId>,
    maps: Option<PassId>,
    marks: Option<PassId>,
    poses: Option<ResourceId>,
    visible: Option<ResourceId>,
    casters: Option<ResourceId>,
    placed: Option<ResourceId>,
    leaves: Option<ResourceId>,
    shadow_maps: Option<ResourceId>,
    depth: Option<ResourceId>,
    shadow_marks: Option<ResourceId>,
}

pub(crate) struct ShadowPass {
    pipelines: Option<ShadowPipelines>,
    maps: ShadowMaps,
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
    compose_buffer: engine_render::buffers::Growable,
    caster_storage: Cell<Vec<Caster<'static>>>,
    sky: SkyMaps,
    sky_groups: u64,
    written_sky_view: Option<engine_core::glam::Mat4>,
    frame: Option<ShadowFrame>,
    draws: u32,
    ids: GraphIds,
}

impl ShadowPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let _owner = engine_render::memory::owned("shadows");
        Self {
            pipelines: None,
            maps: ShadowMaps::new(device),
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
            compose_buffer: engine_render::buffers::Growable::new(
                device,
                "engine/shadow-composes",
                size_of::<ComposeData>() as u64,
                wgpu::BufferUsages::VERTEX,
            ),
            caster_storage: Cell::default(),
            sky: SkyMaps::new(device),
            sky_groups: 0,
            written_sky_view: None,
            frame: None,
            draws: 0,
            ids: GraphIds::default(),
        }
    }

    fn sky_counts(&self) -> (u32, u32) {
        self.sky.counts()
    }

    fn dirty_ground(&mut self, (min, max): (Vec3, Vec3), (pyramid, level): (&Pyramid, u32)) {
        for index in 0..self.clipmap.len() {
            if level >= ground_level(pyramid, self.clipmap.texel(index)) {
                self.clipmap.dirty_level_box(index, min, max);
            }
        }
    }

    fn caster_counts(&self) -> Option<[u32; 3]> {
        self.selection.caster_counts()
    }

    fn plan_groups(&mut self, plan: &BatchPlan, frame: RenderFrame<'_>, gpu: bool) {
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
    fn prepare_maps(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
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
            .get_or_insert_with(|| ShadowPipelines::new(device));
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
        let pipelines = self.pipelines.as_ref().expect("pipelines_created");
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
        let pipelines = self.pipelines.as_ref().expect("pipelines_created");
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

    fn encode_selection(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &GpuProfiler,
    ) {
        let (Some(frame), Some(pipelines), Some(_)) =
            (&self.frame, &self.pipelines, self.maps.textures())
        else {
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

    fn encode_marks(&mut self, cx: &Encode<'_>, encoder: &mut wgpu::CommandEncoder) {
        let (Some(pipelines), Some(_)) = (&self.pipelines, self.maps.textures()) else {
            return;
        };
        let core = &*cx.core;
        let epoch = self.clipmap.epoch();
        self.marks.encode(
            &core.device,
            encoder,
            pipelines,
            (&core.camera.buffer, self.maps.uniform()),
            core.depth.view.bound(),
            core.render_pixels(),
            epoch,
            &core.profiler,
        );
    }

    fn prepare_impostor_casters(&mut self, device: &wgpu::Device) {
        if let Some(pipelines) = &mut self.pipelines
            && pipelines.impostor_casters.is_none()
        {
            pipelines.impostor_casters = Some(pipelines.impostor_caster(device));
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn encode_maps(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &ShadowFrame,
        cx: &Encode<'_>,
    ) -> u32 {
        let (Some(pipelines), Some(maps)) = (&self.pipelines, self.maps.textures()) else {
            return 0;
        };
        let core = &*cx.core;
        let profiler = &core.profiler;
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
                let draw = core.mesh_draw(group.mesh)?;
                let material = match group.material.map(|material| core.material_group(material)) {
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
            let at = DrawAt::ShadowCasters {
                map,
                casters: &pipelines.casters,
                cards: pipelines.impostor_casters.as_ref(),
            };
            draws.set(draws.get() + cx.draw_hook(at, pass));
            draw_meshes(pass, map, &pipelines.casters);
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
}

impl GpuPart for ShadowPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = GraphIds {
            select: table.pass_named("shadow_select"),
            maps: table.pass_named("shadow_maps"),
            marks: table.pass_named("shadow_marks"),
            poses: table.resource("poses"),
            visible: table.resource("visible"),
            casters: table.resource("casters"),
            placed: table.resource("placed"),
            leaves: table.resource("leaves"),
            shadow_maps: table.resource("shadow_maps"),
            depth: table.resource("depth"),
            shadow_marks: table.resource("shadow_marks"),
        };
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        self.plan_groups(&cx.core.plan, cx.frame, cx.gpu_mode);
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        if cx
            .parts
            .face::<dyn PlacedFace>()
            .is_some_and(|placed| placed.restyled())
        {
            self.clipmap.dirty_all();
        }
        self.draws = 0;
        let snapshot = cx.frame.snapshot;
        let gpu_mode = cx.frame.kind.gpu_mode;
        let parity = cx.frame.temporal.as_ref().map_or(0, |frame| frame.parity);
        let core = &*cx.core;
        let levels = snapshot.environment.level_of_detail.enabled;
        let placement = |mesh| core.caster_draw(mesh, levels);
        self.frame = self.prepare_maps(
            &core.device,
            &core.queue,
            &ShadowInputs {
                snapshot,
                presented: gpu_mode.then(|| core.gpu_presentation.presented()),
                poses: core.frame_poses(parity),
                slot_capacity: core.instances.capacity,
                mesh_layout: core.mesh_layout,
                placement: &placement,
                terrain: engine_terrain::RenderTerrains::of(snapshot)
                    .terrains
                    .iter()
                    .any(|terrain| terrain.drawn),
                pixel_angle: engine_render::gpu_data::pixel_angle(
                    snapshot.view_projection,
                    core.display.region.display.height,
                ),
            },
        );
    }

    fn prepare_draws(&mut self, cx: &mut Step<'_>, _encoder: &mut wgpu::CommandEncoder) {
        let placed = cx.parts.face::<dyn PlacedFace>();
        if cx.frame.kind.gpu_mode
            && placed.is_some_and(|placed| {
                placed
                    .impostors()
                    .iter()
                    .any(|record| record.sphere[3] > 0.0)
            })
        {
            self.prepare_impostor_casters(&cx.core.device);
        }
        if let Some(placed) = placed {
            for &(min, max) in placed.moved() {
                self.clipmap.dirty_box(min, max);
            }
        }
        if let Some(heightfield) = cx.parts.face::<dyn HeightfieldFace>() {
            for change in heightfield.ground_changes() {
                self.dirty_ground((change.min, change.max), (&change.pyramid, change.level));
            }
        }
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        if self.frame.is_none() {
            return;
        }
        let ids = self.ids;
        if let Some(pass) = ids.select {
            let _ = graph
                .add(cx.table, pass)
                .read(ids.poses)
                .read_written(ids.visible)
                .write(ids.casters);
        }
        if let Some(pass) = ids.maps {
            let _ = graph
                .add(cx.table, pass)
                .read(ids.poses)
                .read(ids.casters)
                .read_written(ids.placed)
                .read_written(ids.leaves)
                .write(ids.shadow_maps);
        }
        if let Some(pass) = ids.marks {
            let _ = graph
                .add(cx.table, pass)
                .read(ids.depth)
                .write(ids.shadow_marks);
        }
    }

    fn encode(
        &mut self,
        pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        match pass {
            0 => self.encode_selection(&cx.core.device, encoder, &cx.core.profiler),
            1 => {
                self.draws = match &self.frame {
                    Some(frame) => self.encode_maps(encoder, frame, cx),
                    None => 0,
                };
            }
            2 => self.encode_marks(cx, encoder),
            _ => {}
        }
        0
    }

    fn contribute<'s>(&'s self, group: &str, out: &mut Vec<(&'static str, Bound<'s>)>) {
        if group == "frame" {
            out.extend(self.maps.entries());
        }
        if group == "ambient" {
            out.extend(self.sky.bindings());
        }
    }

    fn after_submit(&mut self) {
        self.selection.after_submit();
        self.marks.after_submit();
    }

    fn capacity_bytes(&self) -> u64 {
        self.maps.capacity_bytes() + self.selection.capacity_bytes() + self.marks.capacity_bytes()
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        let frame = self.frame.as_ref();
        let casters = frame.and(self.caster_counts());
        let tiles = frame.map(ShadowFrame::tiles).unwrap_or_default();
        let (sky_maps, sky_drawn) = self.sky_counts();
        stats.shadows = ShadowStats {
            active: frame.is_some(),
            draws: if frame.is_some() { self.draws } else { 0 },
            tiles_marked: tiles.marked,
            tiles_resident: tiles.resident,
            tiles_drawn: tiles.drawn,
            tiles_reclaimed: tiles.reclaimed,
            tiles_missing: tiles.missing,
            moving_tiles: tiles.moving,
            casters: casters.map(|casters| casters[0]),
            moving_casters: casters.map(|casters| casters[1]),
            moving_left_out: casters.map(|casters| casters[2]),
            sky_maps,
            sky_drawn,
        };
        engine_profile::counter("render.shadow_groups", self.casters.groups().len() as f64);
    }

    fn shading(&self, snapshot: &RenderSnapshot, shading: &mut Shading) {
        let enabled = snapshot.environment.shadows.sanitized().enabled;
        shading.set("SHADOW_MAPS", if enabled { 1.0 } else { 0.0 });
    }

    fn asset_generation(&self) -> u64 {
        self.groups_generation
    }

    fn assets<'s>(&self, snapshot: &'s RenderSnapshot, assets: &mut PartAssets<'s>) {
        assets.meshes.extend(
            self.casters
                .groups()
                .iter()
                .map(|group| (group.mesh, &snapshot.meshes[group.source_index])),
        );
    }

    fn batch_group(&self, batch: usize) -> Option<u32> {
        Some(self.casters.batch_group(batch))
    }
}

impl ShadowCastersFace for ShadowPass {
    fn region_views(&self) -> &[(usize, KeptRefresh)] {
        &self.region_views
    }
}
