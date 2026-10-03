//! Sun shadows (decision 020): depth maps seen from the atmosphere's sun.
//!
//! Near cascades, layers of one texture, cover the first stretch of the view,
//! each reaching a third as far as the next (decision 028), and are redrawn
//! every frame. Two far maps share one projection over the shadow region: the kept
//! map holds casters whose pose stayed the same for a few frames and is
//! redrawn only where that changes, a rectangle of its 16x16 tiles per
//! frame, or whole after its projection or the sun moved; the moving map
//! holds every other caster and is redrawn every frame. One compute pass
//! selects each map's casters from every candidate, visible or culled, into
//! indirect draws per mesh; on GPU-presented frames it also keeps each
//! slot's stillness and marks dirty tiles, read back a few frames later. CPU-prepared frames draw
//! every caster into the moving map and leave the kept map unread; public
//! snapshots draw their instances without selection. Casters with virtual
//! geometry (decision 044) are selected into the maps drawn every frame,
//! whose clusters virtual geometry selects and draws after its own
//! selection: [`ShadowPass::encode_selection`], then
//! [`ShadowPass::encode_maps`].
//!
//! Its parts: the maps and their lookups (`maps`), each frame's caster
//! groups (`casters`), where the maps look (`regions`), the selection of
//! each map's casters (`selection`) and the depth pipelines (`pipelines`).
//! What the kept map holds is `shadow_fit::KeptMap`.

mod casters;
mod maps;
mod pipelines;
mod regions;
mod selection;

use std::cell::Cell;

use engine_core::{MaterialHandle, MeshHandle, RenderFrame, RenderSnapshot, glam::Vec3};

use crate::batches::BatchPlan;
use crate::browser::FramePoses;
use crate::gpu_data::{DRAW_ARGS_BYTES, MAX_CASCADES};
use crate::mesh_arena::{Bindings, Ids, MeshDraw};
use crate::profiling::GpuProfiler;
use crate::shadow_fit::{KeptMap, LightBasis, Refresh};

use casters::CasterGroups;
pub(crate) use casters::ShadowGroup;
pub(crate) use maps::SHADOW_FORMAT;
use maps::{ShadowMaps, map_pass};
use pipelines::ShadowPipelines;
use regions::{Regions, SelectionCounts};
use selection::Selection;

// Near cascades are maps 0 to MAX_CASCADES - 1; the far maps follow.
const MOVING_MAP: usize = MAX_CASCADES;
const KEPT_MAP: usize = MAX_CASCADES + 1;
const MAPS: usize = MAX_CASCADES + 2;
// Uniform offset alignment between view matrices: each near cascade's, then
// the far maps'.
const VIEW_STRIDE: u64 = 256;
const FAR_VIEW: u64 = MAX_CASCADES as u64 * VIEW_STRIDE;
const GROUP: u32 = 256;
const SKINNED_GROUP: u32 = 64;
// Mode bits of the selection pass.
const STILLNESS: u32 = 1;
const REFRESH: u32 = 2;

/// Planes of a virtual shadow view: its selection region's five, and the
/// view frustum slice's that the sun's sweep never crosses.
pub(crate) const VIRTUAL_SHADOW_PLANES: usize = 12;

/// A shadow map drawn this frame whose virtual casters' clusters virtual
/// geometry selects (decision 044): its planes around where casters can
/// darken visible receivers (unused ones hold everything), the camera
/// distance its receivers start at (`None` at the camera), its texel's
/// world size, the direction it looks along (away from the sun) and the
/// offset of its view matrix.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualShadowView {
    pub map: usize,
    pub planes: [[f32; 4]; VIRTUAL_SHADOW_PLANES],
    pub min_distance: Option<f32>,
    pub texel: f32,
    pub direction: Vec3,
    pub view_offset: u32,
}

/// Where this frame's casters come from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    /// GPU presentation's candidates, batches and flags.
    Presented,
    /// Every caster of a CPU-prepared retained frame.
    Listed,
    /// The instances of a public snapshot, drawn from the instance buffer.
    Snapshot,
}

/// Inputs of GPU-presented frames' selection, from GPU presentation.
pub(crate) struct Presented<'a> {
    pub candidates: &'a wgpu::Buffer,
    pub batches: &'a wgpu::Buffer,
    pub flags: &'a wgpu::Buffer,
    pub count: u32,
    /// Skinned slots (decision 036) and their world bounds from this
    /// frame's palettes.
    pub skinned: &'a wgpu::Buffer,
    pub skin_bounds: &'a wgpu::Buffer,
    pub skinned_count: u32,
    /// Advances when any of these buffers is replaced.
    pub generation: u64,
}

/// How a caster group's mesh draws (decision 047): its index count, first
/// index and base vertex, and its coarser levels of detail (first index,
/// index count and geometric error in mesh units, finest first).
pub(crate) struct CasterDraw {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub lods: Vec<(u32, u32, f32)>,
}

impl CasterDraw {
    /// The coarsest level whose error is at most `error` world units at unit
    /// scale; 0 for the mesh itself.
    fn level_within(&self, error: f32) -> u8 {
        self.lods
            .iter()
            .take_while(|&&(_, _, lod_error)| lod_error <= error)
            .count() as u8
    }

    /// Index count, first index and base vertex of level `level`.
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

/// What a frame's shadows select from: its snapshot, GPU presentation's
/// buffers on GPU-presented frames, the pose storage of retained frames and
/// the slots it holds, the mesh arena layout (decision 046) and each mesh's
/// placement in it.
pub(crate) struct ShadowInputs<'a> {
    pub snapshot: &'a RenderSnapshot,
    pub presented: Option<Presented<'a>>,
    pub poses: Option<FramePoses<'a>>,
    pub slot_capacity: usize,
    pub mesh_layout: u64,
    pub placement: &'a dyn Fn(MeshHandle) -> CasterDraw,
}

/// What depth passes draw: the plan's batches from public snapshots'
/// `instances`, each mesh's uploaded draw (and whether it is skinned), each
/// masked material's group, and the clusters of a map's virtual casters
/// (decision 044), drawn last in its pass, which returns its draws.
pub(crate) struct MapSources<'a> {
    pub plan: &'a BatchPlan,
    pub instances: &'a wgpu::Buffer,
    pub meshes: &'a dyn Fn(MeshHandle) -> Option<(MeshDraw<'a>, bool)>,
    pub materials: &'a dyn Fn(MaterialHandle) -> Option<&'a wgpu::BindGroup>,
    pub virtual_draw: &'a dyn Fn(&mut wgpu::RenderPass<'_>, usize) -> u32,
}

/// A retained caster group as depth passes draw it: its pipeline choice
/// (masked, skinned), mesh, joints stream, material and group index.
struct Caster<'m> {
    choice: usize,
    draw: MeshDraw<'m>,
    skin: Option<&'m wgpu::Buffer>,
    material: Option<&'m wgpu::BindGroup>,
    index: usize,
}

/// One frame's shadow work, from [`ShadowPass::prepare`].
pub(crate) struct ShadowFrame {
    source: Source,
    candidates: u32,
    // Skinned slots selected by their palette bounds (GPU-presented frames).
    skinned: u32,
    // Near cascades drawn and looked up.
    cascades: u32,
    refresh: Option<Refresh>,
    pose_index: usize,
    read_dirty: bool,
}

impl ShadowFrame {
    /// Kept-map redraw of this frame, for profiling counters: 0 none, 1
    /// tiles, 2 the whole map.
    pub(crate) fn refresh_kind(&self) -> u8 {
        match self.refresh {
            None => 0,
            Some(Refresh::Tiles(_)) => 1,
            Some(Refresh::Whole) => 2,
        }
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(crate) struct ShadowPass {
    // Created with the first frame that has shadows: games without them
    // never compile them.
    pipelines: Option<ShadowPipelines>,
    // Vertex and instance strides and the texture coordinate offset, and the
    // material group layout of masked casters.
    layout: (u64, u64, u64),
    material_layout: wgpu::BindGroupLayout,
    maps: ShadowMaps,
    kept: KeptMap,
    casters: CasterGroups,
    selection: Selection,
    // Pose bindings of both pose buffers, for the generation they belong to.
    pose_groups: Option<(u64, [Option<wgpu::BindGroup>; 2])>,
    // Maps drawn this frame that select virtual casters' clusters.
    virtual_views: Vec<VirtualShadowView>,
}

impl ShadowPass {
    /// The bindings the main passes read; pipelines follow with the first
    /// frame that has shadows. `layout`: the renderer's mesh vertex and
    /// public instance record strides and the texture coordinate offset in
    /// a vertex; `material_layout`, the mesh material group.
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: (u64, u64, u64),
        material_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        Self {
            pipelines: None,
            layout,
            material_layout: material_layout.clone(),
            maps: ShadowMaps::new(device),
            kept: KeptMap::default(),
            casters: CasterGroups::new(),
            selection: Selection::new(device),
            pose_groups: None,
            virtual_views: Vec::new(),
        }
    }

    /// Advances when the map views bound by [`Self::entries`] change.
    pub(crate) fn generation(&self) -> u64 {
        self.maps.generation()
    }

    /// Bindings 2 to 7 of the main passes' first group: the lookup uniform,
    /// the near cascades, the kept and moving maps, the comparison sampler
    /// and the texel sampler of receiver planes.
    pub(crate) fn entries(&self) -> [wgpu::BindGroupEntry<'_>; 6] {
        self.maps.entries()
    }

    /// Meshes this frame's shadows draw; CPU-prepared frames may draw culled
    /// meshes the plan's batches lack.
    pub(crate) fn groups(&self) -> &[ShadowGroup] {
        self.casters.groups()
    }

    /// Group of plan batch `index`, for presentation's batch records.
    pub(crate) fn batch_group(&self, index: usize) -> u32 {
        self.casters.batch_group(index)
    }

    /// Maps drawn this frame whose virtual casters' clusters virtual
    /// geometry selects (decision 044); empty without such casters.
    pub(crate) fn virtual_views(&self) -> &[VirtualShadowView] {
        &self.virtual_views
    }

    /// The selected casters' IDs (each map's region `map * capacity`, then
    /// each group's) and indirect arguments (five words per map and group,
    /// `map * groups + group`), the ID capacity per map, and a key that
    /// changes when the buffers do.
    pub(crate) fn caster_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u32, u64) {
        (
            &self.selection.ids,
            &self.selection.args,
            self.selection.id_capacity as u32,
            self.selection.generation(),
        )
    }

    /// The view matrices of every map, at each map's view offset, once
    /// shadows ran.
    pub(crate) fn view_buffer(&self) -> Option<&wgpu::Buffer> {
        self.pipelines
            .as_ref()
            .map(|pipelines| &pipelines.view_buffer)
    }

    /// The latest caster totals of the near cascades together, the moving
    /// map and the kept map read back from GPU-selected frames, a few frames
    /// old; the kept total counts only frames that redrew part of it.
    pub(crate) fn caster_counts(&self) -> Option<[u32; 3]> {
        self.selection.caster_counts()
    }

    /// Bytes of shadow maps and selection buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.capacity_bytes() + self.selection.capacity_bytes()
    }

    /// Plans this frame's draw groups, one per mesh: of the plan's batches
    /// (GPU-presented frames and public snapshots) or of `frame`'s casters
    /// (CPU-prepared frames). Call before mesh uploads, which must include
    /// every group's mesh.
    pub(crate) fn plan_groups(
        &mut self,
        snapshot: &RenderSnapshot,
        plan: &BatchPlan,
        frame: Option<&RenderFrame<'_>>,
        gpu: bool,
    ) {
        if !snapshot.shadows.enabled {
            self.casters.clear();
            return;
        }
        if self.casters.plan(snapshot, plan, frame, gpu) {
            self.selection.groups_changed();
            if self.casters.source() == Source::Presented {
                // Removed casters leave their kept shadows: redraw every tile.
                self.kept.dirty_all();
            }
        }
    }

    /// Prepares this frame's shadow maps and lookups, or disables lookups
    /// when shadows are off.
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
        self.kept.merge(self.selection.take_dirty_tiles());
        self.virtual_views.clear();
        let settings = snapshot.shadows.sanitized();
        if !settings.enabled || self.casters.groups().is_empty() {
            if !settings.enabled {
                self.maps.release();
            }
            self.kept.keep_while(settings.enabled);
            // Shadows off: lookups see state.x 0.
            self.maps.write_uniform(queue, bytemuck::Zeroable::zeroed());
            return None;
        }
        let source = self.casters.source();
        let pipelines = self.pipelines.get_or_insert_with(|| {
            ShadowPipelines::new(device, self.layout, &self.material_layout)
        });
        if source == Source::Snapshot {
            pipelines.ensure_full(device, self.layout, &self.material_layout);
        }
        if self
            .maps
            .ensure(device, settings.resolution, settings.cascades)
        {
            // New maps hold nothing: the kept one is redrawn whole.
            self.kept.invalidate();
        }
        // The far region: the settings' bounds, or a cube around an anchor
        // that follows the camera a quarter of the distance at a time.
        let basis = LightBasis::new(snapshot.atmosphere.sanitized().sun_direction);
        let eye = snapshot.camera_position;
        let far = self.kept.place(
            basis,
            settings.bounds,
            settings.distance,
            eye,
            settings.resolution,
        );
        let regions = Regions::new(basis, far, &settings, snapshot.view_projection, eye);
        let classify = source == Source::Presented;
        if classify
            && self
                .selection
                .ensure_stillness(device, inputs.slot_capacity)
        {
            // Stillness starts over: the kept map must be redrawn whole.
            self.kept.invalidate();
        }
        let refresh = if classify { self.kept.refresh() } else { None };
        // Selection outputs, and the draw arguments' template: each group's
        // mesh placement at the level of detail each map draws it at.
        self.selection
            .ensure_outputs(device, self.casters.groups().len(), self.casters.total());
        let draws: Vec<CasterDraw> = self
            .casters
            .groups()
            .iter()
            .map(|group| (inputs.placement)(group.mesh))
            .collect();
        let levels: Vec<u8> = regions
            .errors()
            .iter()
            .flat_map(|&error| draws.iter().map(move |draw| draw.level_within(error)))
            .collect();
        let update = self.selection.update_template(
            device,
            queue,
            &self.casters,
            levels,
            |index, level| draws[index].range(level),
            inputs.mesh_layout,
        );
        if update.kept_levels {
            // Casters drawn at other levels than the kept map holds redraw it
            // whole.
            self.kept.invalidate();
        }
        if update.dirty_replaced {
            self.kept.dirty_all();
        }
        let pipelines = self.pipelines.as_ref().expect("created above");
        queue.write_buffer(
            &pipelines.view_buffer,
            0,
            bytemuck::cast_slice(&regions.view_matrices()),
        );
        let (candidates, skinned) = match (presented, source) {
            (Some(presented), Source::Presented) => (presented.count, presented.skinned_count),
            (_, Source::Listed) => (self.casters.listed.len() as u32, 0),
            _ => (0, 0),
        };
        if source != Source::Snapshot {
            let cull = regions.selection(
                snapshot,
                &settings,
                refresh,
                classify,
                SelectionCounts {
                    candidates,
                    skinned,
                    groups: self.casters.groups().len() as u32,
                    capacity: self.selection.id_capacity as u32,
                    virtual_geometry: self
                        .casters
                        .groups()
                        .iter()
                        .any(|group| group.virtual_geometry),
                },
                &mut self.virtual_views,
            );
            queue.write_buffer(&pipelines.cull, 0, bytemuck::bytes_of(&cull));
        }
        self.maps
            .write_uniform(queue, regions.lookup_uniform(&settings, classify));
        // Bindings: poses of this frame, and the selection inputs.
        if let Some(poses) = poses {
            if self
                .pose_groups
                .as_ref()
                .is_none_or(|(generation, groups)| {
                    *generation != poses.generation || groups[poses.parity].is_none()
                })
            {
                let group = |buffer: Option<&wgpu::Buffer>, palette: &wgpu::Buffer| {
                    buffer.map(|buffer| {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("engine/shadow-pose-group"),
                            layout: &pipelines.pose_layout,
                            entries: &[entry(0, buffer), entry(1, poses.colors), entry(2, palette)],
                        })
                    })
                };
                self.pose_groups = Some((
                    poses.generation,
                    [
                        group(poses.buffers[0], poses.palettes[0]),
                        group(poses.buffers[1], poses.palettes[1]),
                    ],
                ));
            }
            if source != Source::Snapshot {
                self.selection.bind(
                    device,
                    pipelines,
                    source,
                    presented.as_ref(),
                    poses,
                    skinned,
                );
            }
        }
        Some(ShadowFrame {
            source,
            candidates,
            skinned,
            cascades: regions.near().len() as u32,
            refresh,
            pose_index: poses.as_ref().map_or(0, |poses| poses.parity),
            read_dirty: source != Source::Snapshot,
        })
    }

    /// Records caster selection of `frame`, timed by `profiler`
    /// (`shadow_select`).
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
        self.selection.encode(
            device,
            encoder,
            pipelines,
            frame,
            self.casters.groups().len(),
            profiler,
        );
    }

    /// Records the depth passes of `frame` from `sources`, after
    /// [`Self::encode_selection`] and virtual geometry's selection, timed by
    /// `profiler` (`shadow_near` from the first cascade to the last,
    /// `shadow_moving` and `shadow_kept`); returns the indexed draws, one per
    /// mesh group and map drawn.
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
            plan,
            instances,
            meshes,
            materials,
            virtual_draw,
        } = sources;
        let draws = Cell::new(0_u32);
        let groups = self.casters.groups();
        let group_count = groups.len() as u64;
        let (ids, args, id_capacity) = (
            &self.selection.ids,
            &self.selection.args,
            self.selection.id_capacity,
        );
        let pose_group = self
            .pose_groups
            .as_ref()
            .and_then(|(_, groups)| groups[frame.pose_index].as_ref());
        // Masked groups switch to the masked pipeline and their material.
        let bind_material = |pass: &mut wgpu::RenderPass<'_>,
                             masked: &mut bool,
                             material: Option<MaterialHandle>,
                             pipelines: [&wgpu::RenderPipeline; 2]| {
            let group = material.map(materials);
            if group.is_some() != *masked {
                *masked = group.is_some();
                pass.set_pipeline(pipelines[usize::from(*masked)]);
            }
            if let Some(Some(group)) = group {
                pass.set_bind_group(2, group, &[]);
            }
            // A masked material without a binding draws nothing.
            group.is_none_or(|group| group.is_some())
        };
        // Plain, masked, skinned and masked skinned retained casters.
        let selected = [
            &pipelines.affine,
            &pipelines.affine_masked,
            &pipelines.affine_skinned,
            &pipelines.affine_masked_skinned,
        ];
        // Retained casters in pipeline, vertex buffer and material order,
        // which passes switch once per run: depth alone is written, so the
        // order changes no texel. Virtual geometry draws its casters'
        // clusters (decision 044), and masked materials without bindings
        // draw nothing.
        let mut casters: Vec<Caster<'_>> = if pose_group.is_some() {
            groups
                .iter()
                .enumerate()
                .filter(|(_, group)| !group.virtual_geometry)
                .filter_map(|(index, group)| {
                    let (draw, skinned) = meshes(group.mesh)?;
                    let material = match group.material.map(materials) {
                        Some(None) => return None,
                        Some(Some(material)) => Some(material),
                        None => None,
                    };
                    let skin = draw.skin.filter(|_| skinned);
                    Some(Caster {
                        choice: usize::from(material.is_some()) + 2 * usize::from(skin.is_some()),
                        draw,
                        skin,
                        material,
                        index,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
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
        let draw_selected = |pass: &mut wgpu::RenderPass<'_>, map: usize| {
            let Some(pose_group) = pose_group else {
                return;
            };
            pass.set_pipeline(&pipelines.affine);
            pass.set_bind_group(1, pose_group, &[]);
            let mut bound = 0;
            let mut bindings = Bindings::default();
            for caster in &casters {
                if caster.choice != bound {
                    pass.set_pipeline(selected[caster.choice]);
                    bound = caster.choice;
                }
                if let Some(material) = caster.material {
                    bindings.group(pass, 2, material);
                }
                if let Some(skin) = caster.skin {
                    bindings.vertex(pass, 2, skin);
                }
                bindings.vertex(pass, 0, caster.draw.vertices);
                bindings.index(pass, caster.draw.indices);
                let caster_ids = if self.selection.regions_in_args {
                    Ids::Whole(ids)
                } else {
                    let start = map * id_capacity + groups[caster.index].start as usize;
                    Ids::Region(ids.slice(start as u64 * 4..))
                };
                bindings.ids(pass, 1, caster_ids);
                pass.draw_indexed_indirect(
                    args,
                    (map as u64 * group_count + caster.index as u64) * DRAW_ARGS_BYTES,
                );
                draws.set(draws.get() + 1);
            }
            draws.set(draws.get() + virtual_draw(pass, map));
        };
        let draw_snapshot = |pass: &mut wgpu::RenderPass<'_>| {
            let full = pipelines
                .full
                .as_ref()
                .expect("prepared for snapshot frames");
            pass.set_pipeline(&full[0]);
            pass.set_vertex_buffer(1, instances.slice(..));
            let mut masked = false;
            let mut bindings = Bindings::default();
            for (index, batch) in plan.batches.iter().enumerate() {
                // Transparent surfaces cast no shadow (decision 026).
                let Some(group) = self.casters.group_of_batch(index) else {
                    continue;
                };
                let Some((draw, _)) = meshes(batch.mesh) else {
                    continue;
                };
                if !bind_material(pass, &mut masked, group.material, [&full[0], &full[1]]) {
                    continue;
                }
                bindings.vertex(pass, 0, draw.vertices);
                bindings.index(pass, draw.indices);
                pass.draw_indexed(
                    draw.index_range(),
                    draw.base_vertex,
                    batch.instances.clone(),
                );
                draws.set(draws.get() + 1);
            }
        };
        // Profiled frames time the near cascades together, the moving map and
        // the kept map.
        let last_cascade = (frame.cascades as usize).saturating_sub(1);
        let writes = |map: usize| {
            let (span, begin, end) = match map {
                MOVING_MAP => ("shadow_moving", true, true),
                KEPT_MAP => ("shadow_kept", true, true),
                cascade => ("shadow_near", cascade == 0, cascade == last_cascade),
            };
            (begin || end)
                .then(|| profiler.span(span))
                .flatten()
                .map(|span| span.render_part(begin, end))
        };
        // Each near cascade in use (map and view matrix `cascade`), then the
        // moving map.
        let cascades = (0..frame.cascades as usize).map(|cascade| {
            (
                cascade,
                &maps.near_layers[cascade],
                cascade as u64 * VIEW_STRIDE,
            )
        });
        for (map, view, offset) in cascades.chain([(MOVING_MAP, &maps.moving, FAR_VIEW)]) {
            let mut pass = map_pass(encoder, view, wgpu::LoadOp::Clear(1.0), writes(map));
            pass.set_bind_group(0, &pipelines.view_group, &[offset as u32]);
            if frame.source == Source::Snapshot {
                draw_snapshot(&mut pass);
            } else {
                draw_selected(&mut pass, map);
            }
        }
        if let Some(refresh) = frame.refresh {
            let whole = refresh == Refresh::Whole;
            let mut pass = map_pass(
                encoder,
                &maps.kept,
                if whole {
                    wgpu::LoadOp::Clear(1.0)
                } else {
                    wgpu::LoadOp::Load
                },
                writes(KEPT_MAP),
            );
            // Tiles refresh in GPU-presented frames, which bind poses.
            if let Refresh::Tiles(region) = refresh
                && let Some(pose_group) = pose_group
            {
                let [x, y, width, height] = region.scissor(maps.resolution);
                pass.set_scissor_rect(x, y, width, height);
                pass.set_bind_group(0, &pipelines.view_group, &[FAR_VIEW as u32]);
                pass.set_bind_group(1, pose_group, &[]);
                pass.set_pipeline(&pipelines.clear);
                pass.draw(0..3, 0..1);
            }
            pass.set_bind_group(0, &pipelines.view_group, &[FAR_VIEW as u32]);
            draw_selected(&mut pass, KEPT_MAP);
        }
        draws.get()
    }

    /// Requests maps of dirty tiles recorded in the frame just submitted.
    pub(crate) fn after_submit(&mut self) {
        self.selection.after_submit();
    }
}
