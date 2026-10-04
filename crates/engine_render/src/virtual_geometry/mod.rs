//! Virtual geometry (decision 044): resident virtual meshes' hierarchies,
//! clusters and pages in one arena, and each frame's selection of the
//! clusters every view draws (`virtual_geometry.wgsl`). The camera's
//! clusters go to per-material bins the opaque pass draws 128 triangles a
//! cluster from (`meshes/vertex.wgsl`'s `vs_virtual`); each shadow map drawn every
//! frame (decision 020) selects its casters' clusters into a bin of its own,
//! drawn into the map by `vs_shadow`.
//!
//! Frames culling occluded instances (decision 040) retest the camera's
//! clusters the last frame's depth pyramid hid against this frame's, after
//! phase one, and draw those it does not hide in phase two from twin bins.
//!
//! Selection runs on the GPU only, for all views in one pass; each bin's
//! region, the retest list and the node and group queues grow to what
//! frames read back. Its parts: the arena and its page streaming (`arena`),
//! each frame's selection storage and uniform (`selection`), the counts and
//! group feedback read back (`feedback`), and the pipelines (`layouts`).

mod arena;
mod feedback;
mod layouts;
mod selection;

use std::sync::Arc;

use engine_core::glam::{Mat4, Vec3};
use engine_core::{MeshHandle, VirtualMesh};

use crate::bindings::VIRTUAL_SHADOW;
use crate::browser::FramePoses;
use crate::buffers::entry;
use crate::cached::Cached;
use crate::gpu_data::VIRTUAL_SHADER;
use crate::shadows::VIRTUAL_SHADOW_PLANES as PLANES;

use arena::VirtualArena;
use feedback::Readbacks;
use layouts::{Pipelines, ShadowDrawing};
use selection::SelectionStorage;

const NONE: u32 = u32::MAX;
const GROUP: u32 = 64;
/// Counters before the bins' (`virtual_geometry.wgsl`).
const COUNTERS: usize = 12;
/// Views one frame selects for: the camera's and those of the shadow maps
/// drawn every frame (up to four cascades and the moving map).
pub(crate) const MAX_VIEWS: usize = 8;
/// Floats of a view record: its planes, the eye, the error projection and
/// a direction.
const VIEW_FLOATS: usize = PLANES * 4 + 12;
/// Bytes of the uniform before its views: counts, two projections, the
/// viewport, the camera's tests and the feedback's locations.
const FRAME_BYTES: usize = 192;
const UNIFORM_BYTES: u64 = (FRAME_BYTES + MAX_VIEWS * VIEW_FLOATS * 4) as u64;
/// Camera tests (`virtual_geometry.wgsl`).
const TEST_PREVIOUS: u32 = 1;
const TEST_SUBPIXEL: u32 = 2;
/// Bytes of an entry of the selected clusters: visible ID and cluster.
pub(crate) const VISIBLE_BYTES: u64 = 8;
/// ID sources of batches: the camera's, and the shadow maps' casters.
const CAMERA: usize = 0;
const SHADOWS: usize = 1;

/// A batch of instances of a virtual mesh in one view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VirtualBatch {
    /// Start of its IDs: visible IDs (GPU-presented frames) or slot IDs
    /// (CPU-prepared ones) for the camera, a map's casters for shadows.
    pub start: u32,
    /// Instances, when known; otherwise the instance count of draw
    /// arguments `draw` (GPU presentation's, or the shadow maps').
    pub count: Option<u32>,
    pub draw: u32,
    pub mesh: u32,
    pub bin: u32,
    pub double_sided: bool,
    pub view: u32,
}

/// How a view tests clusters' normal cones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Cone {
    /// From the eye (perspective views).
    Eye,
    /// Along a direction the view looks (orthographic views).
    Along(Vec3),
}

/// A view clusters are selected for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualView {
    /// Inward planes, normalized; unused ones hold everything.
    pub planes: [[f32; 4]; PLANES],
    /// The camera's eye, which errors project from.
    pub eye: Vec3,
    /// The smallest distance errors project from.
    pub min_distance: f32,
    /// cot(fovy / 2) / 2 of the camera.
    pub projection: f32,
    /// The largest projected error drawn, as a fraction of the viewport
    /// height.
    pub threshold: f32,
    /// The world error a shadow view accepts where the camera sees nothing
    /// (0 for the camera's own view).
    pub unseen_error: f32,
    pub cone: Cone,
}

/// Counts a frame read back: clusters and triangles the camera and the
/// shadow views selected, the camera's clusters occlusion culling left
/// undrawn, and entries dropped for want of room; and this frame's pages:
/// resident, loading, and loads dropped for want of room.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VirtualStats {
    pub clusters: u32,
    pub triangles: u32,
    pub shadow_clusters: u32,
    pub shadow_triangles: u32,
    pub occluded: u32,
    pub dropped: u32,
    pub resident_pages: u32,
    pub loading_pages: u32,
    pub missing_pages: u32,
    pub dropped_pages: u32,
}

/// The camera's cluster tests of a frame (decisions 040 and 041): the
/// projection it rasterizes, the viewport and depth pyramid levels, the
/// projection the last frame rasterized when its pyramid is this frame's
/// to reuse, the first phase-two bin when occlusion culling draws two
/// phases, and whether clusters covering no pixel center stay undrawn.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraTests {
    pub view_projection: Mat4,
    pub viewport: (u32, u32, u32),
    pub previous: Option<Mat4>,
    pub phase_two: Option<u32>,
    pub subpixel: bool,
}

/// The buffers one source of batches reads: IDs and draw arguments, with a
/// key that changes when either does.
#[derive(Clone, Copy)]
pub(crate) struct IdBuffers<'a> {
    pub ids: &'a wgpu::Buffer,
    pub draws: &'a wgpu::Buffer,
    pub key: u64,
}

/// What a frame's selection reads: the batches of the camera's and the
/// shadow maps' ID sources, the bins they select into, the views, the
/// camera's tests and the budget of resident pages; this frame's poses, the
/// IDs and draw arguments of both sources (the shadow maps' only while
/// shadow views select), the most instances a batch of each source has,
/// the shadow maps' view matrices shadow bins are drawn with, and
/// occlusion culling's depth pyramid with its generation.
pub(crate) struct SelectionFrame<'a> {
    pub batches: [&'a [VirtualBatch]; 2],
    pub bins: usize,
    pub views: &'a [VirtualView],
    pub tests: CameraTests,
    pub budget: Option<u32>,
    pub poses: FramePoses<'a>,
    pub camera: IdBuffers<'a>,
    pub shadows: Option<IdBuffers<'a>>,
    pub max_instances: [u32; 2],
    pub shadow_views: Option<&'a wgpu::Buffer>,
    pub pyramid: Option<(&'a wgpu::TextureView, u64)>,
}

/// One pose buffer's selection groups.
struct Groups {
    // Per ID source; shadows' only while shadow views select.
    seed: [Option<wgpu::BindGroup>; 2],
    // Every pass writing dispatch arguments: each level's, the work's and
    // the retest's.
    args: wgpu::BindGroup,
    // Both parities of the traversal.
    traverse: wgpu::BindGroup,
    select: wgpu::BindGroup,
    bin_args: wgpu::BindGroup,
    retest: wgpu::BindGroup,
}

/// Generations of the arena, selection storage, poses, camera IDs, shadow
/// IDs and the depth pyramid bind groups were built for.
type GroupsKey = (u64, u64, u64, u64, Option<u64>, Option<u64>);
/// The arena generation, the poses' key and the shadow views' buffer a
/// shadow drawing group binds.
type ShadowGroupKey = (u64, u64, wgpu::Buffer);

pub(crate) struct VirtualGeometryPass {
    arena: VirtualArena,
    storage: SelectionStorage,
    readbacks: Readbacks,
    // Built with the first arena (decision 049): scenes without virtual
    // meshes compile none of them.
    module: Option<wgpu::ShaderModule>,
    pipelines: Option<Pipelines>,
    // Bindings per pose buffer parity (temporal frames alternate them), and
    // this frame's parity.
    groups: [Cached<GroupsKey, Groups>; 2],
    parity: usize,
    // This frame's poses and their key, for shadow drawing.
    poses: Option<(wgpu::Buffer, u64)>,
    // Depth-only drawing of shadow bins, once shadow views select, and its
    // groups per pose buffer parity.
    shadow_drawing: Option<ShadowDrawing>,
    shadow_groups: [Cached<ShadowGroupKey, wgpu::BindGroup>; 2],
    // Stands in for the depth pyramid on frames without one.
    _no_pyramid: wgpu::Texture,
    no_pyramid: wgpu::TextureView,
}

impl VirtualGeometryPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let no_pyramid = crate::targets::texture_2d(
            device,
            "engine/virtual-no-pyramid",
            (1, 1),
            1,
            1,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let no_pyramid_view = no_pyramid.create_view(&Default::default());
        Self {
            arena: VirtualArena::new(),
            storage: SelectionStorage::new(device),
            readbacks: Readbacks::new(),
            module: None,
            pipelines: None,
            groups: Default::default(),
            parity: 0,
            poses: None,
            shadow_drawing: None,
            shadow_groups: Default::default(),
            _no_pyramid: no_pyramid,
            no_pyramid: no_pyramid_view,
        }
    }

    /// Makes `geometry` resident as mesh `handle`'s, once.
    pub(crate) fn insert(&mut self, handle: MeshHandle, geometry: &Arc<VirtualMesh>) {
        self.arena.insert(handle, geometry);
    }

    /// Releases every resident mesh `keep` rejects.
    pub(crate) fn retain(&mut self, keep: impl Fn(MeshHandle) -> bool) {
        self.arena.retain(keep);
    }

    /// The arena index of mesh `handle`'s geometry, once uploaded.
    pub(crate) fn mesh_index(&self, handle: MeshHandle) -> Option<u32> {
        self.arena.mesh_index(handle)
    }

    /// Whether resident mesh `index` is closed (`VirtualMesh::closed`).
    pub(crate) fn closed(&self, index: u32) -> bool {
        self.arena.closed(index)
    }

    /// Rebuilds the arena when resident meshes changed: every record
    /// rebased to absolute indices, and a new page pool holding the pinned
    /// pages (and, without `budget`, every page the meshes' words hold).
    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: Option<u32>,
    ) {
        if self.arena.upload(device, queue, budget) {
            // Groups of the old arena hold its buffers.
            self.groups = Default::default();
        }
    }

    /// The arena buffers vertex shaders read (meshes, clusters, pages) and
    /// their generation.
    pub(crate) fn arena(&self) -> Option<([&wgpu::Buffer; 3], u64)> {
        self.arena.vertex_buffers()
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.arena.capacity_bytes() + self.storage.capacity_bytes()
    }

    /// Prepares `frame`'s selection: streams pages for the groups the last
    /// read-back frame used and requested (keeping at most the budget
    /// resident), grows what it lacked room for, uploads the batches, bins,
    /// views and the camera's tests, and binds the frame's inputs.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &SelectionFrame<'_>,
    ) {
        let mut drops = [0; 4];
        let mut requests = None;
        if let Some(feedback) = self.readbacks.take() {
            drops = feedback.drops;
            let (_, layout_frame) = self.arena.frames();
            if feedback.frame >= layout_frame {
                requests = Some((feedback.used, feedback.requested, feedback.frame));
            }
        }
        self.arena.stream(device, queue, frame.budget, requests);
        self.storage.grow(device, drops, frame.bins);
        let feedback_words = (self.arena.group_count() as u64).div_ceil(32);
        if self
            .storage
            .size_bins(device, queue, frame.bins, feedback_words)
        {
            self.readbacks.clear();
        }
        self.storage
            .upload_batches(device, queue, frame.batches, frame.max_instances);
        self.storage.write_uniform(
            queue,
            frame.bins,
            frame.views,
            &frame.tests,
            self.arena.bits_base(),
        );
        self.bind(device, frame);
    }

    /// Binds this frame's slot poses, the IDs and draw arguments of both
    /// sources, the shadow maps' views and the depth pyramid.
    fn bind(&mut self, device: &wgpu::Device, frame: &SelectionFrame<'_>) {
        let poses = &frame.poses;
        self.poses = Some((poses.current().clone(), poses.key()));
        self.parity = poses.parity;
        let Some(arena) = self.arena.buffers() else {
            return;
        };
        let module: &wgpu::ShaderModule = self.module.get_or_insert_with(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/virtual_geometry.wgsl"),
                source: wgpu::ShaderSource::Wgsl(VIRTUAL_SHADER.into()),
            })
        });
        let pipelines = self
            .pipelines
            .get_or_insert_with(|| Pipelines::new(device, module));
        if let Some(views) = frame.shadow_views {
            let drawing = self
                .shadow_drawing
                .get_or_insert_with(|| ShadowDrawing::new(device, module));
            let key = (self.arena.generation(), poses.key(), views.clone());
            self.shadow_groups[self.parity].get_or_insert_with(key, || {
                crate::pipelines::group(
                    device,
                    "engine/virtual-shadow-group",
                    &VIRTUAL_SHADOW,
                    &drawing.layout,
                    &[
                        entry("meshes", &arena.meshes),
                        entry("clusters", &arena.clusters),
                        entry("poses", poses.current()),
                        entry("pages", &arena.pages),
                        (
                            "shadow_view",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: views,
                                offset: 0,
                                size: wgpu::BufferSize::new(64),
                            }),
                        ),
                    ],
                )
            });
        }
        let key = (
            self.arena.generation(),
            self.storage.generation(),
            poses.key(),
            frame.camera.key,
            frame.shadows.map(|shadows| shadows.key),
            frame.pyramid.map(|(_, generation)| generation),
        );
        let pyramid = frame.pyramid.map_or(&self.no_pyramid, |(view, _)| view);
        let storage = &self.storage;
        self.groups[self.parity].get_or_insert_with(key, || {
            let texture = || ("depth_pyramid", crate::pipelines::view(pyramid));
            let [queue_a, queue_b] = &storage.queues;
            let seed = |source: usize, ids: &IdBuffers<'_>| {
                pipelines.group(
                    device,
                    &pipelines.seed_family,
                    &[
                        entry("frame", &storage.uniform),
                        entry("batches", &storage.batches[source]),
                        entry("meshes", &arena.meshes),
                        entry("ids", ids.ids),
                        entry("draws", ids.draws),
                        entry("queue_a", queue_a),
                        entry("queue_b", queue_b),
                        entry("counters", &storage.counters),
                    ],
                )
            };
            Groups {
                seed: [
                    Some(seed(CAMERA, &frame.camera)),
                    frame.shadows.as_ref().map(|shadows| seed(SHADOWS, shadows)),
                ],
                args: pipelines.group(
                    device,
                    &pipelines.seed_family,
                    &[
                        entry("frame", &storage.uniform),
                        entry("counters", &storage.counters),
                        entry("dispatch", &storage.dispatch),
                    ],
                ),
                traverse: pipelines.group(
                    device,
                    &pipelines.traverse_family,
                    &[
                        entry("frame", &storage.uniform),
                        entry("nodes", &arena.nodes),
                        entry("poses", poses.current()),
                        entry("queue_a", queue_a),
                        entry("queue_b", queue_b),
                        entry("work", &storage.work),
                        entry("counters", &storage.counters),
                    ],
                ),
                select: pipelines.group(
                    device,
                    &pipelines.select_family,
                    &[
                        entry("frame", &storage.uniform),
                        entry("groups", &arena.groups),
                        entry("clusters", &arena.clusters),
                        entry("poses", poses.current()),
                        entry("work", &storage.work),
                        entry("counters", &storage.counters),
                        entry("visible", &storage.visible),
                        entry("bins", &storage.bins),
                        texture(),
                        entry("retest", &storage.retest),
                    ],
                ),
                bin_args: pipelines.group(
                    device,
                    &pipelines.bin_family,
                    &[
                        entry("frame", &storage.uniform),
                        entry("counters", &storage.counters),
                        entry("bins", &storage.bins),
                        entry("draw_args", &storage.draw_args),
                    ],
                ),
                retest: pipelines.group(
                    device,
                    &pipelines.select_family,
                    &[
                        entry("frame", &storage.uniform),
                        entry("clusters", &arena.clusters),
                        entry("poses", poses.current()),
                        entry("counters", &storage.counters),
                        entry("visible", &storage.visible),
                        entry("bins", &storage.bins),
                        texture(),
                        entry("retest", &storage.retest),
                    ],
                ),
            }
        });
    }

    /// Records this frame's selection: seeds of both ID sources, one
    /// traversal per hierarchy level, the groups' clusters and each bin's
    /// draw arguments, then the counters' copy for read-back. Nothing
    /// without bound batches.
    pub(crate) fn encode(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let Some(groups) = self.groups[self.parity].get() else {
            return;
        };
        let (batch_counts, max_instances) = self.storage.batches();
        if batch_counts == [0, 0] {
            return;
        }
        let pipelines = self.pipelines.as_ref().expect("bound groups build them");
        let storage = &self.storage;
        encoder.clear_buffer(&storage.counters, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/virtual-selection"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.seed);
            for source in [CAMERA, SHADOWS] {
                if batch_counts[source] == 0 {
                    continue;
                }
                if let Some(group) = &groups.seed[source] {
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(
                        max_instances[source].div_ceil(GROUP).max(1),
                        batch_counts[source],
                        1,
                    );
                }
            }
            for level in 0..self.arena.depth() as usize {
                let parity = level % 2;
                pass.set_pipeline(&pipelines.level_args[parity]);
                pass.set_bind_group(0, &groups.args, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&pipelines.traverse[parity]);
                pass.set_bind_group(0, &groups.traverse, &[]);
                pass.dispatch_workgroups_indirect(&storage.dispatch, 0);
            }
            pass.set_pipeline(&pipelines.work_args);
            pass.set_bind_group(0, &groups.args, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&pipelines.select);
            pass.set_bind_group(0, &groups.select, &[]);
            pass.dispatch_workgroups_indirect(&storage.dispatch, 12);
            pass.set_pipeline(&pipelines.bin_args);
            pass.set_bind_group(0, &groups.bin_args, &[]);
            pass.dispatch_workgroups((storage.bin_count() as u32).div_ceil(GROUP).max(1), 1, 1);
        }
        // Frames retesting read their counters back after the retest.
        if !storage.retesting() {
            self.read_counters(device, encoder);
        }
    }

    /// Records a copy of this frame's counters for read-back, when a
    /// read-back buffer is free.
    fn read_counters(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let (frame, _) = self.arena.frames();
        self.readbacks.record(
            device,
            encoder,
            &self.storage.counters,
            frame,
            self.storage.feedback(),
        );
    }

    /// Records the retest after phase one and this frame's depth pyramid
    /// (decision 040): the waiting clusters it does not hide go to phase
    /// two's bins, whose draw arguments follow. Nothing on frames drawing
    /// one phase.
    pub(crate) fn encode_retest(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let Some(groups) = self.groups[self.parity].get() else {
            return;
        };
        let (batch_counts, _) = self.storage.batches();
        if !self.storage.retesting() || batch_counts == [0, 0] {
            return;
        }
        let pipelines = self.pipelines.as_ref().expect("bound groups build them");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/virtual-retest"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.retest_args);
            pass.set_bind_group(0, &groups.args, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&pipelines.retest);
            pass.set_bind_group(0, &groups.retest, &[]);
            pass.dispatch_workgroups_indirect(&self.storage.dispatch, 12);
            pass.set_pipeline(&pipelines.bin_args);
            pass.set_bind_group(0, &groups.bin_args, &[]);
            pass.dispatch_workgroups(
                (self.storage.bin_count() as u32).div_ceil(GROUP).max(1),
                1,
                1,
            );
        }
        self.read_counters(device, encoder);
    }

    /// Every bin's selected clusters (visible ID and cluster), and a
    /// generation that advances when the buffer is replaced.
    pub(crate) fn visible(&self) -> (&wgpu::Buffer, u64) {
        (&self.storage.visible, self.storage.generation())
    }

    /// Bin `bin`'s first entry of [`Self::visible`].
    pub(crate) fn bin_base(&self, bin: usize) -> u32 {
        self.storage.bin_base(bin)
    }

    /// The selected clusters of bin `bin` (an instance-stepped vertex stream
    /// of visible ID and cluster) and its indirect draw arguments.
    pub(crate) fn bin_source(&self, bin: usize) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        self.storage.bin_source(bin)
    }

    /// Draws shadow bin `bin` into the map of view matrix `view_offset`
    /// (an offset into the shadow maps' views). Returns whether it drew;
    /// the pass's pipeline and first group are this drawing's afterwards.
    pub(crate) fn draw_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        bin: usize,
        view_offset: u32,
    ) -> bool {
        let Some(drawing) = &self.shadow_drawing else {
            return false;
        };
        let Some(group) = self.shadow_groups[self.parity].get() else {
            return false;
        };
        if bin >= self.storage.bin_count() {
            return false;
        }
        pass.set_pipeline(&drawing.pipeline);
        pass.set_bind_group(0, group, &[view_offset]);
        let (stream, args, offset) = self.storage.bin_source(bin);
        pass.set_vertex_buffer(0, stream);
        pass.draw_indirect(args, offset);
        true
    }

    /// Counts the last read-back frame selected (none before the first
    /// read-back), and this frame's pages.
    pub(crate) fn stats(&self) -> Option<VirtualStats> {
        self.arena.buffers().map(|_| {
            let [resident_pages, loading_pages, missing_pages, dropped_pages] = self.arena.pages();
            VirtualStats {
                resident_pages,
                loading_pages,
                missing_pages,
                dropped_pages,
                ..self.readbacks.last().unwrap_or_default()
            }
        })
    }

    /// Requests maps of counters recorded in the frame just submitted.
    pub(crate) fn after_submit(&mut self) {
        self.readbacks.after_submit(self.storage.retest_capacity());
    }
}

/// Normalized inward planes of `view_projection`'s frustum (WebGPU depth).
pub(crate) fn planes(view_projection: engine_core::glam::Mat4) -> [[f32; 4]; 6] {
    let (planes, _) = crate::gpu_data::frustum_uniform(view_projection);
    planes.map(|plane| {
        let length = Vec3::new(plane[0], plane[1], plane[2]).length();
        if length > 0.0 && length.is_finite() {
            plane.map(|value| value / length)
        } else {
            [0.0, 0.0, 0.0, 1.0]
        }
    })
}
