#![expect(clippy::expect_used, reason = "bound groups build their buffers")]

mod arena;
mod feedback;
mod layouts;
mod selection;

use std::sync::Arc;

use engine_core::glam::{Mat4, Vec3};
use engine_core::{MeshHandle, VirtualMesh};

use crate::bindings::VIRTUAL_SHADOW;
use crate::buffers::entry;
use crate::cached::Cached;
use crate::gpu_calls::Queue;
use crate::gpu_data::VIRTUAL_SHADER;
use crate::renderer::FramePoses;
use crate::shadows::VIRTUAL_SHADOW_PLANES as PLANES;

pub(crate) use crate::gpu_layouts::constants::MAX_VIEWS;
use crate::gpu_layouts::constants::NONE;
use crate::gpu_layouts::constants::TEST_PREVIOUS;
use crate::gpu_layouts::constants::TEST_SUBPIXEL;
use crate::gpu_layouts::constants::VIRTUAL_GROUP as GROUP;
use arena::VirtualArena;
use feedback::Readbacks;
use layouts::{Pipelines, ShadowDrawing};
use selection::SelectionStorage;

const COUNTERS: usize = 12;
pub(crate) const VISIBLE_BYTES: u64 = 8;
const CAMERA: usize = 0;
const SHADOWS: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VirtualBatch {
    pub start: u32,
    pub count: Option<u32>,
    pub draw: u32,
    pub mesh: u32,
    pub bin: u32,
    pub double_sided: bool,
    pub view: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Cone {
    From(Vec3),
    Along(Vec3),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualView {
    pub planes: [[f32; 4]; PLANES],
    pub eye: Vec3,
    pub min_distance: f32,
    pub projection: f32,
    pub threshold: f32,
    pub unseen_error: f32,
    pub cone: Cone,
}

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
    pub failed_pages: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraTests {
    pub view_projection: Mat4,
    pub viewport: (u32, u32, u32),
    pub previous: Option<Mat4>,
    pub phase_two: Option<u32>,
    pub subpixel: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct IdBuffers<'a> {
    pub ids: &'a wgpu::Buffer,
    pub draws: &'a wgpu::Buffer,
    pub key: u64,
}

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

struct Groups {
    seed: [Option<wgpu::BindGroup>; 2],
    args: wgpu::BindGroup,
    traverse: wgpu::BindGroup,
    select: wgpu::BindGroup,
    bin_args: wgpu::BindGroup,
    retest: wgpu::BindGroup,
}

type GroupsKey = (u64, u64, u64, u64, Option<u64>, Option<u64>);
type ShadowGroupKey = (u64, u64, wgpu::Buffer);

pub(crate) struct VirtualGeometryPass {
    arena: VirtualArena,
    storage: SelectionStorage,
    readbacks: Readbacks,
    module: Option<wgpu::ShaderModule>,
    pipelines: Option<Pipelines>,
    groups: [Cached<GroupsKey, Groups>; 2],
    parity: usize,
    poses: Option<(wgpu::Buffer, u64)>,
    shadow_drawing: Option<ShadowDrawing>,
    shadow_groups: [Cached<ShadowGroupKey, wgpu::BindGroup>; 2],
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

    pub(crate) fn insert(&mut self, handle: MeshHandle, geometry: &Arc<VirtualMesh>) {
        self.arena.insert(handle, geometry);
    }

    pub(crate) fn retain(&mut self, keep: impl Fn(MeshHandle) -> bool) {
        self.arena.retain(keep);
    }

    pub(crate) fn mesh_index(&self, handle: MeshHandle) -> Option<u32> {
        self.arena.mesh_index(handle)
    }

    pub(crate) fn closed(&self, index: u32) -> bool {
        self.arena.closed(index)
    }

    pub(crate) fn upload(&mut self, device: &wgpu::Device, queue: &Queue, budget: Option<u32>) {
        if self.arena.upload(device, queue, budget) {
            self.groups = Default::default();
        }
    }

    pub(crate) fn arena(&self) -> Option<([&wgpu::Buffer; 3], u64)> {
        self.arena.vertex_buffers()
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.arena.capacity_bytes() + self.storage.capacity_bytes()
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
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

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
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
        if !storage.retesting() {
            self.read_counters(device, encoder);
        }
    }

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

    pub(crate) fn visible(&self) -> (&wgpu::Buffer, u64) {
        (&self.storage.visible, self.storage.generation())
    }

    pub(crate) fn bin_base(&self, bin: usize) -> u32 {
        self.storage.bin_base(bin)
    }

    pub(crate) fn bin_source(&self, bin: usize) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        self.storage.bin_source(bin)
    }

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

    pub(crate) fn page_failure(&self) -> Option<&str> {
        self.arena.page_failure()
    }

    pub(crate) fn stats(&self) -> Option<VirtualStats> {
        self.arena.buffers().map(|_| {
            let [
                resident_pages,
                loading_pages,
                missing_pages,
                dropped_pages,
                failed_pages,
            ] = self.arena.pages();
            VirtualStats {
                resident_pages,
                loading_pages,
                missing_pages,
                dropped_pages,
                failed_pages,
                ..self.readbacks.last().unwrap_or_default()
            }
        })
    }

    pub(crate) fn after_submit(&mut self) {
        self.readbacks.after_submit(self.storage.retest_capacity());
    }
}

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
