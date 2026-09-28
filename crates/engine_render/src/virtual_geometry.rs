//! Virtual geometry (decision 044): resident virtual meshes' hierarchies,
//! clusters and pages in one arena, and each frame's selection of the
//! clusters the visible instances of their draw groups draw
//! (`virtual_geometry.wgsl`), into per-material bins the opaque pass draws
//! 128 triangles a cluster from (`meshes.wgsl`'s `vs_virtual`).
//!
//! Arena records carry absolute indices, rebased once at upload; pages
//! upload as the file stores them. Selection runs on the GPU only; each
//! bin's region and the node and group queues grow to what frames read back.

use std::{cell::Cell, collections::HashMap, rc::Rc, sync::Arc};

use engine_core::glam::Vec3;
use engine_core::{CLUSTER_WORDS, GROUP_WORDS, MeshHandle, NODE_WORDS, VirtualMesh};

use crate::gpu_data::VIRTUAL_SHADER;

const NONE: u32 = u32::MAX;
const GROUP: u32 = 64;
/// Counters before the bins' (`virtual_geometry.wgsl`).
const COUNTERS: usize = 8;
const READBACKS: usize = 3;
/// Starting capacities, grown from read-back counts.
const START_QUEUE: u64 = 1 << 16;
const START_BIN: u64 = 1 << 14;
/// Bytes of an entry of the selected clusters: visible ID and cluster.
pub(crate) const VISIBLE_BYTES: u64 = 8;

/// A batch of instances of a virtual mesh with one material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VirtualBatch {
    /// Start of its visible IDs (GPU-presented frames) or slot IDs
    /// (CPU-prepared ones).
    pub start: u32,
    /// Instances of CPU-prepared frames; GPU-presented ones read the count
    /// of plan batch `draw`'s arguments.
    pub count: Option<u32>,
    pub draw: u32,
    pub mesh: u32,
    pub bin: u32,
    pub double_sided: bool,
}

/// The view clusters are selected for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualView {
    /// Inward frustum planes, normalized.
    pub planes: [[f32; 4]; 6],
    pub eye: Vec3,
    pub near: f32,
    /// cot(fovy / 2) / 2.
    pub projection: f32,
    /// The largest projected error drawn, as a fraction of the viewport
    /// height.
    pub threshold: f32,
}

/// Counts a frame read back: clusters and triangles selected, and entries
/// dropped for want of room.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VirtualStats {
    pub clusters: u32,
    pub triangles: u32,
    pub dropped: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadbackState {
    Idle,
    Recorded,
    Pending,
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Rc<Cell<ReadbackState>>,
}

struct Resident {
    handle: MeshHandle,
    geometry: Arc<VirtualMesh>,
}

struct ArenaBuffers {
    meshes: wgpu::Buffer,
    nodes: wgpu::Buffer,
    groups: wgpu::Buffer,
    clusters: wgpu::Buffer,
    pages: wgpu::Buffer,
}

struct Pipelines {
    seed: wgpu::ComputePipeline,
    level_args: [wgpu::ComputePipeline; 2],
    traverse: [wgpu::ComputePipeline; 2],
    work_args: wgpu::ComputePipeline,
    select: wgpu::ComputePipeline,
    bin_args: wgpu::ComputePipeline,
}

struct Groups {
    seed: wgpu::BindGroup,
    level_args: [wgpu::BindGroup; 2],
    traverse: [wgpu::BindGroup; 2],
    work_args: wgpu::BindGroup,
    select: wgpu::BindGroup,
    bin_args: wgpu::BindGroup,
}

/// Generations of the arena, frame buffers, poses, IDs and draw arguments
/// bind groups were built for.
type GroupsKey = (u64, u64, u64, u64, u64);
/// Counts a frame read back and its dropped queue, group and cluster
/// entries.
type ReadCounts = (VirtualStats, [u32; 3]);

pub(crate) struct VirtualGeometryPass {
    pipelines: Pipelines,
    resident: Vec<Resident>,
    index: HashMap<MeshHandle, u32>,
    dirty: bool,
    arena: Option<ArenaBuffers>,
    // Advances whenever arena buffers are replaced.
    arena_generation: u64,
    // Nodes on the longest root-to-leaf path of any resident mesh.
    depth: u32,
    uniform: wgpu::Buffer,
    batches: wgpu::Buffer,
    batch_capacity: u64,
    bins: wgpu::Buffer,
    // Each bin's base and capacity, as uploaded.
    bin_regions: Vec<(u64, u64)>,
    bin_capacity: u64,
    queues: [wgpu::Buffer; 2],
    work: wgpu::Buffer,
    queue_capacity: u64,
    visible: wgpu::Buffer,
    visible_capacity: u64,
    counters: wgpu::Buffer,
    counter_capacity: u64,
    dispatch: wgpu::Buffer,
    draw_args: wgpu::Buffer,
    // Advances whenever frame buffers are replaced.
    frame_generation: u64,
    groups: Option<(GroupsKey, Groups)>,
    batch_count: u32,
    max_instances: u32,
    readbacks: Vec<Readback>,
    // Counts and dropped queue, group and cluster entries mapped since the
    // last frame took them.
    read: Rc<Cell<Option<ReadCounts>>>,
    last: Option<VirtualStats>,
    grow: [bool; 3],
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

fn storage_init(device: &wgpu::Device, label: &str, words: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let mut padded;
    let contents: &[u32] = if words.len() < 4 {
        padded = words.to_vec();
        padded.resize(4, 0);
        &padded
    } else {
        words
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(contents),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl VirtualGeometryPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/virtual_geometry.wgsl"),
            source: wgpu::ShaderSource::Wgsl(VIRTUAL_SHADER.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            seed: pipeline("seed"),
            level_args: [pipeline("level_args_even"), pipeline("level_args_odd")],
            traverse: [pipeline("traverse_even"), pipeline("traverse_odd")],
            work_args: pipeline("work_args"),
            select: pipeline("select_clusters"),
            bin_args: pipeline("bin_args"),
        };
        let none = wgpu::BufferUsages::empty();
        Self {
            pipelines,
            resident: Vec::new(),
            index: HashMap::new(),
            dirty: false,
            arena: None,
            arena_generation: 0,
            depth: 0,
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/virtual-frame"),
                size: 144,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            batches: storage(device, "engine/virtual-batches", 32, none),
            batch_capacity: 1,
            bins: storage(device, "engine/virtual-bins", 16, none),
            bin_regions: Vec::new(),
            bin_capacity: START_BIN,
            queues: [
                storage(device, "engine/virtual-queue-a", START_QUEUE * 16, none),
                storage(device, "engine/virtual-queue-b", START_QUEUE * 16, none),
            ],
            work: storage(device, "engine/virtual-work", START_QUEUE * 16, none),
            queue_capacity: START_QUEUE,
            visible: storage(
                device,
                "engine/virtual-visible",
                VISIBLE_BYTES,
                wgpu::BufferUsages::VERTEX,
            ),
            visible_capacity: 1,
            counters: storage(
                device,
                "engine/virtual-counters",
                COUNTERS as u64 * 4,
                wgpu::BufferUsages::COPY_SRC,
            ),
            counter_capacity: COUNTERS as u64,
            dispatch: storage(
                device,
                "engine/virtual-dispatch",
                24,
                wgpu::BufferUsages::INDIRECT,
            ),
            draw_args: storage(
                device,
                "engine/virtual-draws",
                16,
                wgpu::BufferUsages::INDIRECT,
            ),
            frame_generation: 0,
            groups: None,
            batch_count: 0,
            max_instances: 0,
            readbacks: Vec::new(),
            read: Rc::new(Cell::new(None)),
            last: None,
            grow: [false; 3],
        }
    }

    /// Makes `geometry` resident as mesh `handle`'s, once.
    pub(crate) fn insert(&mut self, handle: MeshHandle, geometry: &Arc<VirtualMesh>) {
        if !self.index.contains_key(&handle)
            && !self
                .resident
                .iter()
                .any(|resident| resident.handle == handle)
        {
            self.resident.push(Resident {
                handle,
                geometry: Arc::clone(geometry),
            });
            self.dirty = true;
        }
    }

    /// Releases every resident mesh `keep` rejects.
    pub(crate) fn retain(&mut self, keep: impl Fn(MeshHandle) -> bool) {
        let before = self.resident.len();
        self.resident.retain(|resident| keep(resident.handle));
        if self.resident.len() != before {
            self.dirty = true;
        }
    }

    /// The arena index of mesh `handle`'s geometry, once uploaded.
    pub(crate) fn mesh_index(&self, handle: MeshHandle) -> Option<u32> {
        self.index.get(&handle).copied()
    }

    /// Rebuilds the arena when resident meshes changed: every record
    /// rebased to absolute indices, every page copied.
    pub(crate) fn upload(&mut self, device: &wgpu::Device) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        self.index.clear();
        self.depth = 0;
        let (mut meshes, mut nodes, mut groups, mut clusters, mut pages) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (mesh_index, resident) in self.resident.iter().enumerate() {
            let geometry = &resident.geometry;
            self.index.insert(resident.handle, mesh_index as u32);
            let node_base = (nodes.len() / NODE_WORDS) as u32;
            let group_base = (groups.len() / GROUP_WORDS) as u32;
            let cluster_base = (clusters.len() / CLUSTER_WORDS) as u32;
            let page_base = pages.len() as u32;
            let (position_min, position_step, uv_min, uv_step) = geometry.quantization();
            meshes.extend([node_base, group_base, cluster_base, page_base]);
            meshes.extend([geometry.levels(), geometry.depth(), 0, 0]);
            meshes.extend(position_min.map(f32::to_bits));
            meshes.push(0);
            meshes.extend(position_step.map(f32::to_bits));
            meshes.push(0);
            meshes.extend([uv_min[0], uv_min[1], uv_step[0], uv_step[1]].map(f32::to_bits));
            for words in geometry.node_section().as_chunks::<NODE_WORDS>().0 {
                let mut node = *words;
                if node[5] != NONE {
                    node[5] += group_base;
                } else {
                    node[6] += node_base;
                }
                nodes.extend(node);
            }
            for words in geometry.group_section().as_chunks::<GROUP_WORDS>().0 {
                let mut group = *words;
                group[5] += cluster_base;
                groups.extend(group);
            }
            let table = geometry.page_table();
            for words in geometry.cluster_section().as_chunks::<CLUSTER_WORDS>().0 {
                let page = page_base + table[words[7] as usize * 2];
                let refined = if words[5] == NONE {
                    NONE
                } else {
                    words[5] + group_base
                };
                clusters.extend([
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                    words[4],
                    refined,
                    words[6] + group_base,
                    mesh_index as u32,
                    page + words[8],
                    page + words[9],
                    words[10],
                    0,
                ]);
            }
            pages.extend_from_slice(geometry.page_section());
            self.depth = self.depth.max(geometry.depth());
        }
        self.arena = Some(ArenaBuffers {
            meshes: storage_init(device, "engine/virtual-meshes", &meshes),
            nodes: storage_init(device, "engine/virtual-nodes", &nodes),
            groups: storage_init(device, "engine/virtual-groups", &groups),
            clusters: storage_init(device, "engine/virtual-clusters", &clusters),
            pages: storage_init(device, "engine/virtual-pages", &pages),
        });
        self.arena_generation += 1;
        self.groups = None;
    }

    /// The arena buffers vertex shaders read (meshes, clusters, pages) and
    /// their generation.
    pub(crate) fn arena(&self) -> Option<([&wgpu::Buffer; 3], u64)> {
        self.arena.as_ref().map(|arena| {
            (
                [&arena.meshes, &arena.clusters, &arena.pages],
                self.arena_generation,
            )
        })
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        let arena = self.arena.as_ref().map_or(0, |arena| {
            arena.meshes.size()
                + arena.nodes.size()
                + arena.groups.size()
                + arena.clusters.size()
                + arena.pages.size()
        });
        arena
            + self.queues[0].size()
            + self.queues[1].size()
            + self.work.size()
            + self.visible.size()
            + self.counters.size()
    }

    /// Uploads this frame's batches, bins and view, growing the queues and
    /// bin regions to what frames read back needed.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batches: &[VirtualBatch],
        bins: usize,
        view: &VirtualView,
    ) {
        if let Some((stats, [queue_drops, work_drops, cluster_drops])) = self.read.take() {
            self.last = Some(stats);
            self.grow = [queue_drops > 0, work_drops > 0, cluster_drops > 0];
        }
        let binding = device.limits().max_storage_buffer_binding_size;
        if (self.grow[0] || self.grow[1]) && self.queue_capacity * 32 <= binding {
            self.queue_capacity *= 2;
            let none = wgpu::BufferUsages::empty();
            self.queues = [
                storage(
                    device,
                    "engine/virtual-queue-a",
                    self.queue_capacity * 16,
                    none,
                ),
                storage(
                    device,
                    "engine/virtual-queue-b",
                    self.queue_capacity * 16,
                    none,
                ),
            ];
            self.work = storage(
                device,
                "engine/virtual-work",
                self.queue_capacity * 16,
                none,
            );
            self.frame_generation += 1;
        }
        if self.grow[2] && self.bin_capacity * VISIBLE_BYTES * 2 * bins as u64 <= binding {
            self.bin_capacity *= 2;
        }
        self.grow = [false; 3];
        // Bins' regions: equal shares of the visible storage.
        let wanted = self.bin_capacity * bins.max(1) as u64;
        if wanted > self.visible_capacity {
            self.visible_capacity = wanted;
            self.visible = storage(
                device,
                "engine/virtual-visible",
                wanted * VISIBLE_BYTES,
                wgpu::BufferUsages::VERTEX,
            );
            self.frame_generation += 1;
        }
        self.bin_regions = (0..bins)
            .map(|bin| (bin as u64 * self.bin_capacity, self.bin_capacity))
            .collect();
        let bin_words: Vec<u32> = self
            .bin_regions
            .iter()
            .flat_map(|&(base, capacity)| [base as u32, capacity as u32, 0, 0])
            .collect();
        if bins as u64 > self.counter_capacity - COUNTERS as u64
            || bin_words.len() * 4 > self.bins.size() as usize
        {
            let capacity = (bins.max(1) as u64).next_power_of_two();
            self.counter_capacity = COUNTERS as u64 + capacity;
            self.counters = storage(
                device,
                "engine/virtual-counters",
                self.counter_capacity * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            self.bins = storage(
                device,
                "engine/virtual-bins",
                capacity * 16,
                wgpu::BufferUsages::empty(),
            );
            self.draw_args = storage(
                device,
                "engine/virtual-draws",
                capacity * 16,
                wgpu::BufferUsages::INDIRECT,
            );
            self.readbacks.clear();
            self.frame_generation += 1;
        }
        if !bin_words.is_empty() {
            queue.write_buffer(&self.bins, 0, bytemuck::cast_slice(&bin_words));
        }
        if batches.len() as u64 > self.batch_capacity {
            self.batch_capacity = (batches.len() as u64).next_power_of_two();
            self.batches = storage(
                device,
                "engine/virtual-batches",
                self.batch_capacity * 32,
                wgpu::BufferUsages::empty(),
            );
            self.frame_generation += 1;
        }
        let batch_words: Vec<u32> = batches
            .iter()
            .flat_map(|batch| {
                [
                    batch.start,
                    batch.count.unwrap_or(NONE),
                    batch.draw,
                    batch.mesh,
                    batch.bin,
                    u32::from(batch.double_sided),
                    0,
                    0,
                ]
            })
            .collect();
        if !batch_words.is_empty() {
            queue.write_buffer(&self.batches, 0, bytemuck::cast_slice(&batch_words));
        }
        self.batch_count = batches.len() as u32;
        let mut uniform = Vec::with_capacity(36);
        for plane in view.planes {
            uniform.extend(plane);
        }
        uniform.extend([view.eye.x, view.eye.y, view.eye.z, view.near]);
        uniform.extend([view.projection, view.threshold, 0.0, 0.0]);
        let counts = [
            batches.len() as u32,
            self.queue_capacity as u32,
            self.queue_capacity as u32,
            bins as u32,
        ];
        let mut bytes: Vec<u8> = bytemuck::cast_slice(&uniform).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(&counts));
        queue.write_buffer(&self.uniform, 0, &bytes);
    }

    /// Binds this frame's slot poses, visible IDs (or slot IDs) and draw
    /// arguments, each with a key that changes when the buffer does, and
    /// the most instances any batch has.
    pub(crate) fn bind(
        &mut self,
        device: &wgpu::Device,
        poses: (&wgpu::Buffer, u64),
        ids: (&wgpu::Buffer, u64),
        draws: (&wgpu::Buffer, u64),
        max_instances: u32,
    ) {
        self.max_instances = max_instances;
        let Some(arena) = &self.arena else {
            return;
        };
        let key = (
            self.arena_generation,
            self.frame_generation,
            poses.1,
            ids.1,
            draws.1,
        );
        if self.groups.as_ref().is_some_and(|(built, _)| *built == key) {
            return;
        }
        let group = |pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry<'_>]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/virtual-group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })
        };
        let pipelines = &self.pipelines;
        let [queue_a, queue_b] = &self.queues;
        let level = |pipeline| {
            group(
                pipeline,
                &[
                    entry(0, &self.uniform),
                    entry(12, &self.counters),
                    entry(15, &self.dispatch),
                ],
            )
        };
        let traverse = |pipeline| {
            group(
                pipeline,
                &[
                    entry(0, &self.uniform),
                    entry(3, &arena.nodes),
                    entry(6, poses.0),
                    entry(9, queue_a),
                    entry(10, queue_b),
                    entry(11, &self.work),
                    entry(12, &self.counters),
                ],
            )
        };
        let groups = Groups {
            seed: group(
                &pipelines.seed,
                &[
                    entry(0, &self.uniform),
                    entry(1, &self.batches),
                    entry(2, &arena.meshes),
                    entry(7, ids.0),
                    entry(8, draws.0),
                    entry(9, queue_a),
                    entry(10, queue_b),
                    entry(12, &self.counters),
                ],
            ),
            level_args: [
                level(&pipelines.level_args[0]),
                level(&pipelines.level_args[1]),
            ],
            traverse: [
                traverse(&pipelines.traverse[0]),
                traverse(&pipelines.traverse[1]),
            ],
            work_args: level(&pipelines.work_args),
            select: group(
                &pipelines.select,
                &[
                    entry(0, &self.uniform),
                    entry(4, &arena.groups),
                    entry(5, &arena.clusters),
                    entry(6, poses.0),
                    entry(11, &self.work),
                    entry(12, &self.counters),
                    entry(13, &self.visible),
                    entry(14, &self.bins),
                ],
            ),
            bin_args: group(
                &pipelines.bin_args,
                &[
                    entry(0, &self.uniform),
                    entry(12, &self.counters),
                    entry(14, &self.bins),
                    entry(16, &self.draw_args),
                ],
            ),
        };
        self.groups = Some((key, groups));
    }

    /// Records this frame's selection: seeds, one traversal per hierarchy
    /// level, the groups' clusters and each bin's draw arguments, then the
    /// counters' copy for read-back. Nothing without bound batches.
    pub(crate) fn encode(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let Some((_, groups)) = &self.groups else {
            return;
        };
        if self.batch_count == 0 {
            return;
        }
        encoder.clear_buffer(&self.counters, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/virtual-selection"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.seed);
            pass.set_bind_group(0, &groups.seed, &[]);
            pass.dispatch_workgroups(
                self.max_instances.div_ceil(GROUP).max(1),
                self.batch_count,
                1,
            );
            for level in 0..self.depth as usize {
                let parity = level % 2;
                pass.set_pipeline(&self.pipelines.level_args[parity]);
                pass.set_bind_group(0, &groups.level_args[parity], &[]);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&self.pipelines.traverse[parity]);
                pass.set_bind_group(0, &groups.traverse[parity], &[]);
                pass.dispatch_workgroups_indirect(&self.dispatch, 0);
            }
            pass.set_pipeline(&self.pipelines.work_args);
            pass.set_bind_group(0, &groups.work_args, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.pipelines.select);
            pass.set_bind_group(0, &groups.select, &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch, 12);
            pass.set_pipeline(&self.pipelines.bin_args);
            pass.set_bind_group(0, &groups.bin_args, &[]);
            pass.dispatch_workgroups((self.bin_regions.len() as u32).div_ceil(GROUP).max(1), 1, 1);
        }
        if let Some(index) = self.idle_readback(device) {
            let readback = &self.readbacks[index];
            encoder.copy_buffer_to_buffer(
                &self.counters,
                0,
                &readback.buffer,
                0,
                COUNTERS as u64 * 4,
            );
            readback.state.set(ReadbackState::Recorded);
        }
    }

    /// The selected clusters of bin `bin` (an instance-stepped vertex stream
    /// of visible ID and cluster) and its indirect draw arguments.
    pub(crate) fn bin_source(&self, bin: usize) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        let (base, capacity) = self.bin_regions[bin];
        (
            self.visible
                .slice(base * VISIBLE_BYTES..(base + capacity) * VISIBLE_BYTES),
            &self.draw_args,
            bin as u64 * 16,
        )
    }

    /// Counts the last read-back frame selected.
    pub(crate) fn stats(&self) -> Option<VirtualStats> {
        self.last
    }

    fn idle_readback(&mut self, device: &wgpu::Device) -> Option<usize> {
        if let Some(index) = self
            .readbacks
            .iter()
            .position(|readback| readback.state.get() == ReadbackState::Idle)
        {
            return Some(index);
        }
        if self.readbacks.len() == READBACKS {
            return None;
        }
        self.readbacks.push(Readback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/virtual-counters-readback"),
                size: COUNTERS as u64 * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Rc::new(Cell::new(ReadbackState::Idle)),
        });
        Some(self.readbacks.len() - 1)
    }

    /// Requests maps of counters recorded in the frame just submitted.
    pub(crate) fn after_submit(&mut self) {
        for readback in &self.readbacks {
            if readback.state.get() != ReadbackState::Recorded {
                continue;
            }
            readback.state.set(ReadbackState::Pending);
            let state = Rc::clone(&readback.state);
            let read = Rc::clone(&self.read);
            let buffer = readback.buffer.clone();
            readback
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            let word = |index: usize| {
                                view.get(index * 4..index * 4 + 4).map_or(0, |bytes| {
                                    u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
                                })
                            };
                            let drops = [word(3), word(4), word(5)];
                            read.set(Some((
                                VirtualStats {
                                    clusters: word(6),
                                    triangles: word(7),
                                    dropped: drops.iter().sum(),
                                },
                                drops,
                            )));
                        }
                        buffer.unmap();
                    }
                    state.set(ReadbackState::Idle);
                });
        }
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
