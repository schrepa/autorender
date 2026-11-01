#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Streams {
    Plain = 0,
    Extended = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub first_index: u32,
    pub index_count: u32,
    pub streams: Streams,
    pub base_vertex: u32,
    pub vertex_count: u32,
}

#[derive(Debug, Default)]
struct Space {
    capacity: u32,
    free: Vec<(u32, u32)>,
}

impl Space {
    fn allocate(&mut self, length: u32) -> Option<u32> {
        let slot = self.free.iter().position(|&(_, free)| free >= length)?;
        let (start, free) = self.free[slot];
        if free == length {
            self.free.remove(slot);
        } else {
            self.free[slot] = (start + length, free - length);
        }
        Some(start)
    }

    fn release(&mut self, start: u32, length: u32) {
        if length == 0 {
            return;
        }
        let slot = self.free.partition_point(|&(free, _)| free < start);
        self.free.insert(slot, (start, length));
        if slot + 1 < self.free.len() {
            let (next_start, next_length) = self.free[slot + 1];
            if start + length == next_start {
                self.free[slot].1 += next_length;
                self.free.remove(slot + 1);
            }
        }
        if slot > 0 {
            let (previous_start, previous_length) = self.free[slot - 1];
            if previous_start + previous_length == start {
                self.free[slot - 1].1 += self.free[slot].1;
                self.free.remove(slot);
            }
        }
    }

    fn grow(&mut self, capacity: u32) {
        let old = self.capacity;
        self.capacity = capacity;
        self.release(old, capacity - old);
    }
}

fn grown_capacity(space: &Space, length: u32, bytes: u64, limit: u64) -> Option<u32> {
    let needed = u64::from(space.capacity) + u64::from(length);
    let grown = (u64::from(space.capacity) * 2).max(needed).max(1024);
    let fits = (limit / bytes).min(u64::from(u32::MAX));
    let capacity = grown.min(fits);
    (capacity >= needed).then_some(capacity as u32)
}

pub use gpu::*;

mod gpu {
    use std::collections::BTreeMap;

    use engine_core::{Mesh, MeshHandle, MeshLods};

    use super::*;
    use crate::gpu_calls::Queue;
    use crate::gpu_data::{ExtraVertex, LOD_LEVELS, Vertex};
    use crate::memory::buffer_init;

    const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        crate::pipelines::vertex_attributes!(Vertex {
            0 => position as Float32x3,
            1 => normal as Snorm16x2,
            7 => uv as Float32x2,
        });
    const EXTRA_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        crate::pipelines::vertex_attributes!(ExtraVertex {
            9 => color as Unorm16x4,
            10 => uv1 as Float32x2,
        });

    pub fn extra_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<ExtraVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &EXTRA_ATTRIBUTES,
        })
    }

    pub fn vertex_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        })
    }

    #[derive(Clone, Copy)]
    pub struct MeshDraw<'a> {
        pub vertices: &'a wgpu::Buffer,
        pub extras: Option<&'a wgpu::Buffer>,
        pub indices: &'a wgpu::Buffer,
        pub first_index: u32,
        pub base_vertex: i32,
        pub index_count: u32,
    }

    impl MeshDraw<'_> {
        pub fn index_range(&self) -> std::ops::Range<u32> {
            self.first_index..self.first_index + self.index_count
        }
    }

    pub enum Ids<'a> {
        Whole(&'a wgpu::Buffer),
        Region(wgpu::BufferSlice<'a>),
    }

    #[derive(Default)]
    pub struct Bindings {
        vertex: [Option<*const wgpu::Buffer>; 4],
        index: Option<*const wgpu::Buffer>,
        groups: [Option<*const wgpu::BindGroup>; 4],
    }

    impl Bindings {
        pub fn vertex(
            &mut self,
            pass: &mut wgpu::RenderPass<'_>,
            slot: u32,
            buffer: &wgpu::Buffer,
        ) {
            let bound = &mut self.vertex[slot as usize];
            if *bound != Some(buffer as *const _) {
                pass.set_vertex_buffer(slot, buffer.slice(..));
                *bound = Some(buffer as *const _);
            }
        }

        pub fn index(&mut self, pass: &mut wgpu::RenderPass<'_>, buffer: &wgpu::Buffer) {
            if self.index != Some(buffer as *const _) {
                pass.set_index_buffer(buffer.slice(..), wgpu::IndexFormat::Uint32);
                self.index = Some(buffer as *const _);
            }
        }

        pub fn ids(&mut self, pass: &mut wgpu::RenderPass<'_>, slot: u32, ids: Ids<'_>) {
            match ids {
                Ids::Whole(buffer) => self.vertex(pass, slot, buffer),
                Ids::Region(region) => {
                    pass.set_vertex_buffer(slot, region);
                    self.vertex[slot as usize] = None;
                }
            }
        }

        pub fn group(
            &mut self,
            pass: &mut wgpu::RenderPass<'_>,
            index: u32,
            group: &wgpu::BindGroup,
        ) {
            let bound = &mut self.groups[index as usize];
            if *bound != Some(group as *const _) {
                pass.set_bind_group(index, group, &[]);
                *bound = Some(group as *const _);
            }
        }
    }

    struct Class {
        buffers: Vec<wgpu::Buffer>,
        strides: Vec<u64>,
        space: Space,
    }

    pub struct MeshArena {
        classes: [Class; 2],
        indices: wgpu::Buffer,
        index_space: Space,
    }

    const VERTEX_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::VERTEX
        .union(wgpu::BufferUsages::COPY_DST)
        .union(wgpu::BufferUsages::COPY_SRC);
    const INDEX_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::INDEX
        .union(wgpu::BufferUsages::COPY_DST)
        .union(wgpu::BufferUsages::COPY_SRC);
    const INDEX_BYTES: u64 = 4;

    fn buffer(
        device: &wgpu::Device,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        crate::memory::buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4).next_multiple_of(4),
                usage,
                mapped_at_creation: false,
            },
        )
    }

    fn regrown(
        device: &wgpu::Device,
        queue: &Queue,
        old: &wgpu::Buffer,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let grown = buffer(device, "engine/mesh-arena", size, usage);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/mesh-arena-growth"),
        });
        encoder.copy_buffer_to_buffer(old, 0, &grown, 0, old.size().min(size));
        queue.submit_now([encoder.finish()]);
        grown
    }

    impl MeshArena {
        pub fn new(device: &wgpu::Device, [vertex, extra]: [u64; 2]) -> Self {
            let class = |strides: Vec<u64>| Class {
                buffers: strides
                    .iter()
                    .map(|_| buffer(device, "engine/mesh-arena", 0, VERTEX_USAGE))
                    .collect(),
                strides,
                space: Space::default(),
            };
            Self {
                classes: [class(vec![vertex]), class(vec![vertex, extra])],
                indices: buffer(device, "engine/mesh-arena-indices", 0, INDEX_USAGE),
                index_space: Space::default(),
            }
        }

        pub fn capacity_bytes(&self) -> u64 {
            self.indices.size()
                + self
                    .classes
                    .iter()
                    .flat_map(|class| &class.buffers)
                    .map(wgpu::Buffer::size)
                    .sum::<u64>()
        }

        pub fn insert(
            &mut self,
            device: &wgpu::Device,
            queue: &Queue,
            streams: &[&[u8]],
            vertex_count: u32,
            indices: &[u32],
        ) -> Option<Placement> {
            let class = match streams.len() {
                1 => Streams::Plain,
                2 => Streams::Extended,
                _ => return None,
            };
            let index_count = u32::try_from(indices.len()).ok()?;
            let limit = device.limits().max_buffer_size;
            let first_index = match self.index_space.allocate(index_count) {
                Some(start) => start,
                None => {
                    let grown = grown_capacity(&self.index_space, index_count, INDEX_BYTES, limit)?;
                    self.indices = regrown(
                        device,
                        queue,
                        &self.indices,
                        u64::from(grown) * INDEX_BYTES,
                        INDEX_USAGE,
                    );
                    self.index_space.grow(grown);
                    self.index_space.allocate(index_count)?
                }
            };
            let arena = &mut self.classes[class as usize];
            let base_vertex = match arena.space.allocate(vertex_count) {
                Some(start) => Some(start),
                None => {
                    let widest = arena.strides.iter().copied().max().unwrap_or(1);
                    grown_capacity(&arena.space, vertex_count, widest, limit).and_then(|grown| {
                        for (buffer, stride) in arena.buffers.iter_mut().zip(&arena.strides) {
                            *buffer = regrown(
                                device,
                                queue,
                                buffer,
                                u64::from(grown) * stride,
                                VERTEX_USAGE,
                            );
                        }
                        arena.space.grow(grown);
                        arena.space.allocate(vertex_count)
                    })
                }
            };
            let Some(base_vertex) = base_vertex else {
                self.index_space.release(first_index, index_count);
                return None;
            };
            for ((buffer, stride), bytes) in arena.buffers.iter().zip(&arena.strides).zip(streams) {
                if !bytes.is_empty() {
                    queue.write_buffer(buffer, u64::from(base_vertex) * stride, bytes);
                }
            }
            if !indices.is_empty() {
                queue.write_buffer(
                    &self.indices,
                    u64::from(first_index) * INDEX_BYTES,
                    bytemuck::cast_slice(indices),
                );
            }
            Some(Placement {
                first_index,
                index_count,
                streams: class,
                base_vertex,
                vertex_count,
            })
        }

        pub fn remove(&mut self, placement: Placement) {
            self.index_space
                .release(placement.first_index, placement.index_count);
            self.classes[placement.streams as usize]
                .space
                .release(placement.base_vertex, placement.vertex_count);
        }

        pub fn draw(&self, placement: Placement) -> MeshDraw<'_> {
            let buffers = &self.classes[placement.streams as usize].buffers;
            MeshDraw {
                vertices: &buffers[0],
                extras: buffers.get(1),
                indices: &self.indices,
                first_index: placement.first_index,
                base_vertex: placement.base_vertex as i32,
                index_count: placement.index_count,
            }
        }
    }

    pub struct GpuMesh {
        placement: Result<Placement, OwnMesh>,
        pub extended: bool,
        pub index_count: u32,
        pub byte_len: u64,
        pub lods: Vec<(u32, u32, f32)>,
        pub plan: Vec<(u32, u32, f32)>,
        pub casters: Vec<(u32, u32)>,
        pub without_levels: bool,
    }

    struct OwnMesh {
        vertices: wgpu::Buffer,
        extras: Option<wgpu::Buffer>,
        indices: wgpu::Buffer,
    }

    impl GpuMesh {
        pub fn draw<'a>(&'a self, arena: &'a MeshArena) -> MeshDraw<'a> {
            match &self.placement {
                Ok(placement) => MeshDraw {
                    index_count: self.index_count,
                    ..arena.draw(*placement)
                },
                Err(own) => MeshDraw {
                    vertices: &own.vertices,
                    extras: own.extras.as_ref(),
                    indices: &own.indices,
                    first_index: 0,
                    base_vertex: 0,
                    index_count: self.index_count,
                },
            }
        }

        pub fn caster_range(&self, level: usize, own: (u32, u32)) -> (u32, u32) {
            self.casters.get(level).copied().unwrap_or(own)
        }
    }

    pub fn lod_levels(mesh: &GpuMesh) -> usize {
        1 + mesh.plan.len().min(LOD_LEVELS - 1)
    }

    fn plan_levels(lods: &[(u32, u32, f32)]) -> Vec<(u32, u32, f32)> {
        let first = lods.first().copied().filter(|&(_, count, _)| count > 0);
        let empty = lods.last().copied().filter(|&(_, count, _)| count == 0);
        first.into_iter().chain(empty).collect()
    }

    type WithLevels = (Vec<u32>, Vec<(u32, u32, f32)>, Vec<(u32, u32)>);

    fn with_levels(mesh: &Mesh, lods: Option<&MeshLods>) -> WithLevels {
        let mut indices =
            Vec::with_capacity(mesh.indices().len() + lods.map_or(0, |lods| lods.indices.len()));
        indices.extend_from_slice(mesh.indices());
        let Some(lods) = lods else {
            return (indices, Vec::new(), Vec::new());
        };
        indices.extend_from_slice(&lods.indices);
        let levels = lods
            .levels
            .iter()
            .map(|level| (level.first_index, level.index_count, level.error))
            .collect();
        (indices, levels, lods.casters.clone())
    }

    pub struct GpuMeshes {
        meshes: BTreeMap<MeshHandle, GpuMesh>,
        arena: MeshArena,
        layout: u64,
        staging: Vec<Vertex>,
        residency: engine_registry::Residency,
    }

    impl GpuMeshes {
        pub fn new(device: &wgpu::Device) -> Self {
            Self {
                meshes: BTreeMap::new(),
                arena: MeshArena::new(
                    device,
                    [size_of::<Vertex>(), size_of::<ExtraVertex>()].map(|bytes| bytes as u64),
                ),
                layout: 0,
                staging: Vec::new(),
                residency: engine_registry::Residency::default(),
            }
        }

        pub fn get(&self, handle: MeshHandle) -> Option<&GpuMesh> {
            self.meshes.get(&handle)
        }

        pub fn contains(&self, handle: MeshHandle) -> bool {
            self.meshes.contains_key(&handle)
        }

        pub fn draw(&self, handle: MeshHandle) -> Option<MeshDraw<'_>> {
            self.meshes
                .get(&handle)
                .map(|mesh| mesh.draw(&self.arena))
        }

        pub fn arena(&self) -> &MeshArena {
            &self.arena
        }

        pub fn len(&self) -> usize {
            self.meshes.len()
        }

        pub fn is_empty(&self) -> bool {
            self.meshes.is_empty()
        }

        pub fn iter(&self) -> impl Iterator<Item = (MeshHandle, &GpuMesh)> {
            self.meshes.iter().map(|(&handle, mesh)| (handle, mesh))
        }

        pub fn layout(&self) -> u64 {
            self.layout
        }

        pub fn residency(&self) -> &engine_registry::Residency {
            &self.residency
        }

        pub fn capacity_bytes(&self) -> u64 {
            self.arena.capacity_bytes()
        }

        pub fn retain(&mut self, mut keep: impl FnMut(MeshHandle) -> bool) {
            let (arena, residency) = (&mut self.arena, &mut self.residency);
            let resident = self.meshes.len();
            self.meshes.retain(|handle, mesh| {
                let kept = keep(*handle);
                if !kept {
                    if let Ok(placement) = mesh.placement {
                        arena.remove(placement);
                    }
                    residency.remove(*handle);
                }
                kept
            });
            if self.meshes.len() != resident {
                self.layout += 1;
            }
        }

        pub fn upload(
            &mut self,
            device: &wgpu::Device,
            queue: &Queue,
            handle: MeshHandle,
            source: &Mesh,
            lods: Option<&MeshLods>,
        ) -> u64 {
            let std::collections::btree_map::Entry::Vacant(entry) = self.meshes.entry(handle)
            else {
                return 0;
            };
            self.staging.clear();
            self.staging.reserve_exact(source.vertices().len());
            self.staging
                .extend(source.vertices().iter().map(Vertex::from));
            let extras = ExtraVertex::stream(source);
            let mut streams: Vec<&[u8]> = vec![bytemuck::cast_slice(&self.staging)];
            if let Some(extras) = &extras {
                streams.push(bytemuck::cast_slice(extras));
            }
            let (indices, levels, casters) = with_levels(source, lods);
            let index_count = source.indices().len() as u32;
            let without_levels = lods.is_none();
            let placement = self
                .arena
                .insert(device, queue, &streams, self.staging.len() as u32, &indices)
                .ok_or_else(|| {
                    let buffer = |label, contents: &[u8], usage| {
                        buffer_init(
                            device,
                            &wgpu::util::BufferInitDescriptor {
                                label: Some(label),
                                contents,
                                usage,
                            },
                        )
                    };
                    let vertex = wgpu::BufferUsages::VERTEX;
                    OwnMesh {
                        vertices: buffer("engine/mesh-vertices", streams[0], vertex),
                        extras: extras.as_ref().map(|stream| {
                            buffer("engine/mesh-extras", bytemuck::cast_slice(stream), vertex)
                        }),
                        indices: buffer(
                            "engine/mesh-indices",
                            bytemuck::cast_slice(&indices),
                            wgpu::BufferUsages::INDEX,
                        ),
                    }
                });
            self.layout += 1;
            let byte_len = source.byte_len() as u64;
            entry.insert(GpuMesh {
                placement,
                extended: extras.is_some(),
                index_count,
                byte_len,
                plan: plan_levels(&levels),
                lods: levels,
                casters,
                without_levels,
            });
            self.residency.insert(handle);
            byte_len
        }
    }
}
