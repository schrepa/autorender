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
    use super::*;
    use crate::gpu_calls::Queue;

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
}
