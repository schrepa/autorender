use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::collections::{Cursor, Since};
use engine_core::{
    Asset, Assets, Error, ErrorCode, Expected, IndexFormat, MeshFile, VertexFormat, VertexStream,
};

use crate::gpu_calls::Queue;

const fn format_of(stored: VertexFormat) -> wgpu::VertexFormat {
    match stored {
        VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        VertexFormat::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
        VertexFormat::Unorm16x4 => wgpu::VertexFormat::Unorm16x4,
        VertexFormat::Uint32x4 => wgpu::VertexFormat::Uint32x4,
    }
}

pub(crate) fn geometry_attributes(stream: &VertexStream) -> Vec<wgpu::VertexAttribute> {
    stream
        .inputs
        .iter()
        .map(|input| wgpu::VertexAttribute {
            format: format_of(input.format),
            offset: u64::from(input.offset),
            shader_location: input.location,
        })
        .collect()
}

pub(crate) fn layout(
    path: String,
    declared: &[VertexStream],
    held: &[VertexStream],
) -> Result<Box<[u32]>, Error> {
    declared
        .iter()
        .map(|stream| held.iter().position(|file| file == stream).map(|at| at as u32))
        .collect::<Option<_>>()
        .ok_or_else(|| {
            Error::new(ErrorCode::InvalidRenderable, path)
                .expected(Expected::Matching)
                .got(serde_json::json!({
                    "declared": format!("{declared:?}"),
                    "file": format!("{held:?}"),
                }))
        })
}

#[derive(Clone, Copy)]
pub(crate) struct MeshDraw<'a> {
    pub(crate) class: u32,
    pub(crate) streams: &'a [VertexStream],
    pub(crate) buffers: &'a [wgpu::Buffer],
    pub(crate) indices: &'a wgpu::Buffer,
    pub(crate) index_format: wgpu::IndexFormat,
    pub(crate) first_index: u32,
    pub(crate) base_vertex: i32,
    pub(crate) index_count: u32,
}

#[derive(Default)]
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

const VERTEX_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::VERTEX
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);
const INDEX_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::INDEX
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    crate::gpu_calls::create_buffer(device, &wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4).next_multiple_of(4),
        usage,
        mapped_at_creation: false,
    })
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

fn write_aligned(queue: &Queue, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
    let whole = bytes.len() / 4 * 4;
    if whole > 0 {
        queue.write_buffer(buffer, offset, &bytes[..whole]);
    }
    if whole < bytes.len() {
        let mut tail = [0; 4];
        tail[..bytes.len() - whole].copy_from_slice(&bytes[whole..]);
        queue.write_buffer(buffer, offset + whole as u64, &tail);
    }
}

struct Shelf {
    space: Space,
    buffers: Vec<wgpu::Buffer>,
    units: Vec<u64>,
    usage: wgpu::BufferUsages,
}

impl Shelf {
    fn new(device: &wgpu::Device, units: Vec<u64>, usage: wgpu::BufferUsages) -> Self {
        Self {
            space: Space::default(),
            buffers: units
                .iter()
                .map(|_| buffer(device, "engine/mesh-arena", 0, usage))
                .collect(),
            units,
            usage,
        }
    }

    fn take(&mut self, length: u32, limit: u64) -> Option<u32> {
        if let Some(start) = self.space.allocate(length) {
            return Some(start);
        }
        let widest = self.units.iter().copied().max()?;
        let needed = u64::from(self.space.capacity) + u64::from(length);
        let fits = (limit / widest).min(u64::from(u32::MAX));
        let capacity = (u64::from(self.space.capacity) * 2).max(needed).min(fits);
        if capacity < needed {
            return None;
        }
        self.space.grow(capacity as u32);
        self.space.allocate(length)
    }

    fn fit(&mut self, device: &wgpu::Device, queue: &Queue) {
        for (held, unit) in self.buffers.iter_mut().zip(&self.units) {
            let size = u64::from(self.space.capacity) * unit;
            if held.size() < size {
                *held = regrown(device, queue, held, size, self.usage);
            }
        }
    }

    fn write(&self, queue: &Queue, start: u32, bytes: impl Iterator<Item = impl AsRef<[u8]>>) {
        for ((held, unit), bytes) in self.buffers.iter().zip(&self.units).zip(bytes) {
            write_aligned(queue, held, u64::from(start) * unit, bytes.as_ref());
        }
    }
}

struct Class {
    streams: Vec<VertexStream>,
    shelf: Shelf,
}

enum Placed {
    Arena {
        base_vertex: u32,
        vertex_count: u32,
        first_index: u32,
        index_span: u32,
    },
    Own {
        buffers: Vec<wgpu::Buffer>,
        indices: wgpu::Buffer,
    },
}

struct GpuMesh {
    class: u32,
    index_format: IndexFormat,
    index_count: u32,
    bytes: u64,
    placed: Placed,
}

pub(crate) struct GpuMeshes {
    meshes: BTreeMap<Asset<MeshFile>, GpuMesh>,
    classes: Vec<Class>,
    indices: [Shelf; 2],
    cursor: Cursor,
}

const fn shelf_of(format: IndexFormat) -> usize {
    match format {
        IndexFormat::Uint16 => 0,
        IndexFormat::Uint32 => 1,
    }
}

impl GpuMeshes {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let indices = |format: IndexFormat| {
            Shelf::new(device, vec![u64::from(format.bytes())], INDEX_USAGE)
        };
        Self {
            meshes: BTreeMap::new(),
            classes: Vec::new(),
            indices: [indices(IndexFormat::Uint16), indices(IndexFormat::Uint32)],
            cursor: Cursor::default(),
        }
    }

    pub(crate) fn contains(&self, handle: Asset<MeshFile>) -> bool {
        self.meshes.contains_key(&handle)
    }

    #[cfg_attr(
        not(feature = "dev"),
        expect(dead_code, reason = "read by the gpu.assets names")
    )]
    pub(crate) fn len(&self) -> usize {
        self.meshes.len()
    }

    #[cfg_attr(
        not(feature = "dev"),
        expect(dead_code, reason = "read by the gpu.assets names")
    )]
    pub(crate) fn bytes(&self) -> u64 {
        self.meshes.values().map(|mesh| mesh.bytes).sum()
    }

    pub(crate) fn draw(&self, handle: Asset<MeshFile>) -> Option<MeshDraw<'_>> {
        let mesh = self.meshes.get(&handle)?;
        let class = self.classes.get(mesh.class as usize)?;
        let (buffers, indices, first_index, base_vertex) = match &mesh.placed {
            Placed::Arena {
                base_vertex,
                first_index,
                ..
            } => (
                &class.shelf.buffers,
                &self.indices[shelf_of(mesh.index_format)].buffers[0],
                *first_index,
                *base_vertex as i32,
            ),
            Placed::Own { buffers, indices } => (buffers, indices, 0, 0),
        };
        Some(MeshDraw {
            class: mesh.class,
            streams: &class.streams,
            buffers,
            indices,
            index_format: match mesh.index_format {
                IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
            },
            first_index,
            base_vertex,
            index_count: mesh.index_count,
        })
    }

    pub(crate) fn sync(&mut self, device: &wgpu::Device, queue: &Queue, world: &World) -> u64 {
        let Some(assets) = world.get_resource::<Assets>() else {
            return 0;
        };
        let mut listed: Vec<Asset<MeshFile>> = match assets.changed(&mut self.cursor) {
            Since::Same => return 0,
            Since::Listed(ids) => ids
                .iter()
                .filter_map(|&id| assets.asset::<MeshFile>(id))
                .collect(),
            Since::All => self
                .meshes
                .keys()
                .copied()
                .chain(assets.each::<MeshFile>().map(|(handle, _)| handle))
                .collect(),
        };
        listed.sort_unstable();
        listed.dedup();
        let limit = device.limits().max_buffer_size;
        let mut placed = Vec::with_capacity(listed.len());
        for handle in listed {
            if let Some(mesh) = self.meshes.remove(&handle) {
                self.release(&mesh);
            }
            if let Ok(file) = assets.get(handle) {
                placed.push((handle, file, self.place(device, file, limit)));
            }
        }
        for shelf in self
            .classes
            .iter_mut()
            .map(|class| &mut class.shelf)
            .chain(&mut self.indices)
        {
            shelf.fit(device, queue);
        }
        let mut bytes = 0;
        for (handle, file, mesh) in placed {
            if let Placed::Arena {
                base_vertex,
                first_index,
                ..
            } = mesh.placed
            {
                let streams = (0..file.description().streams.len()).map(|at| file.stream(at));
                self.classes[mesh.class as usize]
                    .shelf
                    .write(queue, base_vertex, streams);
                self.indices[shelf_of(mesh.index_format)].write(
                    queue,
                    first_index,
                    std::iter::once(file.indices()),
                );
            }
            bytes += mesh.bytes;
            self.meshes.insert(handle, mesh);
        }
        bytes
    }

    fn class_of(&mut self, device: &wgpu::Device, streams: &[VertexStream]) -> u32 {
        if let Some(at) = self.classes.iter().position(|class| class.streams == streams) {
            return at as u32;
        }
        let units = streams.iter().map(|stream| u64::from(stream.stride)).collect();
        self.classes.push(Class {
            streams: streams.to_vec(),
            shelf: Shelf::new(device, units, VERTEX_USAGE),
        });
        (self.classes.len() - 1) as u32
    }

    fn place(
        &mut self,
        device: &wgpu::Device,
        file: &MeshFile,
        limit: u64,
    ) -> GpuMesh {
        let description = file.description();
        let class = self.class_of(device, &description.streams);
        let format = description.index_format;
        let (vertex_count, index_count) = (description.vertex_count, description.index_count);
        let index_span = index_count.next_multiple_of(4 / format.bytes());
        let indices = &mut self.indices[shelf_of(format)];
        let vertices = &mut self.classes[class as usize].shelf;
        let arena = indices.take(index_span, limit).and_then(|first_index| {
            let Some(base_vertex) = vertices.take(vertex_count, limit) else {
                indices.space.release(first_index, index_span);
                return None;
            };
            Some(Placed::Arena {
                base_vertex,
                vertex_count,
                first_index,
                index_span,
            })
        });
        let placed = arena.unwrap_or_else(|| {
            let made = |label, contents: &[u8], usage| {
                crate::gpu_calls::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
            };
            let vertex = wgpu::BufferUsages::VERTEX;
            Placed::Own {
                buffers: (0..description.streams.len())
                    .map(|at| made("engine/mesh-vertices", file.stream(at), vertex))
                    .collect(),
                indices: made("engine/mesh-indices", file.indices(), wgpu::BufferUsages::INDEX),
            }
        });
        let strides: u64 = description
            .streams
            .iter()
            .map(|stream| u64::from(stream.stride))
            .sum();
        GpuMesh {
            class,
            index_format: format,
            index_count,
            bytes: u64::from(vertex_count) * strides
                + u64::from(index_count) * u64::from(format.bytes()),
            placed,
        }
    }

    fn release(&mut self, mesh: &GpuMesh) {
        let Placed::Arena {
            base_vertex,
            vertex_count,
            first_index,
            index_span,
        } = mesh.placed
        else {
            return;
        };
        self.indices[shelf_of(mesh.index_format)]
            .space
            .release(first_index, index_span);
        if let Some(class) = self.classes.get_mut(mesh.class as usize) {
            class.shelf.space.release(base_vertex, vertex_count);
        }
    }
}
