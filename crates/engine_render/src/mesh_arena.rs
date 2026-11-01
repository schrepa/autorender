use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::{
    ArenaShape, AssetLibrary, AssetResidency, Mesh, MeshHandle, MeshLods, MeshPlacement,
    MeshStreams, ResidentMesh,
};

use crate::gpu_calls::Queue;
use crate::gpu_data::{ExtraVertex, Vertex};
use crate::memory::buffer_init;

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = crate::pipelines::vertex_attributes!(Vertex {
    0 => position as Float32x3,
    1 => normal as Snorm16x2,
    7 => uv as Float32x2,
});
const EXTRA_ATTRIBUTES: [wgpu::VertexAttribute; 2] = crate::pipelines::vertex_attributes!(ExtraVertex {
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

pub(crate) fn arena_shape(device: &wgpu::Device) -> ArenaShape {
    ArenaShape {
        vertex: size_of::<Vertex>() as u64,
        extra: size_of::<ExtraVertex>() as u64,
        limit: device.limits().max_buffer_size,
    }
}

pub(crate) fn configure(
    device: &wgpu::Device,
    world: &mut World,
    (meshes, materials, impostors): (
        &mut GpuMeshes,
        &mut crate::materials::MaterialStore,
        &mut crate::impostors::Impostors,
    ),
) {
    meshes.synced = None;
    materials.forget_sync();
    impostors.forget();
    engine_core::configure_assets(
        world,
        arena_shape(device),
        crate::impostors::atlas_device(device),
    );
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

struct Class {
    buffers: Vec<wgpu::Buffer>,
    strides: Vec<u64>,
    capacity: u32,
}

struct MeshArena {
    classes: [Class; 2],
    indices: wgpu::Buffer,
    index_capacity: u32,
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
    fn new(device: &wgpu::Device, [vertex, extra]: [u64; 2]) -> Self {
        let class = |strides: Vec<u64>| Class {
            buffers: strides
                .iter()
                .map(|_| buffer(device, "engine/mesh-arena", 0, VERTEX_USAGE))
                .collect(),
            strides,
            capacity: 0,
        };
        Self {
            classes: [class(vec![vertex]), class(vec![vertex, extra])],
            indices: buffer(device, "engine/mesh-arena-indices", 0, INDEX_USAGE),
            index_capacity: 0,
        }
    }

    fn capacity_bytes(&self) -> u64 {
        self.indices.size()
            + self
                .classes
                .iter()
                .flat_map(|class| &class.buffers)
                .map(wgpu::Buffer::size)
                .sum::<u64>()
    }

    fn fit(&mut self, device: &wgpu::Device, queue: &Queue, vertices: [u32; 2], indices: u32) {
        if indices > self.index_capacity {
            self.indices = regrown(
                device,
                queue,
                &self.indices,
                u64::from(indices) * INDEX_BYTES,
                INDEX_USAGE,
            );
            self.index_capacity = indices;
        }
        for (class, capacity) in self.classes.iter_mut().zip(vertices) {
            if capacity <= class.capacity {
                continue;
            }
            for (buffer, stride) in class.buffers.iter_mut().zip(&class.strides) {
                *buffer = regrown(
                    device,
                    queue,
                    buffer,
                    u64::from(capacity) * stride,
                    VERTEX_USAGE,
                );
            }
            class.capacity = capacity;
        }
    }

    fn write(&self, queue: &Queue, placement: MeshPlacement, streams: &[&[u8]], indices: &[u32]) {
        let class = &self.classes[placement.streams as usize];
        for ((buffer, stride), bytes) in class.buffers.iter().zip(&class.strides).zip(streams) {
            if !bytes.is_empty() {
                queue.write_buffer(buffer, u64::from(placement.base_vertex) * stride, bytes);
            }
        }
        if !indices.is_empty() {
            queue.write_buffer(
                &self.indices,
                u64::from(placement.first_index) * INDEX_BYTES,
                bytemuck::cast_slice(indices),
            );
        }
    }

    fn draw(&self, placement: MeshPlacement) -> MeshDraw<'_> {
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

struct OwnMesh {
    vertices: wgpu::Buffer,
    extras: Option<wgpu::Buffer>,
    indices: wgpu::Buffer,
}

pub struct GpuMesh {
    placement: Option<MeshPlacement>,
    own: Option<OwnMesh>,
    pub extended: bool,
    pub index_count: u32,
    pub byte_len: u64,
}

impl GpuMesh {
    fn draw<'a>(&'a self, arena: &'a MeshArena) -> Option<MeshDraw<'a>> {
        match (&self.placement, &self.own) {
            (Some(placement), _) => Some(MeshDraw {
                index_count: self.index_count,
                ..arena.draw(*placement)
            }),
            (None, Some(own)) => Some(MeshDraw {
                vertices: &own.vertices,
                extras: own.extras.as_ref(),
                indices: &own.indices,
                first_index: 0,
                base_vertex: 0,
                index_count: self.index_count,
            }),
            (None, None) => None,
        }
    }
}

fn with_levels(mesh: &Mesh, lods: Option<&MeshLods>) -> Vec<u32> {
    let mut indices =
        Vec::with_capacity(mesh.indices().len() + lods.map_or(0, |lods| lods.indices.len()));
    indices.extend_from_slice(mesh.indices());
    if let Some(lods) = lods {
        indices.extend_from_slice(&lods.indices);
    }
    indices
}

pub struct GpuMeshes {
    meshes: BTreeMap<MeshHandle, GpuMesh>,
    arena: MeshArena,
    staging: Vec<Vertex>,
    synced: Option<u64>,
}

impl GpuMeshes {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            meshes: BTreeMap::new(),
            arena: MeshArena::new(
                device,
                [size_of::<Vertex>(), size_of::<ExtraVertex>()].map(|bytes| bytes as u64),
            ),
            staging: Vec::new(),
            synced: None,
        }
    }

    pub fn get(&self, handle: MeshHandle) -> Option<&GpuMesh> {
        self.meshes.get(&handle)
    }

    pub fn contains(&self, handle: MeshHandle) -> bool {
        self.meshes.contains_key(&handle)
    }

    pub fn draw(&self, handle: MeshHandle) -> Option<MeshDraw<'_>> {
        self.meshes.get(&handle)?.draw(&self.arena)
    }

    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.arena.capacity_bytes()
    }

    pub fn layout(&self) -> u64 {
        self.synced.unwrap_or(0)
    }

    pub fn sync(&mut self, device: &wgpu::Device, queue: &Queue, world: &World) -> u64 {
        let (Some(residency), Some(library)) = (
            world.get_resource::<AssetResidency>(),
            world.get_resource::<AssetLibrary>(),
        ) else {
            return 0;
        };
        if self.synced == Some(residency.layout()) {
            return 0;
        }
        let Some((vertices, indices)) = residency.capacities() else {
            return 0;
        };
        self.meshes.retain(|&handle, mesh| {
            residency
                .mesh(handle)
                .is_some_and(|resident| resident.placement() == mesh.placement)
        });
        self.arena.fit(device, queue, vertices, indices);
        let mut bytes = 0;
        for (handle, resident) in residency.meshes() {
            if self.meshes.contains_key(&handle) {
                continue;
            }
            let Some(source) = library.mesh(handle) else {
                continue;
            };
            let lods = library.mesh_lods(handle).flatten().map(|lods| &**lods);
            bytes += self.upload(device, queue, (handle, resident), source, lods);
        }
        self.synced = Some(residency.layout());
        bytes
    }

    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        (handle, resident): (MeshHandle, &ResidentMesh),
        source: &Mesh,
        lods: Option<&MeshLods>,
    ) -> u64 {
        self.staging.clear();
        self.staging.reserve_exact(source.vertices().len());
        self.staging
            .extend(source.vertices().iter().map(Vertex::from));
        let extras = ExtraVertex::stream(source);
        let mut streams: Vec<&[u8]> = vec![bytemuck::cast_slice(&self.staging)];
        if let Some(extras) = &extras {
            streams.push(bytemuck::cast_slice(extras));
        }
        let indices = with_levels(source, lods);
        let placement = resident.placement();
        let own = match placement {
            Some(placement) => {
                debug_assert_eq!(
                    placement.streams == MeshStreams::Extended,
                    extras.is_some(),
                    "resident_mesh_streams_match"
                );
                self.arena.write(queue, placement, &streams, &indices);
                None
            }
            None => {
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
                Some(OwnMesh {
                    vertices: buffer("engine/mesh-vertices", streams[0], vertex),
                    extras: extras.as_ref().map(|stream| {
                        buffer("engine/mesh-extras", bytemuck::cast_slice(stream), vertex)
                    }),
                    indices: buffer(
                        "engine/mesh-indices",
                        bytemuck::cast_slice(&indices),
                        wgpu::BufferUsages::INDEX,
                    ),
                })
            }
        };
        let byte_len = resident.byte_len();
        self.meshes.insert(
            handle,
            GpuMesh {
                placement,
                own,
                extended: extras.is_some(),
                index_count: resident.index_count(),
                byte_len,
            },
        );
        byte_len
    }
}
