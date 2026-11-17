use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::{
    ArenaShape, AssetLibrary, AssetResidency, Attribute, AttributeFormat, ClassStream, Cursor,
    GeometryClass, GeometryHandle, GeometryStream, Mesh, MeshLods, MeshPlacement, MeshResidence,
    ResidentMesh, Since, math::clamp,
};
use wgpu::util::DeviceExt as _;

use crate::gpu_calls::Queue;

const fn format_of(stored: AttributeFormat) -> wgpu::VertexFormat {
    match stored {
        AttributeFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        AttributeFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        AttributeFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        AttributeFormat::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
        AttributeFormat::Unorm16x4 => wgpu::VertexFormat::Unorm16x4,
        AttributeFormat::Uint32x4 => wgpu::VertexFormat::Uint32x4,
    }
}

pub(crate) fn geometry_attributes(stream: GeometryStream) -> Vec<wgpu::VertexAttribute> {
    stream
        .inputs()
        .into_iter()
        .map(|(location, stored, offset)| wgpu::VertexAttribute {
            format: format_of(stored),
            offset,
            shader_location: location,
        })
        .collect()
}

fn octahedral(normal: [f32; 3]) -> [i16; 2] {
    let [x, y, z] = normal;
    let length = x.abs() + y.abs() + z.abs();
    let (u, v) = (x / length, y / length);
    let sign = |value: f32| if value >= 0.0 { 1.0 } else { -1.0 };
    let (u, v) = if z < 0.0 {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    } else {
        (u, v)
    };
    [u, v].map(|value| (clamp(value, -1.0, 1.0) * 32767.0).round() as i16)
}

fn packed(mesh: &Mesh, index: usize, attribute: Attribute, bytes: &mut Vec<u8>) {
    let vertex = &mesh.vertices()[index];
    let floats = |bytes: &mut Vec<u8>, values: &[f32]| {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    };
    match attribute {
        Attribute::Position => floats(bytes, &vertex.position),
        Attribute::Normal => {
            for value in octahedral(vertex.normal) {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        Attribute::Uv0 => floats(bytes, &vertex.uv),
        Attribute::Uv1 => floats(bytes, &mesh.uv1().map_or(vertex.uv, |uv1| uv1[index])),
        Attribute::Color => {
            let color = mesh.colors().map_or([1.0; 4], |colors| colors[index]);
            for value in color {
                let unit = (clamp(value, 0.0, 1.0) * 65535.0).round() as u16;
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
        }
        Attribute::Tangent => {}
    }
}

fn stream_bytes(mesh: &Mesh, stream: &ClassStream, bytes: &mut Vec<u8>) {
    bytes.clear();
    let vertices = mesh.vertices().len();
    bytes.reserve(vertices * stream.stride() as usize);
    let fixed = match stream {
        ClassStream::Vertex => GeometryStream::Vertex,
        ClassStream::Extra => GeometryStream::Extra,
        ClassStream::Named(name, format) => {
            let held = mesh
                .streams()
                .iter()
                .find(|held| held.name() == name && held.format() == *format);
            match held {
                Some(held) => bytes.extend_from_slice(held.bytes()),
                None => bytes.resize(vertices * format.bytes() as usize, 0),
            }
            return;
        }
    };
    for index in 0..vertices {
        for (attribute, _, _) in fixed.placed() {
            packed(mesh, index, attribute, bytes);
        }
    }
}

pub(crate) fn arena_shape(device: &wgpu::Device) -> ArenaShape {
    ArenaShape {
        limit: device.limits().max_buffer_size,
    }
}

pub(crate) fn configure(
    device: &wgpu::Device,
    world: &mut World,
    (meshes, materials): (&mut GpuMeshes, &mut crate::materials::MaterialStore),
) {
    meshes.placed = engine_core::Cursor::default();
    materials.forget_sync();
    engine_core::configure_assets(world, arena_shape(device));
}

#[derive(Clone, Copy)]
pub struct MeshDraw<'a> {
    pub streams: &'a [ClassStream],
    pub buffers: &'a [wgpu::Buffer],
    pub indices: &'a wgpu::Buffer,
    pub first_index: u32,
    pub base_vertex: i32,
    pub index_count: u32,
}

impl<'a> MeshDraw<'a> {
    pub fn stream(&self, stream: GeometryStream) -> Option<&'a wgpu::Buffer> {
        let at = self.streams.iter().position(|held| held.serves(stream))?;
        self.buffers.get(at)
    }

    pub fn holds(&self, streams: &[GeometryStream]) -> bool {
        streams.iter().all(|&stream| self.stream(stream).is_some())
    }
}

struct Class {
    class: GeometryClass,
    buffers: Vec<wgpu::Buffer>,
    strides: Vec<u64>,
    capacity: u32,
}

impl Class {
    fn new(device: &wgpu::Device, class: &GeometryClass) -> Self {
        let strides: Vec<u64> = class.streams.iter().map(ClassStream::stride).collect();
        Self {
            class: class.clone(),
            buffers: strides
                .iter()
                .map(|_| buffer(device, "engine/mesh-arena", 0, VERTEX_USAGE))
                .collect(),
            strides,
            capacity: 0,
        }
    }
}

struct MeshArena {
    classes: Vec<Class>,
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
    device.create_buffer(&wgpu::BufferDescriptor {
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

impl MeshArena {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            classes: Vec::new(),
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

    fn fit(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        classes: &[(&GeometryClass, u32)],
        indices: u32,
    ) -> Vec<u32> {
        let mut replaced: Vec<u32> = (classes.len()..self.classes.len())
            .map(|at| at as u32)
            .collect();
        self.classes.truncate(classes.len());
        for (at, &(class, _)) in classes.iter().enumerate() {
            match self.classes.get_mut(at) {
                Some(held) if held.class == *class => {}
                Some(held) => {
                    *held = Class::new(device, class);
                    replaced.push(at as u32);
                }
                None => self.classes.push(Class::new(device, class)),
            }
        }
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
        for (class, &(_, capacity)) in self.classes.iter_mut().zip(classes) {
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
        replaced
    }

    fn write(&self, queue: &Queue, placement: MeshPlacement, streams: &[Vec<u8>], indices: &[u32]) {
        let Some(class) = self.classes.get(placement.class as usize) else {
            return;
        };
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

    fn draw(&self, placement: MeshPlacement) -> Option<MeshDraw<'_>> {
        let class = self.classes.get(placement.class as usize)?;
        Some(MeshDraw {
            streams: &class.class.streams,
            buffers: &class.buffers,
            indices: &self.indices,
            first_index: placement.first_index,
            base_vertex: placement.base_vertex as i32,
            index_count: placement.index_count,
        })
    }
}

struct OwnMesh {
    streams: Vec<ClassStream>,
    buffers: Vec<wgpu::Buffer>,
    indices: wgpu::Buffer,
}

pub struct GpuMesh {
    placement: MeshResidence,
    revision: u64,
    own: Option<OwnMesh>,
    pub index_count: u32,
}

impl GpuMesh {
    fn in_class(&self, classes: &[u32]) -> bool {
        match self.placement {
            MeshResidence::Arena(placement) => classes.contains(&placement.class),
            MeshResidence::Own => false,
        }
    }

    fn draw<'a>(&'a self, arena: &'a MeshArena) -> Option<MeshDraw<'a>> {
        match (self.placement, &self.own) {
            (MeshResidence::Arena(placement), _) => Some(MeshDraw {
                index_count: self.index_count,
                ..arena.draw(placement)?
            }),
            (MeshResidence::Own, Some(own)) => Some(MeshDraw {
                streams: &own.streams,
                buffers: &own.buffers,
                indices: &own.indices,
                first_index: 0,
                base_vertex: 0,
                index_count: self.index_count,
            }),
            (MeshResidence::Own, None) => None,
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
    meshes: BTreeMap<GeometryHandle, GpuMesh>,
    arena: MeshArena,
    staging: Vec<Vec<u8>>,
    placed: Cursor,
}

impl GpuMeshes {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            meshes: BTreeMap::new(),
            arena: MeshArena::new(device),
            staging: Vec::new(),
            placed: Cursor::default(),
        }
    }

    pub fn get(&self, handle: GeometryHandle) -> Option<&GpuMesh> {
        self.meshes.get(&handle)
    }

    pub fn contains(&self, handle: GeometryHandle) -> bool {
        self.meshes.contains_key(&handle)
    }

    pub fn draw(&self, handle: GeometryHandle) -> Option<MeshDraw<'_>> {
        self.meshes.get(&handle)?.draw(&self.arena)
    }

    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.arena.capacity_bytes()
    }

    pub(crate) fn staging_bytes(&self) -> u64 {
        self.staging.iter().map(Vec::capacity).sum::<usize>() as u64
    }

    pub fn sync(&mut self, device: &wgpu::Device, queue: &Queue, world: &World) -> u64 {
        let (Some(residency), Some(library)) = (
            world.get_resource::<AssetResidency>(),
            world.get_resource::<AssetLibrary>(),
        ) else {
            return 0;
        };
        let Some((classes, indices)) = residency.capacities() else {
            return 0;
        };
        let replaced = self.arena.fit(device, queue, &classes, indices);
        if !replaced.is_empty() {
            self.meshes.retain(|_, mesh| !mesh.in_class(&replaced));
            self.placed = Cursor::default();
        }
        let all: Vec<GeometryHandle>;
        let changed = match residency.placed().since(&mut self.placed) {
            Since::Same => return 0,
            Since::Listed(handles) => handles,
            Since::All => {
                self.meshes.retain(|&handle, mesh| {
                    residency.mesh(handle).is_some_and(|resident| {
                        (resident.placement(), resident.revision())
                            == (mesh.placement, mesh.revision)
                    })
                });
                all = residency.meshes().map(|(handle, _)| handle).collect();
                all.as_slice()
            }
        };
        let mut bytes = 0;
        for &handle in changed {
            let resident = residency.mesh(handle);
            let current = self
                .meshes
                .get(&handle)
                .map(|mesh| (mesh.placement, mesh.revision));
            if resident.is_some_and(|resident| {
                current == Some((resident.placement(), resident.revision()))
            }) {
                continue;
            }
            self.meshes.remove(&handle);
            let (Some(resident), Some(source)) = (resident, library.mesh(handle)) else {
                continue;
            };
            let Some(class) = residency.class(resident.class()) else {
                continue;
            };
            let lods = library.mesh_lods(handle).flatten().map(|lods| &**lods);
            bytes += self.upload(device, queue, (handle, resident, class), source, lods);
        }
        bytes
    }

    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        (handle, resident, class): (GeometryHandle, &ResidentMesh, &GeometryClass),
        source: &Mesh,
        lods: Option<&MeshLods>,
    ) -> u64 {
        let streams = &class.streams;
        self.staging
            .resize_with(streams.len().max(self.staging.len()), Vec::new);
        for (stream, bytes) in streams.iter().zip(&mut self.staging) {
            stream_bytes(source, stream, bytes);
        }
        let staged = &self.staging[..streams.len()];
        let indices = with_levels(source, lods);
        let placement = resident.placement();
        let own = match placement {
            MeshResidence::Arena(placement) => {
                self.arena.write(queue, placement, staged, &indices);
                None
            }
            MeshResidence::Own => {
                let buffer = |label, contents: &[u8], usage| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents,
                        usage,
                    })
                };
                let vertex = wgpu::BufferUsages::VERTEX;
                Some(OwnMesh {
                    streams: streams.clone(),
                    buffers: staged
                        .iter()
                        .map(|bytes| buffer("engine/mesh-vertices", bytes, vertex))
                        .collect(),
                    indices: buffer(
                        "engine/mesh-indices",
                        bytemuck::cast_slice(&indices),
                        wgpu::BufferUsages::INDEX,
                    ),
                })
            }
        };
        self.meshes.insert(
            handle,
            GpuMesh {
                placement,
                revision: resident.revision(),
                own,
                index_count: resident.index_count(),
            },
        );
        resident.byte_len()
    }
}
