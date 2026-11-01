use std::any::TypeId;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;

use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, MaterialHandle, MeshHandle, TextureHandle};

use crate::graph::{Bits, Clear, Load, Ops, Planned};
use crate::install::{Bound, Program, Scoped, Work};
use crate::machine::{CAMERA_STRIDE, DEBUG_FORMATS, DEPTH_FORMAT, Machine, Target, set_formats};
use crate::registration::Installed;
use crate::{AfterRead, CameraUniform, DepthUse, Lifetime, VertexLayout, ViewTarget};

pub(crate) type FrameCore = Run;
pub(crate) type EncodeCore = Run;

pub(crate) struct Allocated {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) size: wgpu::Extent3d,
    pub(crate) mips: u32,
    pub(crate) slices: Vec<(Range<u32>, Range<u32>, wgpu::TextureView)>,
}

impl Allocated {
    fn new(device: &wgpu::Device, label: &'static str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages, size: wgpu::Extent3d, mips: u32) -> Self {
        let texture = crate::memory::texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: mips.max(1),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: (size.depth_or_array_layers > 1).then_some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            texture,
            view,
            size,
            mips: mips.max(1),
            slices: Vec::new(),
        }
    }

    fn slice(&mut self, mips: &Range<u32>, layers: &Range<u32>, dimension: Option<wgpu::TextureViewDimension>) -> &wgpu::TextureView {
        let at = match self.slices.iter().position(|(known_mips, known_layers, _)| known_mips == mips && known_layers == layers) {
            Some(at) => at,
            None => {
                let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mips.start,
                    mip_level_count: Some(mips.end.saturating_sub(mips.start).max(1)),
                    base_array_layer: layers.start,
                    array_layer_count: Some(layers.end.saturating_sub(layers.start).max(1)),
                    dimension,
                    ..Default::default()
                });
                self.slices.push((mips.clone(), layers.clone(), view));
                self.slices.len() - 1
            }
        };
        &self.slices[at].2
    }
}

pub(crate) struct Asked {
    pub(crate) part: u32,
    pub(crate) tag: u32,
    pub(crate) buffer: u32,
    pub(crate) range: Range<u64>,
    pub(crate) then: AfterRead,
    pub(crate) deliver: fn(&mut World, u64, u32, &[u8]),
}

pub(crate) struct Ring {
    pub(crate) part: u32,
    pub(crate) tag: u32,
    pub(crate) slots: Vec<(wgpu::Buffer, u64, std::sync::Arc<std::sync::atomic::AtomicU8>, Option<(u64, fn(&mut World, u64, u32, &[u8]))>)>,
}

pub(crate) struct Filled {
    pub(crate) base: u32,
    pub(crate) textures: Vec<TextureHandle>,
    pub(crate) generation: u64,
    pub(crate) used: u64,
    pub(crate) groups: Vec<(u32, wgpu::BindGroup)>,
}

#[derive(Default)]
pub(crate) struct Set {
    pipeline: Option<(u32, u64)>,
    frame: Option<(u32, u32)>,
    groups: Vec<(u32, u32, usize)>,
}

pub(crate) struct Open {
    pub(crate) pass: wgpu::RenderPass<'static>,
    pub(crate) attachments: Vec<u64>,
    pub(crate) formats: Vec<Option<wgpu::TextureFormat>>,
    pub(crate) depth: Option<(wgpu::TextureFormat, bool)>,
    pub(crate) debug: bool,
    pub(crate) offset: u32,
    pub(crate) shared: bool,
    pub(crate) set: Set,
}

pub(crate) struct Assets {
    pub(crate) meshes: std::collections::BTreeMap<MeshHandle, crate::renderer::GpuMesh>,
    pub(crate) arena: crate::mesh_arena::MeshArena,
    pub(crate) materials: crate::materials::MaterialStore,
}

pub(crate) struct Run {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) machine: Machine,
    pub(crate) parts: Vec<Option<Box<dyn Installed>>>,
    pub(crate) live: Vec<bool>,
    pub(crate) faulted: Vec<Option<Error>>,
    pub(crate) part: u32,
    pub(crate) pass: u32,
    pub(crate) frame: u64,
    pub(crate) parity: usize,
    pub(crate) generation: u64,
    pub(crate) textures: Vec<[Option<Allocated>; 2]>,
    pub(crate) sizes: Vec<Option<(wgpu::Extent3d, u32)>>,
    pub(crate) buffers: Vec<Option<(wgpu::Buffer, u64)>>,
    pub(crate) samplers: Vec<wgpu::Sampler>,
    pub(crate) targets: Vec<(Target, Allocated)>,
    pub(crate) debug_targets: Vec<Allocated>,
    pub(crate) canvas: Option<wgpu::TextureView>,
    pub(crate) runs: Bits,
    pub(crate) asked: Vec<Asked>,
    pub(crate) copies: Vec<(u32, Vec<crate::CopyRegion>)>,
    pub(crate) rings: Vec<Ring>,
    pub(crate) cameras: Vec<CameraUniform>,
    pub(crate) views: Vec<(u32, u32, u32)>,
    pub(crate) camera: wgpu::Buffer,
    pub(crate) pieces: wgpu::Buffer,
    pub(crate) pieces_written: Vec<u8>,
    pub(crate) frame_groups: Vec<([Option<wgpu::BindGroup>; 2], u64)>,
    pub(crate) own_groups: Vec<(Vec<[Option<wgpu::BindGroup>; 2]>, u64)>,
    pub(crate) filled: Vec<Filled>,
    pub(crate) switches: Vec<(u32, bool)>,
    pub(crate) placeholders: Vec<(crate::linker::Ty, Placeholder)>,
    pub(crate) encoder: Option<wgpu::CommandEncoder>,
    pub(crate) open: Option<Open>,
    pub(crate) planned: Option<Planned>,
    pub(crate) debug: bool,
    pub(crate) region: Option<(u32, u32)>,
    pub(crate) assets: Assets,
    pub(crate) surface: wgpu::TextureFormat,
    pub(crate) gates: Vec<Option<engine_registry::GateId>>,
    pub(crate) piece_layout: Vec<(usize, crate::pieces::PieceAt)>,
    pub(crate) post: crate::shared::PostUniform,
    pub(crate) post_buffer: wgpu::Buffer,
    pub(crate) post_written: Option<crate::shared::PostUniform>,
    pub(crate) linear: wgpu::Sampler,
    pub(crate) drawn: u32,
}

pub(crate) enum Placeholder {
    Buffer(wgpu::Buffer),
    Texture(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

pub(crate) struct RawDraw<'e> {
    pub(crate) run: &'e mut Run,
    pub(crate) part: u32,
    pub(crate) point: Option<TypeId>,
}

impl RawDraw<'_> {
    pub(crate) fn reborrow(&mut self) -> RawDraw<'_> {
        RawDraw {
            run: &mut *self.run,
            part: self.part,
            point: self.point,
        }
    }
}

pub(crate) enum RawShape {
    Pulled { vertices: Range<u32>, instances: Range<u32> },
    Buffers { vertices: Vec<u32>, indices: Option<u32>, elements: Range<u32>, instances: Range<u32> },
    Indirect { vertices: Vec<u32>, indices: Option<u32>, args: u32, offset: u64 },
    Mesh { mesh: MeshHandle, material: Option<MaterialHandle>, instances: Range<u32> },
    MeshIndirect { mesh: MeshHandle, material: Option<MaterialHandle>, args: u32, offset: u64 },
}

fn fault(path: String, code: ErrorCode) -> Error {
    Error::new(code, path)
}

fn texture_usage(machine: &Machine, texture: u32) -> wgpu::TextureUsages {
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
    let attached = machine.declared.passes.iter().any(|pass| {
        pass.colors.iter().chain(&pass.depths).any(|(known, _)| *known == texture)
    });
    if attached {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    let stored = machine.pipelines.iter().any(|pipeline| {
        pipeline.groups.iter().any(|group| {
            group.entries.iter().any(|(_, bound, ty)| {
                matches!(ty, crate::linker::Ty::StorageTexture { .. })
                    && matches!(bound, Bound::Texture(known) | Bound::Slice { texture: known, .. } if *known == texture)
            })
        })
    });
    if stored {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    usage
}

impl Run {
    pub(crate) fn new(device: wgpu::Device, queue: wgpu::Queue, machine: Machine, parts: Vec<Option<Box<dyn Installed>>>, assets: Assets) -> Self {
        let buffer = |label, size, usage| {
            crate::memory::buffer(
                &device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                },
            )
        };
        let camera = buffer("engine/cameras", CAMERA_STRIDE, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let post_buffer = buffer(
            "engine/post-uniform",
            size_of::<crate::shared::PostUniform>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/post-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let pieces = buffer("engine/pieces", crate::pieces::PIECE_ALIGN, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let samplers = machine
            .declared
            .samplers
            .iter()
            .map(|desc| {
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("engine/part-sampler"),
                    address_mode_u: desc.address,
                    address_mode_v: desc.address,
                    address_mode_w: desc.address,
                    mag_filter: desc.filter,
                    min_filter: desc.filter,
                    mipmap_filter: match desc.filter {
                        wgpu::FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
                        wgpu::FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
                    },
                    compare: desc.compare,
                    ..Default::default()
                })
            })
            .collect();
        let count = parts.len();
        let textures = machine.declared.textures.iter().map(|_| [None, None]).collect();
        let sizes = machine.declared.textures.iter().map(|_| None).collect();
        let buffers = machine.declared.buffers.iter().map(|_| None).collect();
        let runs = Bits::of(machine.graph.passes());
        let frame_groups = machine.variants.iter().map(|_| ([None, None], 0)).collect();
        let own_groups = machine.pipelines.iter().map(|pipeline| (pipeline.groups.iter().map(|_| [None, None]).collect(), 0)).collect();
        Self {
            device,
            queue,
            machine,
            parts,
            live: vec![true; count],
            faulted: (0..count).map(|_| None).collect(),
            part: 0,
            pass: 0,
            frame: 0,
            parity: 0,
            generation: 1,
            textures,
            sizes,
            buffers,
            samplers,
            targets: Vec::new(),
            debug_targets: Vec::new(),
            canvas: None,
            runs,
            asked: Vec::new(),
            copies: Vec::new(),
            rings: Vec::new(),
            cameras: vec![bytemuck::Zeroable::zeroed()],
            views: Vec::new(),
            camera,
            pieces,
            pieces_written: Vec::new(),
            frame_groups,
            own_groups,
            filled: Vec::new(),
            switches: Vec::new(),
            placeholders: Vec::new(),
            encoder: None,
            open: None,
            planned: None,
            debug: false,
            region: None,
            assets,
            surface: wgpu::TextureFormat::Bgra8Unorm,
            gates: Vec::new(),
            piece_layout: Vec::new(),
            post: bytemuck::Zeroable::zeroed(),
            post_buffer,
            post_written: None,
            linear,
            drawn: 0,
        }
    }

    pub(crate) fn path(&self, what: &str) -> String {
        let name = self
            .machine
            .declared
            .parts
            .get(self.part as usize)
            .map_or("", |part| part.name);
        format!("$.parts.{name}.{what}")
    }

    pub(crate) fn fault(&mut self, error: Error) {
        if let Some(slot) = self.faulted.get_mut(self.part as usize)
            && slot.is_none()
        {
            *slot = Some(error);
        }
    }

    pub(crate) fn run_pass(&mut self, pass: u32) {
        let owned = self
            .machine
            .declared
            .passes
            .get(pass as usize)
            .is_some_and(|declared| declared.part == self.part);
        if owned {
            self.runs.insert(pass as usize);
        }
    }

    pub(crate) fn allocate(&mut self, texture: u32, size: wgpu::Extent3d, mips: u32) {
        let Some(decl) = self.machine.declared.textures.get(texture as usize) else {
            return;
        };
        let at = texture as usize;
        if self.sizes[at] == Some((size, mips)) && self.textures[at][0].is_some() {
            return;
        }
        let usage = texture_usage(&self.machine, texture);
        let made = |device: &wgpu::Device| Allocated::new(device, decl.label, decl.format, usage, size, mips);
        let previous = decl.lifetime == Lifetime::Previous;
        self.textures[at] = [Some(made(&self.device)), previous.then(|| made(&self.device))];
        self.sizes[at] = Some((size, mips));
        self.generation += 1;
    }

    pub(crate) fn allocate_buffer(&mut self, buffer: u32, bytes: u64) {
        let Some(decl) = self.machine.declared.buffers.get(buffer as usize) else {
            return;
        };
        let at = buffer as usize;
        let bytes = bytes.max(16).next_multiple_of(4);
        if self.buffers[at].as_ref().is_some_and(|(_, known)| *known == bytes) {
            return;
        }
        let usage = decl.usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let made = crate::memory::buffer(
            &self.device,
            &wgpu::BufferDescriptor {
                label: Some(decl.label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            },
        );
        self.buffers[at] = Some((made, bytes));
        self.generation += 1;
    }

    pub(crate) fn write(&mut self, buffer: u32, offset: u64, bytes: &[u8]) {
        let Some(Some((made, size))) = self.buffers.get(buffer as usize) else {
            let error = fault(self.path("write"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        if offset + bytes.len() as u64 > *size || bytes.len() % 4 != 0 {
            let error = fault(self.path("write"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        self.queue.write_buffer(made, offset, bytes);
    }

    pub(crate) fn write_texture(&mut self, texture: u32, at: wgpu::Origin3d, bytes: &[u8], layout: wgpu::TexelCopyBufferLayout, size: wgpu::Extent3d) {
        let parity = self.parity;
        let Some(made) = self.textures.get(texture as usize).and_then(|made| made[parity].as_ref().or(made[0].as_ref())) else {
            let error = fault(self.path("write_texture"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &made.texture,
                mip_level: 0,
                origin: at,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            layout,
            size,
        );
    }

    pub(crate) fn cameras(&mut self, views: &[CameraUniform]) {
        let part = self.part;
        match self.views.iter().position(|(known, _, _)| *known == part) {
            Some(at) if self.views[at].2 as usize == views.len() => {
                let base = self.views[at].1 as usize;
                self.cameras[base..base + views.len()].copy_from_slice(views);
            }
            Some(at) => {
                self.views.remove(at);
                self.repack();
                self.push_views(part, views);
            }
            None => self.push_views(part, views),
        }
    }

    fn push_views(&mut self, part: u32, views: &[CameraUniform]) {
        let base = self.cameras.len() as u32;
        self.cameras.extend_from_slice(views);
        self.views.push((part, base, views.len() as u32));
    }

    fn repack(&mut self) {
        let mut packed = vec![self.cameras[0]];
        for (_, base, count) in &mut self.views {
            let from = *base as usize;
            let kept = &self.cameras[from..from + *count as usize];
            *base = packed.len() as u32;
            packed.extend_from_slice(kept);
        }
        self.cameras = packed;
    }

    pub(crate) fn view_offset(&self, part: u32, view: Option<u32>) -> Option<u32> {
        let Some(view) = view else {
            return Some(0);
        };
        let (_, base, count) = self.views.iter().find(|(known, _, _)| *known == part)?;
        (view < *count).then(|| (base + view) * CAMERA_STRIDE as u32)
    }

    pub(crate) fn write_cameras(&mut self) {
        let bytes = self.cameras.len() as u64 * CAMERA_STRIDE;
        if self.camera.size() < bytes {
            self.camera = crate::memory::buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/cameras"),
                    size: bytes.next_power_of_two(),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );
            self.generation += 1;
        }
        let mut staged = vec![0_u8; bytes as usize];
        for (index, camera) in self.cameras.iter().enumerate() {
            let at = index * CAMERA_STRIDE as usize;
            let raw = bytemuck::bytes_of(camera);
            staged[at..at + raw.len()].copy_from_slice(raw);
        }
        self.queue.write_buffer(&self.camera, 0, &staged);
    }

    pub(crate) fn write_post(&mut self) {
        if self.post_written == Some(self.post) {
            return;
        }
        self.queue.write_buffer(&self.post_buffer, 0, bytemuck::bytes_of(&self.post));
        self.post_written = Some(self.post);
    }

    pub(crate) fn write_pieces(&mut self, bytes: Vec<u8>) {
        if bytes == self.pieces_written {
            return;
        }
        let size = (bytes.len() as u64).max(crate::pieces::PIECE_ALIGN).next_multiple_of(crate::pieces::PIECE_ALIGN);
        if self.pieces.size() != size {
            self.pieces = crate::memory::buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/pieces"),
                    size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );
            self.generation += 1;
        }
        let mut staged = bytes.clone();
        staged.resize(size as usize, 0);
        self.queue.write_buffer(&self.pieces, 0, &staged);
        self.pieces_written = bytes;
    }

    pub(crate) fn read_back(&mut self, asked: Asked) -> bool {
        let free = match self.rings.iter().find(|ring| ring.part == asked.part && ring.tag == asked.tag) {
            Some(ring) => ring.slots.iter().any(|(_, _, state, _)| state.load(std::sync::atomic::Ordering::Acquire) == 0),
            None => true,
        };
        if free {
            self.asked.push(asked);
        }
        free
    }

    pub(crate) fn fill(&mut self, pipeline: u32, textures: &[TextureHandle]) -> u32 {
        let generation = self.assets.materials.generation();
        let base = self.machine.pipelines.len() as u32;
        if let Some(at) = self.filled.iter().position(|filled| filled.base == pipeline && filled.textures == textures && filled.generation == generation) {
            self.filled[at].used = self.frame;
            return base + at as u32;
        }
        self.filled.push(Filled {
            base: pipeline,
            textures: textures.to_vec(),
            generation,
            used: self.frame,
            groups: Vec::new(),
        });
        base + (self.filled.len() - 1) as u32
    }

    pub(crate) fn drop_unfilled(&mut self) {
        let frame = self.frame;
        self.filled.retain(|filled| filled.used + 1 >= frame);
    }

    pub(crate) fn texture_view(&mut self, bound: &Bound, ty: crate::linker::Ty, reading: bool) -> Option<wgpu::TextureView> {
        let parity = self.parity;
        let dimension = match ty {
            crate::linker::Ty::Texture { view, .. } | crate::linker::Ty::StorageTexture { view, .. } => Some(view),
            crate::linker::Ty::Uniform | crate::linker::Ty::Storage { .. } | crate::linker::Ty::Sampler | crate::linker::Ty::Comparison => None,
        };
        match bound {
            Bound::Texture(texture) => {
                let previous = self.machine.declared.textures.get(*texture as usize)?.lifetime == Lifetime::Previous;
                let made = self.textures.get_mut(*texture as usize)?;
                let index = if previous && reading { 1 - parity } else { parity };
                let made = made[index].as_mut()?;
                let whole = 0..made.mips;
                let layers = 0..made.size.depth_or_array_layers;
                Some(made.slice(&whole, &layers, dimension).clone())
            }
            Bound::Slice { texture, mips, layers } => {
                let made = self.textures.get_mut(*texture as usize)?;
                let made = made[parity].as_mut()?;
                Some(made.slice(mips, layers, dimension).clone())
            }
            Bound::Target(view) => {
                let wanted = match (self.pass_layer(), view) {
                    (Some(layer), view) => Target::Copy(layer, *view),
                    (None, ViewTarget::Scene) => Target::Scene,
                    (None, ViewTarget::Depth) => Target::Depth,
                };
                self.targets.iter().find(|(known, _)| *known == wanted).map(|(_, made)| made.view.clone())
            }
            Bound::Buffer(_) | Bound::Sampler(_) | Bound::Asset => None,
        }
    }

    pub(crate) fn pass_layer(&self) -> Option<crate::Layer> {
        let pass = self.machine.declared.passes.get(self.pass as usize)?;
        match pass.work {
            Work::Shared(layer) => Some(layer),
            Work::Render | Work::Copy { .. } | Work::Clear(_) => pass.layer,
        }
    }

    fn placeholder(&mut self, ty: crate::linker::Ty) -> wgpu::BindingResource<'_> {
        if !self.placeholders.iter().any(|(known, _)| *known == ty) {
            let made = placeholder(&self.device, ty);
            self.placeholders.push((ty, made));
        }
        let Some((_, made)) = self.placeholders.iter().find(|(known, _)| *known == ty) else {
            return self.camera.as_entire_binding();
        };
        match made {
            Placeholder::Buffer(buffer) => buffer.as_entire_binding(),
            Placeholder::Texture(view) => wgpu::BindingResource::TextureView(view),
            Placeholder::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
        }
    }

    fn resource(&mut self, bound: &Bound, ty: crate::linker::Ty, reading: bool) -> Resolved {
        match bound {
            Bound::Buffer(buffer) => match self.buffers.get(*buffer as usize).and_then(Option::as_ref) {
                Some((made, _)) => Resolved::Buffer(made.clone()),
                None => Resolved::Placeholder(ty),
            },
            Bound::Sampler(sampler) => match self.samplers.get(*sampler as usize) {
                Some(made) => Resolved::Sampler(made.clone()),
                None => Resolved::Placeholder(ty),
            },
            Bound::Texture(_) | Bound::Slice { .. } | Bound::Target(_) => match self.texture_view(bound, ty, reading) {
                Some(view) => Resolved::View(view),
                None => Resolved::Placeholder(ty),
            },
            Bound::Asset => Resolved::Placeholder(ty),
        }
    }

    fn bind_group(&mut self, layout: &wgpu::BindGroupLayout, entries: Vec<(u32, Resolved)>) -> wgpu::BindGroup {
        for (_, resolved) in &entries {
            if let Resolved::Placeholder(ty) = resolved {
                let _ = self.placeholder(*ty);
            }
        }
        let resources: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .map(|(binding, resolved)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: match resolved {
                    Resolved::Buffer(buffer) => buffer.as_entire_binding(),
                    Resolved::Dynamic(buffer, size) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(*size),
                    }),
                    Resolved::View(view) => wgpu::BindingResource::TextureView(view),
                    Resolved::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    Resolved::Placeholder(ty) => match self.placeholders.iter().find(|(known, _)| known == ty) {
                        Some((_, Placeholder::Buffer(buffer))) => buffer.as_entire_binding(),
                        Some((_, Placeholder::Texture(view))) => wgpu::BindingResource::TextureView(view),
                        Some((_, Placeholder::Sampler(sampler))) => wgpu::BindingResource::Sampler(sampler),
                        None => self.camera.as_entire_binding(),
                    },
                },
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/bound-group"),
            layout,
            entries: &resources,
        })
    }

    pub(crate) fn refresh_groups(&mut self) {
        let generation = self.generation;
        let saved_pass = self.pass;
        self.pass = u32::MAX;
        for variant in 0..self.machine.variants.len() {
            if self.frame_groups[variant].1 == generation {
                continue;
            }
            for parity in 0..2 {
                let saved = self.parity;
                self.parity = parity;
                let entries: Vec<(u32, crate::bindings::Stages)> = self.machine.variants[variant].entries.clone();
                let mut resolved = Vec::with_capacity(entries.len());
                for (binding, _) in entries {
                    let ty = self.frame_ty(binding);
                    let found = match binding {
                        0 => Resolved::Dynamic(self.camera.clone(), size_of::<CameraUniform>() as u64),
                        1 => Resolved::Buffer(self.pieces.clone()),
                        _ => match self.machine.frame_bound.get(&binding).cloned() {
                            Some(bound) => self.resource(&bound, ty, true),
                            None => self.core_frame(binding, ty),
                        },
                    };
                    resolved.push((binding, found));
                }
                let layout = self.machine.variants[variant].layout.clone();
                let group = self.bind_group(&layout, resolved);
                self.frame_groups[variant].0[parity] = Some(group);
                self.parity = saved;
            }
            self.frame_groups[variant].1 = generation;
        }
        for pipeline in 0..self.machine.pipelines.len() {
            if self.own_groups[pipeline].1 == generation {
                continue;
            }
            let groups = self.machine.pipelines[pipeline].groups.len();
            self.pass = self.home_pass(pipeline);
            for group in 0..groups {
                for parity in 0..2 {
                    let saved = self.parity;
                    self.parity = parity;
                    let entries = self.machine.pipelines[pipeline].groups[group].entries.clone();
                    let mut resolved = Vec::with_capacity(entries.len());
                    for (binding, bound, ty) in entries {
                        let found = self.resource(&bound, ty, !ty.writes());
                        resolved.push((binding, found));
                    }
                    let layout = self.machine.pipelines[pipeline].groups[group].layout.clone();
                    let made = self.bind_group(&layout, resolved);
                    self.own_groups[pipeline].0[group][parity] = Some(made);
                    self.parity = saved;
                }
            }
            self.own_groups[pipeline].1 = generation;
        }
        self.pass = saved_pass;
    }

    fn home_pass(&self, pipeline: usize) -> u32 {
        let Some(built) = self.machine.pipelines.get(pipeline) else {
            return u32::MAX;
        };
        let part = built.decl.part;
        match built.decl.scope {
            Scoped::Pass(pass) => pass,
            Scoped::Point(point) => self
                .machine
                .declared
                .passes
                .iter()
                .position(|pass| {
                    if point == TypeId::of::<crate::MainView>() {
                        matches!(pass.work, Work::Shared(_)) && pass.members.contains(&part)
                    } else {
                        pass.claims.contains(&point)
                    }
                })
                .map_or(u32::MAX, |pass| pass as u32),
        }
    }

    fn frame_ty(&self, binding: u32) -> crate::linker::Ty {
        self.machine
            .frame
            .iter()
            .find(|(path, name, _)| self.machine.linker.binding_of(path, name) == Some(binding))
            .map_or(crate::linker::Ty::Uniform, |(_, _, ty)| *ty)
    }

    fn core_frame(&mut self, binding: u32, ty: crate::linker::Ty) -> Resolved {
        let name = self
            .machine
            .frame
            .iter()
            .find(|(path, name, _)| self.machine.linker.binding_of(path, name) == Some(binding))
            .map(|(_, name, _)| name.clone());
        let target = match name.as_deref() {
            Some(crate::shared::POST) => return Resolved::Buffer(self.post_buffer.clone()),
            Some(crate::shared::LINEAR) => return Resolved::Sampler(self.linear.clone()),
            Some(crate::shared::SCENE) => Some(Target::Scene),
            Some(crate::shared::GLOW) => Some(Target::Glow),
            Some(crate::shared::DEPTH) => Some(Target::Depth),
            Some(name) => name
                .strip_prefix(crate::shared::OUTPUT_PREFIX)
                .and_then(|output| self.machine.outputs.iter().position(|known| known.name == output))
                .map(|index| Target::Output(index as u32)),
            None => None,
        };
        let found = target.and_then(|target| {
            self.targets
                .iter()
                .find(|(known, _)| *known == target)
                .map(|(_, made)| made.view.clone())
        });
        match found {
            Some(view) => Resolved::View(view),
            None => Resolved::Placeholder(ty),
        }
    }

    pub(crate) fn size_targets(&mut self, size: (u32, u32), canvas: Option<wgpu::TextureView>) {
        self.canvas = canvas;
        let extent = wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        };
        if self.targets.first().is_some_and(|(_, made)| made.size == extent) {
            return;
        }
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        let mut made = Vec::new();
        let mut add = |target: Target, label: &'static str, format| {
            made.push((target, Allocated::new(&self.device, label, format, usage, extent, 1)));
        };
        add(Target::Scene, "engine/scene", crate::machine::SCENE_FORMAT);
        add(Target::Glow, "engine/glow", crate::machine::GLOW_FORMAT);
        add(Target::Depth, "engine/depth", DEPTH_FORMAT);
        for (index, output) in self.machine.outputs.iter().enumerate() {
            add(Target::Output(index as u32), output.name, output.format);
        }
        for (target, _) in &self.machine.ids.targets {
            if let Target::Copy(_, view) = target {
                let format = match view {
                    ViewTarget::Scene => crate::machine::SCENE_FORMAT,
                    ViewTarget::Depth => DEPTH_FORMAT,
                };
                add(*target, "engine/view-copy", format);
            }
        }
        self.targets = made;
        self.debug_targets = DEBUG_FORMATS
            .iter()
            .map(|format| Allocated::new(&self.device, "engine/debug-target", *format, usage, extent, 1))
            .collect();
        self.generation += 1;
    }

    fn target_view(&self, target: Target) -> Option<&wgpu::TextureView> {
        if target == Target::Canvas {
            return self.canvas.as_ref();
        }
        self.targets.iter().find(|(known, _)| *known == target).map(|(_, made)| &made.view)
    }

    pub(crate) fn begin_render(&mut self, attach: AttachRaw, view: Option<u32>) -> bool {
        let Some(offset) = self.view_offset(self.part, view) else {
            let error = fault(self.path("render.view"), ErrorCode::OutOfRange);
            self.fault(error);
            return false;
        };
        let Some(pass) = self.machine.declared.passes.get(self.pass as usize).cloned() else {
            return false;
        };
        let Some(planned) = self.planned.clone() else {
            return false;
        };
        let layer = self.pass_layer();
        let shared = matches!(pass.work, Work::Shared(_));
        let mut colors: Vec<(u64, Option<wgpu::TextureView>, wgpu::TextureFormat, Ops)> = Vec::new();
        let mut depth: Option<(u64, wgpu::TextureView, wgpu::TextureFormat, Option<Ops>, bool)> = None;
        let ops_of = |id: Option<crate::graph::ResourceId>| {
            planned
                .colors
                .iter()
                .find(|(known, _)| Some(*known) == id)
                .map(|(_, ops)| *ops)
        };
        let key = |target: Target| -> u64 {
            let mut hasher = DefaultHasher::new();
            format!("{target:?}").hash(&mut hasher);
            hasher.finish()
        };
        if matches!(attach, AttachRaw::Declared) {
            if let Some(layer) = layer {
                if self.debug {
                    for (index, made) in self.debug_targets.iter().enumerate() {
                        let ops = Ops { load: Load::Clear(Clear::Color(crate::graph::TRANSPARENT)), store: true };
                        colors.push((900 + index as u64, Some(made.view.clone()), made.texture.format(), ops));
                    }
                } else {
                    for (target, format, _, written) in set_formats(&self.machine.outputs, Some(layer)) {
                        let id = self.machine.ids.target(target);
                        match ops_of(id).filter(|_| written) {
                            Some(ops) => colors.push((key(target), self.target_view(target).cloned(), format, ops)),
                            None => colors.push((key(target), None, format, Ops { load: Load::Load, store: false })),
                        }
                    }
                }
                if let (Some(id), Some(made)) = (self.machine.ids.target(Target::Depth), self.target_view(Target::Depth)) {
                    let ops = planned.depth.and_then(|(known, ops)| (known == id).then_some(ops)).flatten();
                    depth = Some((key(Target::Depth), made.clone(), DEPTH_FORMAT, ops, true));
                }
            }
            if pass.name == crate::shared::COMPOSITE {
                let id = self.machine.ids.target(Target::Canvas);
                if let (Some(ops), Some(canvas)) = (ops_of(id), self.canvas.clone()) {
                    let format = self.canvas_format();
                    colors.insert(0, (key(Target::Canvas), Some(canvas), format, ops));
                }
                for (target, format, _, _) in set_formats(&self.machine.outputs, None).into_iter().skip(2).filter(|(_, _, _, written)| *written) {
                    let id = self.machine.ids.target(target);
                    let ops = ops_of(id).unwrap_or(Ops { load: Load::Load, store: true });
                    colors.push((key(target), self.target_view(target).cloned(), format, ops));
                }
            }
            for &(texture, _) in &pass.colors {
                let id = self.machine.ids.textures.get(texture as usize).copied();
                let Some(ops) = ops_of(id) else { continue };
                let format = self.machine.declared.textures[texture as usize].format;
                let view = self.attachment_view(texture, 0);
                colors.push((u64::from(texture) + 1_000, view, format, ops));
            }
            if depth.is_none()
                && let Some(&(texture, _)) = pass.depths.first()
            {
                let id = self.machine.ids.textures.get(texture as usize).copied();
                let ops = planned.depth.and_then(|(known, ops)| (Some(known) == id).then_some(ops)).flatten();
                let format = self.machine.declared.textures[texture as usize].format;
                if let Some(view) = self.attachment_view(texture, 0) {
                    depth = Some((u64::from(texture) + 2_000, view, format, ops, false));
                }
            }
        } else {
            let (texture, slice, is_depth) = match attach {
                AttachRaw::Color(texture, slice) => (texture, slice, false),
                AttachRaw::Depth(texture, slice) => (texture, slice, true),
                AttachRaw::Declared => return false,
            };
            let declared = if is_depth { &pass.depths } else { &pass.colors };
            if !declared.iter().any(|(known, _)| *known == texture) {
                let error = fault(self.path(&format!("passes.{}.attach", pass.name)), ErrorCode::PassMismatch);
                self.fault(error);
                return false;
            }
            let format = self.machine.declared.textures[texture as usize].format;
            let view = self.attachment_view(texture, slice.unwrap_or(0));
            let id = self.machine.ids.textures.get(texture as usize).copied();
            let attachment = u64::from(texture) * 4_096 + u64::from(slice.unwrap_or(0)) + 10_000;
            if is_depth {
                let ops = planned.depth.and_then(|(known, ops)| (Some(known) == id).then_some(ops)).flatten();
                let ops = ops.or(Some(Ops { load: Load::Load, store: true }));
                if let Some(view) = view {
                    depth = Some((attachment, view, format, ops, false));
                }
            } else {
                let ops = ops_of(id).unwrap_or(Ops { load: Load::Load, store: true });
                colors.push((attachment, view, format, ops));
            }
        }
        let attachments: Vec<u64> = colors.iter().map(|(key, ..)| *key).chain(depth.as_ref().map(|(key, ..)| *key)).collect();
        if let Some(open) = &mut self.open
            && open.shared
            && shared
            && open.attachments == attachments
        {
            if open.offset != offset {
                open.offset = offset;
                open.set.frame = None;
            }
            return true;
        }
        self.open = None;
        let Some(encoder) = self.encoder.as_mut() else {
            return false;
        };
        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = colors
            .iter()
            .map(|(_, view, _, ops)| {
                view.as_ref().map(|view| wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: color_ops(*ops),
                })
            })
            .collect();
        let depth_attachment = depth.as_ref().map(|(_, view, _, ops, _)| wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(depth_ops(ops.unwrap_or(Ops { load: Load::Load, store: true }))),
            stencil_ops: None,
        });
        let mut render = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(pass.name),
                color_attachments: &color_attachments,
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();
        if layer.is_some()
            && let Some(region) = self.region
        {
            crate::targets::draw_within(&mut render, region);
        }
        self.open = Some(Open {
            pass: render,
            attachments,
            formats: colors.iter().map(|(_, view, format, _)| view.as_ref().map(|_| *format)).collect(),
            depth: depth.map(|(_, _, format, _, view)| (format, view)),
            debug: self.debug && layer.is_some(),
            offset,
            shared,
            set: Set::default(),
        });
        true
    }

    pub(crate) fn canvas_format(&self) -> wgpu::TextureFormat {
        self.surface
    }

    fn attachment_view(&mut self, texture: u32, layer: u32) -> Option<wgpu::TextureView> {
        let parity = self.parity;
        let made = self.textures.get_mut(texture as usize)?;
        let made = made[parity].as_mut()?;
        Some(made.slice(&(0..1), &(layer..layer + 1), Some(wgpu::TextureViewDimension::D2)).clone())
    }

    pub(crate) fn end_render(&mut self) {
        self.open = None;
    }

    fn pipeline_of(&self, handle: u32) -> Option<(u32, Option<usize>)> {
        let count = self.machine.pipelines.len() as u32;
        if handle < count {
            return Some((handle, None));
        }
        let filled = (handle - count) as usize;
        Some((self.filled.get(filled)?.base, Some(filled)))
    }

    fn render_pipeline(&mut self, pipeline: u32) -> Option<(u64, wgpu::RenderPipeline)> {
        let open = self.open.as_ref()?;
        let (formats, depth, debug) = (formats.clone(), depth, debug);
        self.compile(pipeline, &formats, depth, debug)
    }

    pub(crate) fn compile(
        &mut self,
        pipeline: u32,
        formats: &[Option<wgpu::TextureFormat>],
        depth: Option<(wgpu::TextureFormat, bool)>,
        debug: bool,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let Program::Render(desc) = &built.decl.program else {
            return None;
        };
        if debug && desc.debug.is_none() {
            return None;
        }
        let switches: Vec<(u32, bool)> = self
            .switches
            .iter()
            .filter(|(id, _)| built.stages.iter().any(|stage| stage.composed.switches.contains(id)))
            .copied()
            .collect();
        let mut hasher = DefaultHasher::new();
        formats.hash(&mut hasher);
        depth.hash(&mut hasher);
        debug.hash(&mut hasher);
        switches.hash(&mut hasher);
        let key = hasher.finish();
        if let Some((_, made)) = built.made.iter().find(|(known, _)| *known == key) {
            return Some((key, made.clone()));
        }
        let constants: Vec<(String, f64)> = switches.iter().map(|(id, on)| (id.to_string(), f64::from(u8::from(*on)))).collect();
        let named: Vec<(&str, f64)> = constants.iter().map(|(name, value)| (name.as_str(), *value)).collect();
        let options = |stage: usize| wgpu::PipelineCompilationOptions {
            constants: if built.stages.get(stage).is_some_and(|stage| !stage.composed.switches.is_empty()) { &named } else { &[] },
            zero_initialize_workgroup_memory: true,
        };
        let fragment_at = if debug { 1 + usize::from(desc.fragment.is_some()) } else { 1 };
        let fragment_entry = if debug { desc.debug } else { desc.fragment };
        let mut layouts: Vec<Option<wgpu::VertexBufferLayout<'static>>> = Vec::new();
        let mut meshes = 0;
        for layout in &desc.vertices {
            match layout {
                VertexLayout::Mesh => {
                    layouts.push(if meshes == 0 { crate::mesh_arena::vertex_layout() } else { crate::mesh_arena::extra_layout() });
                    meshes += 1;
                }
                VertexLayout::Buffer(layout) => layouts.push(Some(layout.clone())),
            }
        }
        let view_depth = depth.is_some_and(|(_, view)| view);
        let depth_stencil = depth.map(|(format, _)| {
            let (write, compare, bias) = match desc.depth {
                DepthUse::None => (false, wgpu::CompareFunction::Always, wgpu::DepthBiasState::default()),
                DepthUse::Test => (false, if view_depth { wgpu::CompareFunction::GreaterEqual } else { wgpu::CompareFunction::LessEqual }, wgpu::DepthBiasState::default()),
                DepthUse::Write { bias } => (true, if view_depth { wgpu::CompareFunction::Greater } else { wgpu::CompareFunction::Less }, bias),
                DepthUse::Overwrite => (true, wgpu::CompareFunction::Always, wgpu::DepthBiasState::default()),
            };
            wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: wgpu::StencilState::default(),
                bias,
            }
        });
        let targets: Vec<Option<wgpu::ColorTargetState>> = open
            .formats
            .iter()
            .map(|format| {
                format.map(|format| wgpu::ColorTargetState {
                    format,
                    blend: if format.sample_type(None, None).is_some_and(|sample| matches!(sample, wgpu::TextureSampleType::Float { filterable: true })) { desc.blend } else { None },
                    write_mask: desc.writes,
                })
            })
            .collect();
        let vertex_module = built.modules.first()?;
        let fragment_module = fragment_entry.and_then(|_| built.modules.get(fragment_at));
        let made = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(desc.vertex.name()),
            layout: Some(&built.layout),
            vertex: wgpu::VertexState {
                module: vertex_module,
                entry_point: Some(desc.vertex.name()),
                compilation_options: options(0),
                buffers: &layouts,
            },
            primitive: desc.primitive,
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            fragment: fragment_module.zip(fragment_entry).map(|(module, entry)| wgpu::FragmentState {
                module,
                entry_point: Some(entry.name()),
                compilation_options: options(fragment_at),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize) {
            built.made.push((key, made.clone()));
        }
        Some((key, made))
    }

    pub(crate) fn warm(&mut self) {
        for pipeline in 0..self.machine.pipelines.len() {
            let built = &self.machine.pipelines[pipeline];
            if !matches!(built.decl.program, Program::Render(_)) {
                continue;
            }
            let home = self.home_pass(pipeline);
            let Some(pass) = self.machine.declared.passes.get(home as usize) else {
                continue;
            };
            let layer = match pass.work {
                Work::Shared(layer) => Some(layer),
                Work::Render | Work::Copy { .. } | Work::Clear(_) => pass.layer,
            };
            let mut formats: Vec<Option<wgpu::TextureFormat>> = Vec::new();
            let mut depth = None;
            if let Some(layer) = layer {
                for (_, format, _, written) in set_formats(&self.machine.outputs, Some(layer)) {
                    formats.push(written.then_some(format));
                }
                depth = Some((DEPTH_FORMAT, true));
            }
            for &(texture, _) in &pass.colors {
                formats.push(self.machine.declared.textures.get(texture as usize).map(|texture| texture.format));
            }
            if depth.is_none()
                && let Some(&(texture, _)) = pass.depths.first()
            {
                depth = self.machine.declared.textures.get(texture as usize).map(|texture| (texture.format, false));
            }
            if pass.name == crate::shared::COMPOSITE {
                formats.insert(0, Some(self.surface));
            }
            let _ = self.compile(pipeline as u32, &formats, depth, false);
        }
    }

    fn scoped(&self, part: u32, point: Option<TypeId>, pipeline: u32) -> bool {
        let Some(built) = self.machine.pipelines.get(pipeline as usize) else {
            return false;
        };
        if built.decl.part != part {
            return false;
        }
        match (built.decl.scope, point) {
            (Scoped::Pass(pass), None) => pass == self.pass,
            (Scoped::Point(known), Some(point)) => known == point,
            (Scoped::Point(known), None) => {
                known == TypeId::of::<crate::MainView>()
                    && self.machine.declared.passes.get(self.pass as usize).is_some_and(|pass| matches!(pass.work, Work::Shared(_)))
            }
            (Scoped::Pass(_), Some(_)) => false,
        }
    }

    fn input(&self, part: u32, input: u32) -> Option<wgpu::Buffer> {
        let decl = self.machine.declared.inputs.get(input as usize)?;
        if decl.part != part {
            return None;
        }
        self.buffers.get(decl.buffer as usize)?.as_ref().map(|(buffer, _)| buffer.clone())
    }

    pub(crate) fn draw(&mut self, part: u32, point: Option<TypeId>, handle: u32, shape: RawShape) {
        let Some((pipeline, filled)) = self.pipeline_of(handle) else {
            return;
        };
        if !self.scoped(part, point, pipeline) {
            let saved = self.part;
            self.part = part;
            let error = fault(self.path("draw.pipeline"), ErrorCode::PassMismatch);
            self.fault(error);
            self.part = saved;
            return;
        }
        if self.machine.pipelines[pipeline as usize].decl.holes() > 0 && filled.is_none() {
            return;
        }
        if filled.is_some_and(|filled| self.filled[filled].groups.is_empty()) {
            self.fill_groups(pipeline, filled);
        }
        let Some((key, made)) = self.render_pipeline(pipeline) else {
            return;
        };
        let parity = self.parity;
        let variant = self.machine.pipelines[pipeline as usize].variant;
        let frame_group = self.frame_groups.get(variant as usize).and_then(|groups| groups.0[parity].clone());
        let own: Vec<(u32, Option<wgpu::BindGroup>)> = self.machine.pipelines[pipeline as usize]
            .groups
            .iter()
            .enumerate()
            .map(|(at, group)| {
                let filled_group = filled.and_then(|filled| {
                    self.filled[filled].groups.iter().find(|(index, _)| *index == group.index).map(|(_, made)| made.clone())
                });
                (group.index, filled_group.or_else(|| self.own_groups[pipeline as usize].0[at][parity].clone()))
            })
            .collect();
        let material = self.machine.pipelines[pipeline as usize].material;
        let vertex_inputs: Vec<Option<wgpu::Buffer>> = match &shape {
            RawShape::Buffers { vertices, .. } | RawShape::Indirect { vertices, .. } => {
                vertices.iter().map(|input| self.input(part, *input)).collect()
            }
            RawShape::Pulled { .. } | RawShape::Mesh { .. } | RawShape::MeshIndirect { .. } => Vec::new(),
        };
        let index_input = match &shape {
            RawShape::Buffers { indices, .. } | RawShape::Indirect { indices, .. } => indices.and_then(|input| self.input(part, input)),
            RawShape::Pulled { .. } | RawShape::Mesh { .. } | RawShape::MeshIndirect { .. } => None,
        };
        let args_input = match &shape {
            RawShape::Indirect { args, .. } | RawShape::MeshIndirect { args, .. } => self.input(part, *args),
            RawShape::Pulled { .. } | RawShape::Buffers { .. } | RawShape::Mesh { .. } => None,
        };
        let mesh = match &shape {
            RawShape::Mesh { mesh, material, .. } | RawShape::MeshIndirect { mesh, material, .. } => {
                let Some(gpu) = self.assets.meshes.get(mesh) else {
                    return;
                };
                let draw = gpu.draw(&self.assets.arena);
                let (group, _) = self.assets.materials.binding(*material);
                Some((draw.vertices.clone(), draw.extras.cloned(), draw.indices.clone(), draw.first_index, draw.base_vertex, draw.index_count, group.clone()))
            }
            RawShape::Pulled { .. } | RawShape::Buffers { .. } | RawShape::Indirect { .. } => None,
        };
        let Some(open) = self.open.as_mut() else {
            return;
        };
        let render = &mut open.pass;
        if open.set.pipeline != Some((pipeline, key)) {
            render.set_pipeline(&made);
            open.set.pipeline = Some((pipeline, key));
            open.set.groups.clear();
            open.set.frame = None;
        }
        if open.set.frame != Some((variant, open.offset))
            && let Some(group) = &frame_group
        {
            render.set_bind_group(0, group, &[open.offset]);
            open.set.frame = Some((variant, open.offset));
        }
        for (index, group) in &own {
            if let Some(group) = group {
                render.set_bind_group(*index, group, &[]);
            }
        }
        let _ = material;
        match shape {
            RawShape::Pulled { vertices, instances } => render.draw(vertices, instances),
            RawShape::Buffers { elements, instances, .. } => {
                for (slot, buffer) in vertex_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(slot as u32, buffer.slice(..));
                    }
                }
                match index_input {
                    Some(indices) => {
                        render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        render.draw_indexed(elements, 0, instances);
                    }
                    None => render.draw(elements, instances),
                }
            }
            RawShape::Indirect { offset, .. } => {
                for (slot, buffer) in vertex_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(slot as u32, buffer.slice(..));
                    }
                }
                let Some(args) = args_input else { return };
                match index_input {
                    Some(indices) => {
                        render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        render.draw_indexed_indirect(&args, offset);
                    }
                    None => render.draw_indirect(&args, offset),
                }
            }
            RawShape::Mesh { instances, .. } => {
                let Some((vertices, extras, indices, first, base, count, group)) = mesh else { return };
                render.set_bind_group(1, &group, &[]);
                render.set_vertex_buffer(0, vertices.slice(..));
                if let Some(extras) = &extras {
                    render.set_vertex_buffer(1, extras.slice(..));
                }
                render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                render.draw_indexed(first..first + count, base, instances);
            }
            RawShape::MeshIndirect { offset, .. } => {
                let Some((vertices, extras, indices, _, _, _, group)) = mesh else { return };
                let Some(args) = args_input else { return };
                render.set_bind_group(1, &group, &[]);
                render.set_vertex_buffer(0, vertices.slice(..));
                if let Some(extras) = &extras {
                    render.set_vertex_buffer(1, extras.slice(..));
                }
                render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                render.draw_indexed_indirect(&args, offset);
            }
        }
        self.drawn += 1;
    }

    fn fill_groups(&mut self, pipeline: u32, filled: Option<usize>) {
        let Some(filled) = filled else { return };
        let textures = self.filled[filled].textures.clone();
        let groups = self.machine.pipelines[pipeline as usize].groups.len();
        let mut made = Vec::with_capacity(groups);
        for group in 0..groups {
            let entries = self.machine.pipelines[pipeline as usize].groups[group].entries.clone();
            let index = self.machine.pipelines[pipeline as usize].groups[group].index;
            let mut hole = 0;
            let mut resolved = Vec::with_capacity(entries.len());
            for (binding, bound, ty) in entries {
                let found = if bound == Bound::Asset {
                    let texture = textures.get(hole).copied();
                    hole += 1;
                    match texture.and_then(|texture| self.assets.materials.texture_view(texture)) {
                        Some(view) => Resolved::View(view.clone()),
                        None => Resolved::Placeholder(ty),
                    }
                } else {
                    self.resource(&bound, ty, !ty.writes())
                };
                resolved.push((binding, found));
            }
            let layout = self.machine.pipelines[pipeline as usize].groups[group].layout.clone();
            made.push((index, self.bind_group(&layout, resolved)));
        }
        self.filled[filled].groups = made;
    }

    pub(crate) fn scissor(&mut self, rect: [u32; 4]) {
        if let Some(open) = self.open.as_mut() {
            open.pass.set_scissor_rect(rect[0], rect[1], rect[2], rect[3]);
        }
    }

    pub(crate) fn reset_scissor(&mut self) {
        let size = self.targets.first().map(|(_, made)| (made.size.width, made.size.height));
        if let (Some(open), Some((width, height))) = (self.open.as_mut(), self.region.or(size)) {
            open.pass.set_scissor_rect(0, 0, width, height);
        }
    }

    pub(crate) fn dispatch(&mut self, part: u32, handle: u32, count: CountRaw) -> u32 {
        self.open = None;
        let Some(built) = self.machine.pipelines.get(handle as usize) else {
            return 0;
        };
        if built.decl.part != part || built.decl.scope != Scoped::Pass(self.pass) {
            let error = fault(self.path("dispatch.pipeline"), ErrorCode::PassMismatch);
            self.fault(error);
            return 0;
        }
        let Some(compute) = built.compute.clone() else {
            return 0;
        };
        let parity = self.parity;
        let frame_group = self.frame_groups.get(built.variant as usize).and_then(|groups| groups.0[parity].clone());
        let own: Vec<(u32, Option<wgpu::BindGroup>)> = built
            .groups
            .iter()
            .enumerate()
            .map(|(at, group)| (group.index, self.own_groups[handle as usize].0[at][parity].clone()))
            .collect();
        let indirect = match count {
            CountRaw::Indirect(input, offset) => match self.input(part, input) {
                Some(buffer) => Some((buffer, offset)),
                None => return 0,
            },
            CountRaw::Groups(_) => None,
        };
        let Some(encoder) = self.encoder.as_mut() else {
            return 0;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/part-dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute);
        if let Some(group) = &frame_group {
            pass.set_bind_group(0, group, &[0]);
        }
        for (index, group) in &own {
            if let Some(group) = group {
                pass.set_bind_group(*index, group, &[]);
            }
        }
        match (count, indirect) {
            (_, Some((buffer, offset))) => pass.dispatch_workgroups_indirect(&buffer, offset),
            (CountRaw::Groups([x, y, z]), None) => pass.dispatch_workgroups(x, y, z),
            (CountRaw::Indirect(..), None) => {}
        }
        1
    }

    pub(crate) fn copy(&mut self, pass: u32) {
        self.open = None;
        let Some(at) = self.copies.iter().position(|(known, _)| *known == pass) else {
            return;
        };
        let (_, regions) = self.copies.swap_remove(at);
        let Some(decl) = self.machine.declared.passes.get(pass as usize).cloned() else {
            return;
        };
        if decl.part != self.part {
            let error = fault(self.path("copy"), ErrorCode::PassMismatch);
            self.fault(error);
            return;
        }
        let Work::Copy { from, to } = decl.work else {
            return;
        };
        let parity = self.parity;
        let Some(encoder) = self.encoder.as_mut() else {
            return;
        };
        for region in &regions {
            match (region, from, to) {
                (crate::CopyRegion::Bytes { from: at, to: into, size }, crate::install::Copied::Buffer(source), crate::install::Copied::Buffer(target)) => {
                    if let (Some(Some((source, _))), Some(Some((target, _)))) = (self.buffers.get(source as usize), self.buffers.get(target as usize)) {
                        encoder.copy_buffer_to_buffer(source, *at, target, *into, *size);
                    }
                }
                (crate::CopyRegion::Texels { from: at, to: into, size }, crate::install::Copied::Texture(source), crate::install::Copied::Texture(target)) => {
                    let made = |texture: u32| self.textures.get(texture as usize).and_then(|made| made[parity].as_ref().or(made[0].as_ref()));
                    if let (Some(source), Some(target)) = (made(source), made(target)) {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo { texture: &source.texture, mip_level: 0, origin: *at, aspect: wgpu::TextureAspect::All },
                            wgpu::TexelCopyTextureInfo { texture: &target.texture, mip_level: 0, origin: *into, aspect: wgpu::TextureAspect::All },
                            *size,
                        );
                    }
                }
                (crate::CopyRegion::Bytes { .. } | crate::CopyRegion::Texels { .. }, _, _) => {}
            }
        }
    }

    pub(crate) fn clear_buffer(&mut self, buffer: u32) {
        self.open = None;
        let Some(Some((made, _))) = self.buffers.get(buffer as usize) else {
            return;
        };
        if let Some(encoder) = self.encoder.as_mut() {
            encoder.clear_buffer(made, 0, None);
        }
    }

    pub(crate) fn copy_view(&mut self, layer: crate::Layer, view: ViewTarget) {
        self.open = None;
        let from = match view {
            ViewTarget::Scene => Target::Scene,
            ViewTarget::Depth => Target::Depth,
        };
        let source = self.targets.iter().find(|(known, _)| *known == from).map(|(_, made)| made.texture.clone());
        let target = self.targets.iter().find(|(known, _)| *known == Target::Copy(layer, view)).map(|(_, made)| made.texture.clone());
        if let (Some(source), Some(target), Some(encoder)) = (source, target, self.encoder.as_mut()) {
            encoder.copy_texture_to_texture(source.as_image_copy(), target.as_image_copy(), source.size());
        }
    }

    pub(crate) fn record_readbacks(&mut self) {
        self.open = None;
        let asked = std::mem::take(&mut self.asked);
        for request in asked {
            let Some(Some((buffer, _))) = self.buffers.get(request.buffer as usize).map(|made| made.as_ref().map(|(buffer, size)| (buffer.clone(), *size))) else {
                continue;
            };
            let bytes = request.range.end.saturating_sub(request.range.start).next_multiple_of(4);
            let ring = match self.rings.iter().position(|ring| ring.part == request.part && ring.tag == request.tag) {
                Some(at) => at,
                None => {
                    self.rings.push(Ring { part: request.part, tag: request.tag, slots: Vec::new() });
                    self.rings.len() - 1
                }
            };
            if self.rings[ring].slots.len() < READBACK_SLOTS {
                let staging = crate::memory::buffer(
                    &self.device,
                    &wgpu::BufferDescriptor {
                        label: Some("engine/readback"),
                        size: bytes.max(4),
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                );
                self.rings[ring].slots.push((staging, bytes, std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)), None));
            }
            let Some(slot) = self.rings[ring].slots.iter_mut().find(|(staging, size, state, _)| {
                state.load(std::sync::atomic::Ordering::Acquire) == 0 && *size >= bytes && staging.size() >= bytes
            }) else {
                continue;
            };
            let Some(encoder) = self.encoder.as_mut() else {
                continue;
            };
            encoder.copy_buffer_to_buffer(&buffer, request.range.start, &slot.0, 0, bytes);
            if request.then == AfterRead::Clear {
                encoder.clear_buffer(&buffer, request.range.start, Some(bytes));
            }
            slot.2.store(1, std::sync::atomic::Ordering::Release);
            slot.3 = Some((self.frame, request.deliver));
        }
    }

    pub(crate) fn map_readbacks(&mut self) {
        for ring in &self.rings {
            for (staging, bytes, state, _) in &ring.slots {
                if state.load(std::sync::atomic::Ordering::Acquire) != 1 {
                    continue;
                }
                state.store(2, std::sync::atomic::Ordering::Release);
                let mapped = std::sync::Arc::clone(state);
                staging.slice(..*bytes).map_async(
                    wgpu::MapMode::Read,
                    crate::platform::map_callback(move |result| {
                        mapped.store(if result.is_ok() { 3 } else { 0 }, std::sync::atomic::Ordering::Release);
                    }),
                );
            }
        }
    }

    pub(crate) fn deliver(&mut self, world: &mut World) {
        #[cfg(not(target_arch = "wasm32"))]
        crate::platform::run_arrived_maps();
        for ring in &mut self.rings {
            for (staging, bytes, state, asked) in &mut ring.slots {
                if state.load(std::sync::atomic::Ordering::Acquire) != 3 {
                    continue;
                }
                if let Some((frame, deliver)) = asked.take() {
                    let view = staging.slice(..*bytes).get_mapped_range();
                    deliver(world, frame, ring.tag, &view);
                }
                staging.unmap();
                state.store(0, std::sync::atomic::Ordering::Release);
            }
        }
    }
}

pub(crate) const READBACK_SLOTS: usize = 3;

pub(crate) enum Resolved {
    Buffer(wgpu::Buffer),
    Dynamic(wgpu::Buffer, u64),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
    Placeholder(crate::linker::Ty),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AttachRaw {
    Declared,
    Depth(u32, Option<u32>),
    Color(u32, Option<u32>),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CountRaw {
    Groups([u32; 3]),
    Indirect(u32, u64),
}

fn color_ops(ops: Ops) -> wgpu::Operations<wgpu::Color> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Color([r, g, b, a])) => wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
            Load::Clear(Clear::Depth(_)) | Load::Load => wgpu::LoadOp::Load,
        },
        store: if ops.store { wgpu::StoreOp::Store } else { wgpu::StoreOp::Discard },
    }
}

fn depth_ops(ops: Ops) -> wgpu::Operations<f32> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Depth(depth)) => wgpu::LoadOp::Clear(depth),
            Load::Clear(Clear::Color(_)) | Load::Load => wgpu::LoadOp::Load,
        },
        store: if ops.store { wgpu::StoreOp::Store } else { wgpu::StoreOp::Discard },
    }
}

fn placeholder(device: &wgpu::Device, ty: crate::linker::Ty) -> Placeholder {
    use crate::linker::{Sample, Ty};
    let buffer = |usage| {
        Placeholder::Buffer(crate::memory::buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("engine/placeholder"),
                size: 4_096,
                usage,
                mapped_at_creation: false,
            },
        ))
    };
    match ty {
        Ty::Uniform => buffer(wgpu::BufferUsages::UNIFORM),
        Ty::Storage { .. } => buffer(wgpu::BufferUsages::STORAGE),
        Ty::Sampler => Placeholder::Sampler(device.create_sampler(&wgpu::SamplerDescriptor::default())),
        Ty::Comparison => Placeholder::Sampler(device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        })),
        Ty::Texture { sample, view } => {
            let format = match sample {
                Sample::Float => wgpu::TextureFormat::Rgba8Unorm,
                Sample::Depth => DEPTH_FORMAT,
                Sample::Uint => wgpu::TextureFormat::R32Uint,
                Sample::Sint => wgpu::TextureFormat::R32Sint,
            };
            placeholder_texture(device, format, view, wgpu::TextureUsages::TEXTURE_BINDING)
        }
        Ty::StorageTexture { format, view, .. } => placeholder_texture(device, format, view, wgpu::TextureUsages::STORAGE_BINDING),
    }
}

fn placeholder_texture(device: &wgpu::Device, format: wgpu::TextureFormat, view: wgpu::TextureViewDimension, usage: wgpu::TextureUsages) -> Placeholder {
    let layers = match view {
        wgpu::TextureViewDimension::Cube | wgpu::TextureViewDimension::CubeArray => 6,
        wgpu::TextureViewDimension::D1 | wgpu::TextureViewDimension::D2 | wgpu::TextureViewDimension::D2Array | wgpu::TextureViewDimension::D3 => 1,
    };
    let dimension = match view {
        wgpu::TextureViewDimension::D3 => wgpu::TextureDimension::D3,
        wgpu::TextureViewDimension::D1 => wgpu::TextureDimension::D1,
        wgpu::TextureViewDimension::D2 | wgpu::TextureViewDimension::D2Array | wgpu::TextureViewDimension::Cube | wgpu::TextureViewDimension::CubeArray => wgpu::TextureDimension::D2,
    };
    let texture = crate::memory::texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("engine/placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: &[],
        },
    );
    Placeholder::Texture(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(view),
        ..Default::default()
    }))
}
