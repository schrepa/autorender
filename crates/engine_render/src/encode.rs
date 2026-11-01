use std::any::TypeId;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, MaterialHandle, MeshHandle, TextureHandle};

use crate::graph::{Bits, Clear, FAR, Load, Ops, Planned};
use crate::install::{Bound, Copied, Program, Scoped, Work};
use crate::linker::Ty;
use crate::machine::{
    CAMERA_STRIDE, DEBUG_COUNT, DEBUG_FORMATS, DEPTH_FORMAT, Machine, Target, final_target,
    own_colors_only, pass_layer, reads_depth_direct, set_formats, shared, target_read,
};
use crate::registration::Installed;
use crate::{AfterRead, CameraUniform, DepthUse, VertexLayout, ViewTarget};

pub(crate) type FrameCore = Run;
pub(crate) type EncodeCore = Run;

type Specialized = (u32, [(&'static str, f64); 15]);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Compiled {
    pipeline: u32,
    formats: Vec<Option<wgpu::TextureFormat>>,
    depth: Option<(wgpu::TextureFormat, bool)>,
    debug: bool,
    main: bool,
    variant: Option<u32>,
    constants: Vec<(String, f64)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Listing {
    canvas: wgpu::TextureFormat,
    compiled: Vec<Compiled>,
}

fn specialized(compiled: &Compiled) -> Option<Specialized> {
    let bits = compiled.variant?;
    let mut constants = crate::materials::Variant::BASIC.constants();
    for (name, value) in &mut constants {
        if let Some((_, listed)) = compiled.constants.iter().find(|(known, _)| known.as_str() == *name) {
            *value = *listed;
        }
    }
    Some((bits, constants))
}

const GROUPED: bool = cfg!(any(not(target_arch = "wasm32"), feature = "profiling"));

pub(crate) struct Allocated {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) size: wgpu::Extent3d,
    pub(crate) mips: u32,
    pub(crate) slices: Vec<(Range<u32>, Range<u32>, Option<wgpu::TextureViewDimension>, wgpu::TextureView)>,
}

pub(crate) struct Made {
    pub(crate) label: &'static str,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) usage: wgpu::TextureUsages,
    pub(crate) dimension: wgpu::TextureDimension,
}

impl Allocated {
    fn new(device: &wgpu::Device, made: &Made, size: wgpu::Extent3d, mips: u32) -> Self {
        let texture = crate::memory::texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some(made.label),
                size,
                mip_level_count: mips.max(1),
                sample_count: 1,
                dimension: made.dimension,
                format: made.format,
                usage: made.usage,
                view_formats: &[],
            },
        );
        let layered = made.dimension == wgpu::TextureDimension::D2 && size.depth_or_array_layers > 1;
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: layered.then_some(wgpu::TextureViewDimension::D2Array),
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

    fn layers(&self) -> u32 {
        if self.texture.dimension() == wgpu::TextureDimension::D3 {
            1
        } else {
            self.size.depth_or_array_layers
        }
    }

    fn slice(&mut self, mips: &Range<u32>, layers: &Range<u32>, dimension: Option<wgpu::TextureViewDimension>) -> Option<&wgpu::TextureView> {
        let mips = mips.start..mips.end.min(self.mips);
        let layers = layers.start..layers.end.min(self.layers());
        if mips.start >= mips.end || layers.start >= layers.end {
            return None;
        }
        let at = match self.slices.iter().position(|(known_mips, known_layers, known, _)| {
            *known_mips == mips && *known_layers == layers && *known == dimension
        }) {
            Some(at) => at,
            None => {
                let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mips.start,
                    mip_level_count: Some(mips.end - mips.start),
                    base_array_layer: layers.start,
                    array_layer_count: Some(layers.end - layers.start),
                    dimension,
                    ..Default::default()
                });
                self.slices.push((mips, layers, dimension, view));
                self.slices.len() - 1
            }
        };
        Some(&self.slices[at].3)
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

pub(crate) struct Slot {
    buffer: wgpu::Buffer,
    bytes: u64,
    state: Arc<AtomicU8>,
    asked: Option<(u64, fn(&mut World, u64, u32, &[u8]))>,
    once: bool,
}

pub(crate) struct Ring {
    pub(crate) part: u32,
    pub(crate) tag: u32,
    pub(crate) slots: Vec<Slot>,
}

impl Ring {
    fn kept(&self) -> usize {
        self.slots.iter().filter(|slot| !slot.once).count()
    }
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
}

pub(crate) struct Open {
    pub(crate) pass: wgpu::RenderPass<'static>,
    pub(crate) attachments: Vec<u64>,
    pub(crate) formats: Vec<Option<wgpu::TextureFormat>>,
    pub(crate) depth: Option<(wgpu::TextureFormat, bool)>,
    pub(crate) debug: bool,
    pub(crate) main: bool,
    pub(crate) offset: u32,
    pub(crate) shared: bool,
    pub(crate) label: Option<u32>,
    pub(crate) set: Set,
}

pub(crate) struct Assets {
    pub(crate) meshes: crate::mesh_arena::GpuMeshes,
    pub(crate) materials: crate::materials::MaterialStore,
    pub(crate) impostors: crate::impostors::Impostors,
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
    pub(crate) point: Option<TypeId>,
    pub(crate) frame: u64,
    pub(crate) asked_frame: u64,
    pub(crate) parity: usize,
    pub(crate) generation: u64,
    pub(crate) atlas_generation: Option<u64>,
    pub(crate) textures: Vec<[Option<Allocated>; 2]>,
    pub(crate) dimensions: Vec<wgpu::TextureDimension>,
    pub(crate) sizes: Vec<Option<(wgpu::Extent3d, u32)>>,
    pub(crate) buffers: Vec<Option<(wgpu::Buffer, u64)>>,
    pub(crate) samplers: Vec<wgpu::Sampler>,
    pub(crate) targets: Vec<(Target, Allocated)>,
    pub(crate) previous_targets: Vec<(Target, Allocated)>,
    pub(crate) debug_targets: Vec<Allocated>,
    pub(crate) canvas: Option<wgpu::TextureView>,
    pub(crate) runs: Bits,
    pub(crate) asked: Vec<Asked>,
    pub(crate) copies: Vec<(u32, Vec<crate::CopyRegion>)>,
    pub(crate) rings: Vec<Ring>,
    pub(crate) readback_slots: usize,
    pub(crate) readback_slot_bytes: Option<u64>,
    pub(crate) overflows: u64,
    pub(crate) cameras: Vec<CameraUniform>,
    pub(crate) views: Vec<(u32, u32, u32)>,
    pub(crate) camera: wgpu::Buffer,
    pub(crate) pieces: wgpu::Buffer,
    pub(crate) pieces_written: Vec<u8>,
    pub(crate) frame_groups: Vec<([Option<wgpu::BindGroup>; 2], u64)>,
    pub(crate) own_groups: Vec<(Vec<[Option<wgpu::BindGroup>; 2]>, u64)>,
    pub(crate) filled: Vec<Filled>,
    pub(crate) wanted_textures: Vec<TextureHandle>,
    pub(crate) switches: Vec<(u32, bool)>,
    pub(crate) placeholders: Vec<(Ty, Placeholder)>,
    pub(crate) encoder: Option<wgpu::CommandEncoder>,
    pub(crate) open: Option<Open>,
    pub(crate) labelled: Option<u32>,
    pub(crate) merge: bool,
    pub(crate) planned: Option<Planned>,
    pub(crate) debug: bool,
    pub(crate) temporal: bool,
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
    pub(crate) compiled: Vec<Compiled>,
    pub(crate) listed: Option<Vec<Compiled>>,
    pub(crate) up_front: u32,
    pub(crate) unlisted: u32,
}

pub(crate) enum Placeholder {
    Buffer(wgpu::Buffer),
    Texture(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

pub(crate) enum RawShape {
    Pulled { vertices: Range<u32>, instances: Range<u32> },
    Buffers { vertices: Vec<u32>, indices: Option<u32>, elements: Range<u32>, instances: Range<u32> },
    Indirect { vertices: Vec<u32>, indices: Option<u32>, args: u32, offset: u64 },
    Mesh { mesh: MeshHandle, material: Option<MaterialHandle>, instances: Range<u32>, streams: Vec<(u32, u64)> },
    MeshIndirect { mesh: MeshHandle, material: Option<MaterialHandle>, args: u32, offset: u64, streams: Vec<(u32, u64)> },
}

fn fault(path: String, code: ErrorCode) -> Error {
    Error::new(code, path)
}

fn bound_texture(bound: &Bound) -> Option<u32> {
    match bound {
        Bound::Texture(texture) | Bound::Slice { texture, .. } => Some(*texture),
        Bound::Buffer(_) | Bound::Sampler(_) | Bound::Target(_) | Bound::Asset => None,
    }
}

fn texture_uses(machine: &Machine, texture: u32) -> impl Iterator<Item = Ty> + '_ {
    let own = machine.pipelines.iter().flat_map(move |pipeline| {
        pipeline.groups.iter().flat_map(move |group| {
            group
                .entries
                .iter()
                .filter(move |(_, bound, _)| bound_texture(bound) == Some(texture))
                .map(|(_, _, ty)| *ty)
        })
    });
    let frame = machine.frame_bound.iter().filter_map(move |(binding, bound)| {
        (bound_texture(bound) == Some(texture))
            .then(|| {
                machine
                    .frame
                    .iter()
                    .find(|(path, name, _)| machine.linker.binding_of(path, name) == Some(*binding))
                    .map(|(_, _, ty)| *ty)
            })
            .flatten()
    });
    own.chain(frame)
}

fn texture_usage(machine: &Machine, texture: u32) -> wgpu::TextureUsages {
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
    let attached = machine.declared.passes.iter().any(|pass| {
        pass.colors.iter().chain(&pass.depths).any(|(known, _)| *known == texture)
    });
    if attached {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if texture_uses(machine, texture).any(|ty| matches!(ty, Ty::StorageTexture { .. })) {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    usage
}

fn texture_dimension(machine: &Machine, texture: u32) -> wgpu::TextureDimension {
    let volume = texture_uses(machine, texture).any(|ty| {
        matches!(
            ty,
            Ty::Texture { view: wgpu::TextureViewDimension::D3, .. } | Ty::StorageTexture { view: wgpu::TextureViewDimension::D3, .. }
        )
    });
    if volume {
        wgpu::TextureDimension::D3
    } else {
        wgpu::TextureDimension::D2
    }
}

fn sampler_of(device: &wgpu::Device, desc: &crate::SamplerDesc) -> wgpu::Sampler {
    let [u, v, w] = desc.address;
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("engine/part-sampler"),
        address_mode_u: u,
        address_mode_v: v,
        address_mode_w: w,
        mag_filter: desc.filter,
        min_filter: desc.filter,
        mipmap_filter: desc.mipmap,
        compare: desc.compare,
        anisotropy_clamp: desc.anisotropy.max(1),
        ..Default::default()
    })
}

fn target_format(target: Target, outputs: &[crate::machine::OutputDecl]) -> Option<wgpu::TextureFormat> {
    match target {
        Target::Scene | Target::Copy(_, ViewTarget::Scene) => Some(crate::machine::SCENE_FORMAT),
        Target::Glow => Some(crate::machine::GLOW_FORMAT),
        Target::Depth | Target::Copy(_, ViewTarget::Depth) => Some(DEPTH_FORMAT),
        Target::Output(index) => outputs.get(index as usize).map(|output| output.format),
        Target::Debug(index) | Target::Copy(_, ViewTarget::Debug(index)) => DEBUG_FORMATS.get(index as usize).copied(),
        Target::Canvas => None,
    }
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
        let samplers = machine.declared.samplers.iter().map(|desc| sampler_of(&device, desc)).collect();
        let count = parts.len();
        let textures = machine.declared.textures.iter().map(|_| [None, None]).collect();
        let dimensions = (0..machine.declared.textures.len())
            .map(|texture| texture_dimension(&machine, texture as u32))
            .collect();
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
            point: None,
            frame: 0,
            asked_frame: 0,
            parity: 0,
            generation: 1,
            atlas_generation: None,
            textures,
            dimensions,
            sizes,
            buffers,
            samplers,
            targets: Vec::new(),
            previous_targets: Vec::new(),
            debug_targets: Vec::new(),
            canvas: None,
            runs,
            asked: Vec::new(),
            copies: Vec::new(),
            rings: Vec::new(),
            readback_slots: 3,
            readback_slot_bytes: None,
            overflows: 0,
            cameras: vec![bytemuck::Zeroable::zeroed()],
            views: Vec::new(),
            camera,
            pieces,
            pieces_written: Vec::new(),
            frame_groups,
            own_groups,
            filled: Vec::new(),
            wanted_textures: Vec::new(),
            switches: Vec::new(),
            placeholders: Vec::new(),
            encoder: None,
            open: None,
            labelled: None,
            merge: true,
            planned: None,
            debug: false,
            temporal: false,
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
            compiled: Vec::new(),
            listed: None,
            up_front: 0,
            unlisted: 0,
        }
    }

    pub(crate) fn listing(&self) -> Vec<u8> {
        let listing = Listing {
            canvas: self.surface,
            compiled: self.compiled.clone(),
        };
        serde_json::to_vec(&listing).unwrap_or_default()
    }

    pub(crate) fn create_listed(&mut self, record: &[u8]) -> Result<u32, Error> {
        if record.is_empty() {
            return Ok(0);
        }
        let listing: Listing = serde_json::from_slice(record).map_err(|_| {
            crate::platform::failed("$.pipelines").expected(engine_core::Expected::Json)
        })?;
        let canvas = self.surface;
        let mut keys = listing.compiled;
        for key in &mut keys {
            for format in key.formats.iter_mut().flatten() {
                if *format == listing.canvas {
                    *format = canvas;
                }
            }
        }
        self.listed = Some(keys.clone());
        self.unlisted = 0;
        let mut created = 0;
        for key in keys {
            if key.pipeline as usize >= self.machine.pipelines.len() {
                continue;
            }
            let before = self.compiled.len();
            let _ = self.compile(key.pipeline, (&key.formats, key.depth), (key.debug, key.main), specialized(&key));
            if self.compiled.len() > before {
                created += 1;
            }
        }
        self.up_front += created;
        Ok(created)
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

    pub(crate) fn drawable(&self, part: u32) -> bool {
        self.live.get(part as usize).copied().unwrap_or(false)
            && self.faulted.get(part as usize).is_some_and(Option::is_none)
    }

    pub(crate) fn take_member(&mut self, part: u32) -> Option<Box<dyn Installed>> {
        if !self.drawable(part) {
            return None;
        }
        self.parts.get_mut(part as usize).and_then(Option::take)
    }

    pub(crate) fn give_member(&mut self, part: u32, held: Box<dyn Installed>) {
        if let Some(slot) = self.parts.get_mut(part as usize) {
            *slot = Some(held);
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

    fn doubled(&self, texture: u32) -> bool {
        self.machine.doubled.get(texture as usize).copied().unwrap_or(false)
    }

    pub(crate) fn allocate(&mut self, texture: u32, size: wgpu::Extent3d, mips: u32) {
        let Some(decl) = self.machine.declared.textures.get(texture as usize) else {
            return;
        };
        let at = texture as usize;
        if self.sizes[at] == Some((size, mips)) && self.textures[at][0].is_some() {
            return;
        }
        let made = Made {
            label: decl.label,
            format: decl.format,
            usage: texture_usage(&self.machine, texture),
            dimension: self.dimensions.get(at).copied().unwrap_or(wgpu::TextureDimension::D2),
        };
        let doubled = self.doubled(texture);
        self.textures[at] = [
            Some(Allocated::new(&self.device, &made, size, mips)),
            doubled.then(|| Allocated::new(&self.device, &made, size, mips)),
        ];
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
        crate::gpu_calls::upload_buffer(&self.queue, made, offset, bytes);
    }

    fn written_texture(&self, texture: u32) -> Option<&Allocated> {
        let made = self.textures.get(texture as usize)?;
        made[self.parity].as_ref().or(made[0].as_ref())
    }

    pub(crate) fn write_texture(
        &mut self,
        texture: u32,
        level: u32,
        at: wgpu::Origin3d,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        let Some(made) = self.written_texture(texture) else {
            let error = fault(self.path("write_texture"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        if level >= made.mips {
            let error = fault(self.path("write_texture"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        crate::gpu_calls::upload_texture(
            &self.queue,
            wgpu::TexelCopyTextureInfo {
                texture: &made.texture,
                mip_level: level,
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
        crate::gpu_calls::upload_buffer(&self.queue, &self.camera, 0, &staged);
    }

    pub(crate) fn write_post(&mut self) {
        if self.post_written == Some(self.post) {
            return;
        }
        crate::gpu_calls::upload_buffer(&self.queue, &self.post_buffer, 0, bytemuck::bytes_of(&self.post));
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
        crate::gpu_calls::upload_buffer(&self.queue, &self.pieces, 0, &staged);
        self.pieces_written = bytes;
    }

    pub(crate) fn read_back(&mut self, asked: Asked) -> bool {
        let slots = self.readback_slots;
        let free = match self.rings.iter().find(|ring| ring.part == asked.part && ring.tag == asked.tag) {
            Some(ring) => {
                ring.kept() < slots
                    || ring.slots.iter().any(|slot| !slot.once && slot.state.load(Ordering::Acquire) == 0)
            }
            None => true,
        };
        if free {
            self.asked.push(asked);
        }
        free
    }

    pub(crate) fn fill(&mut self, pipeline: u32, textures: &[TextureHandle]) -> u32 {
        for texture in textures {
            if !self.wanted_textures.contains(texture) {
                self.wanted_textures.push(*texture);
            }
        }
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

    fn reads_direct(&self) -> bool {
        let index = self.pass as usize;
        self.machine.declared.passes.get(index).is_some_and(|pass| {
            reads_depth_direct(pass, self.machine.depth_writes.get(index).copied().unwrap_or(true))
        })
    }

    fn output_view(&self, index: u32, writing: bool) -> Option<&wgpu::TextureView> {
        let target = Target::Output(index);
        let doubled = self
            .machine
            .outputs
            .get(index as usize)
            .is_some_and(|output| output.layers.is_empty());
        let second = doubled && (self.parity == 1) == writing;
        let list = if second { &self.previous_targets } else { &self.targets };
        list.iter().find(|(known, _)| *known == target).map(|(_, made)| &made.view)
    }

    fn view_of(&self, target: Target, writing: bool) -> Option<&wgpu::TextureView> {
        match target {
            Target::Canvas => self.canvas.as_ref(),
            Target::Debug(index) => self.debug_targets.get(index as usize).map(|made| &made.view),
            Target::Output(index) => self.output_view(index, writing),
            Target::Scene | Target::Glow | Target::Depth | Target::Copy(..) => {
                self.targets.iter().find(|(known, _)| *known == target).map(|(_, made)| &made.view)
            }
        }
    }

    pub(crate) fn texture_view(&mut self, bound: &Bound, ty: Ty, reading: bool) -> Option<wgpu::TextureView> {
        let parity = self.parity;
        let dimension = match ty {
            Ty::Texture { view, .. } | Ty::StorageTexture { view, .. } => Some(view),
            Ty::Uniform | Ty::Storage { .. } | Ty::Sampler | Ty::Comparison => None,
        };
        match bound {
            Bound::Texture(texture) | Bound::Slice { texture, .. } => {
                let index = if self.doubled(*texture) && reading { 1 - parity } else { parity };
                let made = self.textures.get_mut(*texture as usize)?;
                let made = match index {
                    1 if made[1].is_some() => made[1].as_mut()?,
                    _ => made[0].as_mut()?,
                };
                let (mips, layers) = match bound {
                    Bound::Slice { mips, layers, .. } => (mips.clone(), layers.clone()),
                    Bound::Texture(_) | Bound::Buffer(_) | Bound::Sampler(_) | Bound::Target(_) | Bound::Asset => {
                        (0..made.mips, 0..made.layers())
                    }
                };
                made.slice(&mips, &layers, dimension).cloned()
            }
            Bound::Target(view) => {
                let wanted = target_read(*view, self.pass_layer(), self.reads_direct());
                self.view_of(wanted, false).cloned()
            }
            Bound::Buffer(_) | Bound::Sampler(_) | Bound::Asset => None,
        }
    }

    pub(crate) fn pass_layer(&self) -> Option<crate::Layer> {
        pass_layer(self.machine.declared.passes.get(self.pass as usize)?)
    }

    fn placeholder(&mut self, ty: Ty) {
        if !self.placeholders.iter().any(|(known, _)| *known == ty) {
            let made = placeholder(&self.device, ty);
            self.placeholders.push((ty, made));
        }
    }

    fn resource(&mut self, bound: &Bound, ty: Ty, reading: bool) -> Resolved {
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

    #[expect(clippy::disallowed_methods, reason = "the machine binds groups by the linked declarations")]
    fn bind_group(&mut self, layout: &wgpu::BindGroupLayout, entries: Vec<(u32, Resolved)>) -> wgpu::BindGroup {
        for (_, resolved) in &entries {
            if let Resolved::Placeholder(ty) = resolved {
                self.placeholder(*ty);
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
        let atlas = self.assets.impostors.generation();
        if self.atlas_generation != Some(atlas) {
            self.atlas_generation = Some(atlas);
            self.generation += 1;
        }
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

    fn scope_passes(&self, pipeline: usize) -> Vec<u32> {
        let Some(built) = self.machine.pipelines.get(pipeline) else {
            return Vec::new();
        };
        self.machine
            .declared
            .passes
            .iter()
            .enumerate()
            .filter(|(index, pass)| crate::machine::in_pass(built, *index as u32, pass))
            .map(|(index, _)| index as u32)
            .collect()
    }

    fn home_pass(&self, pipeline: usize) -> u32 {
        self.scope_passes(pipeline).first().copied().unwrap_or(u32::MAX)
    }

    fn frame_ty(&self, binding: u32) -> Ty {
        self.machine
            .frame
            .iter()
            .find(|(path, name, _)| self.machine.linker.binding_of(path, name) == Some(binding))
            .map_or(Ty::Uniform, |(_, _, ty)| *ty)
    }

    fn core_frame(&mut self, binding: u32, ty: Ty) -> Resolved {
        let found = self
            .machine
            .frame
            .iter()
            .find(|(path, name, _)| self.machine.linker.binding_of(path, name) == Some(binding))
            .filter(|(path, _, _)| self.machine.linker.is_core(path));
        if let Some((_, name, _)) = found.filter(|(path, _, _)| path == crate::shaders::IMPOSTORS_MODULE) {
            return match self.assets.impostors.frame_resource(name) {
                Some(crate::impostors::FrameResource::Buffer(buffer)) => Resolved::Buffer(buffer.clone()),
                Some(crate::impostors::FrameResource::View(view)) => Resolved::View(view.clone()),
                Some(crate::impostors::FrameResource::Sampler(sampler)) => Resolved::Sampler(sampler.clone()),
                None => Resolved::Placeholder(ty),
            };
        }
        let named = found.map(|(_, name, _)| name.clone());
        let target = match named.as_deref() {
            Some(crate::shared::POST) => return Resolved::Buffer(self.post_buffer.clone()),
            Some(crate::shared::LINEAR) => return Resolved::Sampler(self.linear.clone()),
            Some(name) => crate::machine::core_target(name, &self.machine.outputs),
            None => None,
        };
        match target.and_then(|target| self.view_of(target, false).cloned()) {
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
        let device = self.device.clone();
        let made = |label: &'static str, format| {
            let made = Made {
                label,
                format,
                usage,
                dimension: wgpu::TextureDimension::D2,
            };
            Allocated::new(&device, &made, extent, 1)
        };
        let mut targets = vec![
            (Target::Scene, made("engine/scene", crate::machine::SCENE_FORMAT)),
            (Target::Glow, made("engine/glow", crate::machine::GLOW_FORMAT)),
            (Target::Depth, made("engine/depth", DEPTH_FORMAT)),
        ];
        let mut previous = Vec::new();
        for (index, output) in self.machine.outputs.iter().enumerate() {
            targets.push((Target::Output(index as u32), made(output.name, output.format)));
            if output.layers.is_empty() {
                previous.push((Target::Output(index as u32), made(output.name, output.format)));
            }
        }
        for (target, _) in &self.machine.ids.targets {
            if matches!(target, Target::Copy(..))
                && let Some(format) = target_format(*target, &self.machine.outputs)
            {
                targets.push((*target, made("engine/view-copy", format)));
            }
        }
        self.targets = targets;
        self.previous_targets = previous;
        self.debug_targets = DEBUG_FORMATS.iter().map(|format| made("engine/debug-target", *format)).collect();
        self.generation += 1;
    }

    pub(crate) fn enter(&mut self, pass: u32) -> bool {
        let shares = self
            .machine
            .declared
            .passes
            .get(pass as usize)
            .is_some_and(shared);
        if self.merge && shares && self.open.as_ref().is_some_and(|open| open.shared) {
            return false;
        }
        self.leave();
        true
    }

    pub(crate) fn group(&mut self, pass: u32) {
        if !GROUPED || self.labelled == Some(pass) {
            return;
        }
        let name = self.machine.declared.passes.get(pass as usize).map_or("", |declared| declared.name);
        if let Some(encoder) = self.encoder.as_mut() {
            if self.labelled.is_some() {
                encoder.pop_debug_group();
            }
            encoder.push_debug_group(name);
            self.labelled = Some(pass);
        }
    }

    pub(crate) fn leave(&mut self) {
        self.end_render();
        if GROUPED
            && self.labelled.take().is_some()
            && let Some(encoder) = self.encoder.as_mut()
        {
            encoder.pop_debug_group();
        }
    }

    #[expect(clippy::too_many_lines, reason = "longer than the line ceiling; it may only shrink")]
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
        let is_shared = shared(&pass);
        let mut colors: Vec<(u64, Option<wgpu::TextureView>, wgpu::TextureFormat, Ops)> = Vec::new();
        let mut depth: Option<(u64, wgpu::TextureView, wgpu::TextureFormat, Option<Ops>, bool)> = None;
        let mut main = false;
        let ops_of = |id: Option<crate::graph::ResourceId>| {
            planned
                .colors
                .iter()
                .find(|(known, _)| Some(*known) == id)
                .map(|(_, ops)| *ops)
        };
        let depth_of = |id: Option<crate::graph::ResourceId>| -> Option<Option<Ops>> {
            planned.depth.and_then(|(known, ops)| (Some(known) == id).then_some(ops))
        };
        let declared_ops = |access: crate::Access, clear: Option<Clear>, fallback: Clear| match access {
            crate::Access::Load => Ops { load: Load::Load, store: true },
            crate::Access::Clear => Ops { load: Load::Clear(clear.unwrap_or(fallback)), store: true },
        };
        let key = |target: Target| -> u64 {
            let mut hasher = DefaultHasher::new();
            format!("{target:?}").hash(&mut hasher);
            hasher.finish()
        };
        match attach {
            AttachRaw::Declared => {
                if let Some(layer) = layer {
                    if !own_colors_only(&pass) {
                        main = !self.debug;
                        if self.debug {
                            for index in 0..self.debug_targets.len() {
                                let target = Target::Debug(index as u32);
                                let ops = ops_of(self.machine.ids.target(target)).unwrap_or(Ops { load: Load::Load, store: true });
                                let made = &self.debug_targets[index];
                                colors.push((key(target), Some(made.view.clone()), made.texture.format(), ops));
                            }
                        } else {
                            for (target, format, _, written) in set_formats(&self.machine.outputs, Some(layer)) {
                                let id = self.machine.ids.target(target);
                                match ops_of(id).filter(|_| written) {
                                    Some(ops) => colors.push((key(target), self.view_of(target, true).cloned(), format, ops)),
                                    None => colors.push((key(target), None, format, Ops { load: Load::Load, store: false })),
                                }
                            }
                        }
                    }
                    if let (Some(id), Some(made)) = (self.machine.ids.target(Target::Depth), self.view_of(Target::Depth, true)) {
                        let ops = depth_of(Some(id)).unwrap_or(Some(Ops { load: Load::Load, store: true }));
                        depth = Some((key(Target::Depth), made.clone(), DEPTH_FORMAT, ops, true));
                    }
                }
                if pass.name == crate::shared::COMPOSITE || pass.name == crate::shared::DEBUG_VIEW {
                    let id = self.machine.ids.target(Target::Canvas);
                    if let (Some(ops), Some(canvas)) = (ops_of(id), self.canvas.clone()) {
                        let format = self.canvas_format();
                        colors.insert(0, (key(Target::Canvas), Some(canvas), format, ops));
                    }
                }
                if pass.name == crate::shared::COMPOSITE {
                    for (target, format, _, _) in set_formats(&self.machine.outputs, None).into_iter().skip(2).filter(|(_, _, _, written)| *written) {
                        let id = self.machine.ids.target(target);
                        let ops = ops_of(id).unwrap_or(Ops { load: Load::Load, store: true });
                        colors.push((key(target), self.view_of(target, true).cloned(), format, ops));
                    }
                }
                for &(texture, access) in &pass.colors {
                    let id = self.machine.ids.textures.get(texture as usize).copied();
                    let ops = ops_of(id).unwrap_or_else(|| declared_ops(access, pass.clear_of(texture), Clear::Color(crate::graph::TRANSPARENT)));
                    let format = self.machine.declared.textures[texture as usize].format;
                    let view = self.attachment_view(texture, 0, 0);
                    colors.push((u64::from(texture) + 1_000, view, format, ops));
                }
                if depth.is_none()
                    && let Some(&(texture, access)) = pass.depths.first()
                {
                    let id = self.machine.ids.textures.get(texture as usize).copied();
                    let ops = depth_of(id).unwrap_or_else(|| Some(declared_ops(access, pass.clear_of(texture), Clear::Depth(FAR))));
                    let format = self.machine.declared.textures[texture as usize].format;
                    if let Some(view) = self.attachment_view(texture, 0, 0) {
                        depth = Some((u64::from(texture) + 2_000, view, format, ops, false));
                    }
                }
            }
            AttachRaw::Color(texture, mip, ref layers) => {
                let Some(&(_, access)) = pass.colors.iter().find(|(known, _)| *known == texture) else {
                    return self.mismatch(pass.name);
                };
                let id = self.machine.ids.textures.get(texture as usize).copied();
                let planned = ops_of(id).unwrap_or_else(|| declared_ops(access, pass.clear_of(texture), Clear::Color(crate::graph::TRANSPARENT)));
                let ops = Ops { store: true, ..planned };
                let format = self.machine.declared.textures[texture as usize].format;
                for at in layers.clone() {
                    let view = self.attachment_view(texture, mip, at);
                    let attachment = (u64::from(texture) * 4_096 + u64::from(at)) * 64 + u64::from(mip) + 10_000;
                    colors.push((attachment, view, format, ops));
                }
            }
            AttachRaw::Depth(texture, layer) => {
                if !self.depth_attachment(&pass, texture, layer.unwrap_or(0), &mut depth) {
                    return self.mismatch(pass.name);
                }
            }
            AttachRaw::ColorDepth(color, layer, texture) => {
                let Some(&(_, access)) = pass.colors.iter().find(|(known, _)| *known == color) else {
                    return self.mismatch(pass.name);
                };
                let id = self.machine.ids.textures.get(color as usize).copied();
                let planned = ops_of(id).unwrap_or_else(|| declared_ops(access, pass.clear_of(color), Clear::Color(crate::graph::TRANSPARENT)));
                let ops = Ops { store: true, ..planned };
                let format = self.machine.declared.textures[color as usize].format;
                let at = layer.unwrap_or(0);
                let view = self.attachment_view(color, 0, at);
                colors.push(((u64::from(color) * 4_096 + u64::from(at)) * 64 + 10_000, view, format, ops));
                if !self.depth_attachment(&pass, texture, 0, &mut depth) {
                    return self.mismatch(pass.name);
                }
            }
        }
        let attachments: Vec<u64> = colors.iter().map(|(key, ..)| *key).chain(depth.as_ref().map(|(key, ..)| *key)).collect();
        if let Some(open) = &mut self.open
            && self.merge
            && open.shared
            && is_shared
            && open.attachments == attachments
        {
            if open.offset != offset {
                open.offset = offset;
                open.set.frame = None;
            }
            if GROUPED && open.label != Some(self.pass) && self.labelled != Some(self.pass) {
                if open.label.is_some() {
                    open.pass.pop_debug_group();
                }
                open.pass.push_debug_group(pass.name);
                open.label = Some(self.pass);
            }
            return true;
        }
        self.end_render();
        let current = self.pass;
        self.group(current);
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
            depth_ops: ops.map(depth_ops),
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
            main,
            offset,
            shared: is_shared,
            label: None,
            set: Set::default(),
        });
        true
    }

    fn mismatch(&mut self, name: &str) -> bool {
        let error = fault(self.path(&format!("passes.{name}.attach")), ErrorCode::PassMismatch);
        self.fault(error);
        false
    }

    fn depth_attachment(
        &mut self,
        pass: &crate::install::PassDecl,
        texture: u32,
        layer: u32,
        depth: &mut Option<(u64, wgpu::TextureView, wgpu::TextureFormat, Option<Ops>, bool)>,
    ) -> bool {
        let Some(&(_, access)) = pass.depths.iter().find(|(known, _)| *known == texture) else {
            return false;
        };
        let id = self.machine.ids.textures.get(texture as usize).copied();
        let planned = self
            .planned
            .as_ref()
            .and_then(|planned| planned.depth)
            .and_then(|(known, ops)| (Some(known) == id).then_some(ops));
        let ops = planned.flatten().map(|ops| Ops { store: true, ..ops }).or_else(|| {
            Some(match access {
                crate::Access::Load => Ops { load: Load::Load, store: true },
                crate::Access::Clear => Ops {
                    load: Load::Clear(pass.clear_of(texture).unwrap_or(Clear::Depth(FAR))),
                    store: true,
                },
            })
        });
        let format = self.machine.declared.textures[texture as usize].format;
        if let Some(view) = self.attachment_view(texture, 0, layer) {
            *depth = Some(((u64::from(texture) * 4_096 + u64::from(layer)) * 64 + 20_000, view, format, ops, false));
        }
        true
    }

    pub(crate) fn canvas_format(&self) -> wgpu::TextureFormat {
        self.surface
    }

    fn attachment_view(&mut self, texture: u32, mip: u32, layer: u32) -> Option<wgpu::TextureView> {
        let parity = self.parity;
        let made = self.textures.get_mut(texture as usize)?;
        let made = match parity {
            1 if made[1].is_some() => made[1].as_mut()?,
            _ => made[0].as_mut()?,
        };
        made.slice(&(mip..mip + 1), &(layer..layer + 1), Some(wgpu::TextureViewDimension::D2)).cloned()
    }

    pub(crate) fn end_render(&mut self) {
        if let Some(mut open) = self.open.take()
            && GROUPED
            && open.label.is_some()
        {
            open.pass.pop_debug_group();
        }
    }

    fn pipeline_of(&self, handle: u32) -> Option<(u32, Option<usize>)> {
        let count = self.machine.pipelines.len() as u32;
        if handle < count {
            return Some((handle, None));
        }
        let filled = (handle - count) as usize;
        Some((self.filled.get(filled)?.base, Some(filled)))
    }

    fn render_pipeline(&mut self, pipeline: u32, variant: Option<Specialized>) -> Option<(u64, wgpu::RenderPipeline)> {
        let open = self.open.as_ref()?;
        let (formats, depth, debug, main) = (open.formats.clone(), open.depth, open.debug, open.main);
        self.compile(pipeline, (&formats, depth), (debug, main), variant)
    }

    #[expect(clippy::too_many_lines, reason = "longer than the line ceiling; it may only shrink")]
    #[expect(clippy::disallowed_methods, reason = "the machine keys every pipeline, which the pipeline list records")]
    pub(crate) fn compile(
        &mut self,
        pipeline: u32,
        (formats, depth): (&[Option<wgpu::TextureFormat>], Option<(wgpu::TextureFormat, bool)>),
        (debug, main): (bool, bool),
        variant: Option<Specialized>,
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
        let material: Vec<(&str, f64)> = variant.map_or_else(Vec::new, |(_, constants)| constants.to_vec());
        let specialized = built.stages.iter().any(|stage| !stage.composed.constants(&material).is_empty());
        let bits = variant.filter(|_| specialized).map(|(bits, _)| bits);
        let mut hasher = DefaultHasher::new();
        formats.hash(&mut hasher);
        depth.hash(&mut hasher);
        debug.hash(&mut hasher);
        main.hash(&mut hasher);
        switches.hash(&mut hasher);
        bits.hash(&mut hasher);
        let key = hasher.finish();
        if let Some((_, made)) = built.made.iter().find(|(known, _)| *known == key) {
            return Some((key, made.clone()));
        }
        let constants: Vec<Vec<(String, f64)>> = built
            .stages
            .iter()
            .map(|stage| {
                let mut values: Vec<(String, f64)> = switches
                    .iter()
                    .filter(|(id, _)| stage.composed.switches.contains(id))
                    .map(|(id, on)| (id.to_string(), f64::from(u8::from(*on))))
                    .collect();
                values.extend(stage.composed.constants(&material).into_iter().map(|(name, value)| (name.to_owned(), value)));
                values
            })
            .collect();
        let named: Vec<Vec<(&str, f64)>> = constants
            .iter()
            .map(|values| values.iter().map(|(name, value)| (name.as_str(), *value)).collect())
            .collect();
        let options = |stage: usize| wgpu::PipelineCompilationOptions {
            constants: named.get(stage).map_or(&[][..], Vec::as_slice),
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
                DepthUse::Compare { compare, write, bias } => (write, compare, bias),
            };
            wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: wgpu::StencilState::default(),
                bias,
            }
        });
        let fragment_stage = fragment_entry.and_then(|_| built.stages.get(fragment_at));
        let written = fragment_stage
            .zip(fragment_entry)
            .and_then(|(stage, entry)| fragment_locations(&stage.composed.text, entry.name()));
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let targets: Vec<Option<wgpu::ColorTargetState>> = formats
            .iter()
            .enumerate()
            .map(|(location, format)| {
                let format = (*format)?;
                let (blend, mask) = if debug {
                    let blend = (location == DEBUG_COUNT).then_some(wgpu::BlendState { color: add, alpha: add });
                    (blend, wgpu::ColorWrites::ALL)
                } else {
                    match &desc.targets {
                        Some(targets) => targets.get(location).copied().unwrap_or((None, wgpu::ColorWrites::empty())),
                        None => ((!main || location == 0).then_some(desc.blend).flatten(), desc.writes),
                    }
                };
                let outputs = written.as_ref().is_none_or(|written| written.contains(&(location as u32)));
                Some(wgpu::ColorTargetState {
                    format,
                    blend: blend.filter(|_| blendable(format)),
                    write_mask: if outputs { mask } else { wgpu::ColorWrites::empty() },
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
        crate::pipelines::count(1);
        let compiled = Compiled {
            pipeline,
            formats: formats.to_vec(),
            depth,
            debug,
            main,
            variant: bits,
            constants: if bits.is_some() {
                material.iter().map(|&(name, value)| (name.to_owned(), value)).collect()
            } else {
                Vec::new()
            },
        };
        if self.listed.as_ref().is_some_and(|listed| !listed.contains(&compiled)) {
            self.unlisted += 1;
        }
        self.compiled.push(compiled);
        Some((key, made))
    }

    fn pass_formats(&self, index: u32) -> (Vec<Option<wgpu::TextureFormat>>, Option<(wgpu::TextureFormat, bool)>, bool) {
        let Some(pass) = self.machine.declared.passes.get(index as usize) else {
            return (Vec::new(), None, false);
        };
        let mut formats: Vec<Option<wgpu::TextureFormat>> = Vec::new();
        let mut depth = None;
        let mut main = false;
        if let Some(layer) = pass_layer(pass) {
            if !own_colors_only(pass) {
                main = true;
                for (_, format, _, written) in set_formats(&self.machine.outputs, Some(layer)) {
                    formats.push(written.then_some(format));
                }
            }
            depth = Some((DEPTH_FORMAT, true));
        }
        if pass.name == crate::shared::COMPOSITE || pass.name == crate::shared::DEBUG_VIEW {
            formats.insert(0, Some(self.surface));
        }
        if pass.name == crate::shared::COMPOSITE {
            for (_, format, _, written) in set_formats(&self.machine.outputs, None).into_iter().skip(2) {
                if written {
                    formats.push(Some(format));
                }
            }
        }
        for &(texture, _) in &pass.colors {
            formats.push(self.machine.declared.textures.get(texture as usize).map(|texture| texture.format));
        }
        if depth.is_none()
            && let Some(&(texture, _)) = pass.depths.first()
        {
            depth = self.machine.declared.textures.get(texture as usize).map(|texture| (texture.format, false));
        }
        (formats, depth, main)
    }

    pub(crate) fn warm(&mut self) {
        for pipeline in 0..self.machine.pipelines.len() {
            if !matches!(self.machine.pipelines[pipeline].decl.program, Program::Render(_)) {
                continue;
            }
            for pass in self.scope_passes(pipeline) {
                let (formats, depth, main) = self.pass_formats(pass);
                let _ = self.compile(pipeline as u32, (&formats, depth), (false, main), None);
            }
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
                    && self.machine.declared.passes.get(self.pass as usize).is_some_and(shared)
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

    #[expect(clippy::too_many_lines, reason = "longer than the line ceiling; it may only shrink")]
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
        let variant = match &shape {
            RawShape::Mesh { mesh, material, .. } | RawShape::MeshIndirect { mesh, material, .. } => {
                let extended = self.assets.meshes.get(*mesh).is_some_and(|gpu| gpu.extended);
                let known = material.is_none_or(|handle| self.assets.materials.group(handle).is_some());
                known.then(|| {
                    let variant = self.assets.materials.binding(*material).1.with_extended(extended);
                    (variant.bits(), variant.constants())
                })
            }
            RawShape::Pulled { .. } | RawShape::Buffers { .. } | RawShape::Indirect { .. } => None,
        };
        let Some((key, made)) = self.render_pipeline(pipeline, variant) else {
            return;
        };
        let parity = self.parity;
        let built = &self.machine.pipelines[pipeline as usize];
        let variant = built.variant;
        let mesh_slots = match &built.decl.program {
            Program::Render(desc) => desc.vertices.iter().filter(|layout| matches!(layout, VertexLayout::Mesh)).count() as u32,
            Program::Compute(_) => 0,
        };
        let frame_group = self.frame_groups.get(variant as usize).and_then(|groups| groups.0[parity].clone());
        let own: Vec<(u32, Option<wgpu::BindGroup>)> = built
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
        let stream_inputs: Vec<(Option<wgpu::Buffer>, u64)> = match &shape {
            RawShape::Mesh { streams, .. } | RawShape::MeshIndirect { streams, .. } => {
                streams.iter().map(|(input, offset)| (self.input(part, *input), *offset)).collect()
            }
            RawShape::Pulled { .. } | RawShape::Buffers { .. } | RawShape::Indirect { .. } => Vec::new(),
        };
        let mesh = match &shape {
            RawShape::Mesh { mesh, material, .. } | RawShape::MeshIndirect { mesh, material, .. } => {
                let Some(draw) = self.assets.meshes.draw(*mesh) else {
                    return;
                };
                let group = match material {
                    Some(handle) => match self.assets.materials.group(*handle) {
                        Some(group) => group.clone(),
                        None => return,
                    },
                    None => self.assets.materials.binding(None).0.clone(),
                };
                Some((draw.vertices.clone(), draw.extras.cloned(), draw.indices.clone(), draw.first_index, draw.base_vertex, draw.index_count, group))
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
                if mesh_slots > 1
                    && let Some(extras) = &extras
                {
                    render.set_vertex_buffer(1, extras.slice(..));
                }
                for (at, (buffer, start)) in stream_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(mesh_slots + at as u32, buffer.slice(*start..));
                    }
                }
                render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                render.draw_indexed(first..first + count, base, instances);
            }
            RawShape::MeshIndirect { offset, .. } => {
                let Some((vertices, extras, indices, _, _, _, group)) = mesh else { return };
                let Some(args) = args_input else { return };
                render.set_bind_group(1, &group, &[]);
                render.set_vertex_buffer(0, vertices.slice(..));
                if mesh_slots > 1
                    && let Some(extras) = &extras
                {
                    render.set_vertex_buffer(1, extras.slice(..));
                }
                for (at, (buffer, start)) in stream_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(mesh_slots + at as u32, buffer.slice(*start..));
                    }
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
                    match texture.and_then(|texture| self.assets.materials.texture_view(texture, true)) {
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

    pub(crate) fn within(&mut self, size: (u32, u32)) {
        if let Some(open) = self.open.as_mut() {
            crate::targets::draw_within(&mut open.pass, size);
        }
    }

    pub(crate) fn reset_scissor(&mut self) {
        let size = self.targets.first().map(|(_, made)| (made.size.width, made.size.height));
        if let (Some(open), Some((width, height))) = (self.open.as_mut(), self.region.or(size)) {
            open.pass.set_scissor_rect(0, 0, width, height);
        }
    }

    pub(crate) fn dispatch(&mut self, part: u32, handle: u32, count: CountRaw) -> u32 {
        self.end_render();
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
        let label = self.machine.declared.passes.get(self.pass as usize).map_or("engine/part-dispatch", |pass| pass.name);
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
        let current = self.pass;
        self.group(current);
        let Some(encoder) = self.encoder.as_mut() else {
            return 0;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
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

    fn copied_texture(&self, copied: Copied) -> Option<wgpu::Texture> {
        match copied {
            Copied::Texture(texture) => self.written_texture(texture).map(|made| made.texture.clone()),
            Copied::Target(view) => {
                let target = final_target(view);
                match target {
                    Target::Debug(index) => self.debug_targets.get(index as usize).map(|made| made.texture.clone()),
                    Target::Scene | Target::Glow | Target::Depth | Target::Output(_) | Target::Canvas | Target::Copy(..) => {
                        self.targets.iter().find(|(known, _)| *known == target).map(|(_, made)| made.texture.clone())
                    }
                }
            }
            Copied::Buffer(_) => None,
        }
    }

    pub(crate) fn copy(&mut self, pass: u32) {
        self.end_render();
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
        let (source, target) = (self.copied_texture(from), self.copied_texture(to));
        let buffer = |copied: Copied| match copied {
            Copied::Buffer(buffer) => self.buffers.get(buffer as usize).and_then(Option::as_ref).map(|(made, _)| made.clone()),
            Copied::Texture(_) | Copied::Target(_) => None,
        };
        let (from_buffer, to_buffer) = (buffer(from), buffer(to));
        self.group(pass);
        let Some(encoder) = self.encoder.as_mut() else {
            return;
        };
        for region in &regions {
            match region {
                crate::CopyRegion::Bytes { from: at, to: into, size } => {
                    if let (Some(source), Some(target)) = (&from_buffer, &to_buffer) {
                        encoder.copy_buffer_to_buffer(source, *at, target, *into, *size);
                    }
                }
                crate::CopyRegion::Texels { from: at, to: into, size } => {
                    if let (Some(source), Some(target)) = (&source, &target) {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo { texture: source, mip_level: 0, origin: *at, aspect: wgpu::TextureAspect::All },
                            wgpu::TexelCopyTextureInfo { texture: target, mip_level: 0, origin: *into, aspect: wgpu::TextureAspect::All },
                            *size,
                        );
                    }
                }
                crate::CopyRegion::Levels { from: (from_mip, at), to: (to_mip, into), size } => {
                    if let (Some(source), Some(target)) = (&source, &target) {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo { texture: source, mip_level: *from_mip, origin: *at, aspect: wgpu::TextureAspect::All },
                            wgpu::TexelCopyTextureInfo { texture: target, mip_level: *to_mip, origin: *into, aspect: wgpu::TextureAspect::All },
                            *size,
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn clear_buffer(&mut self, buffer: u32) {
        self.end_render();
        let Some(Some((made, _))) = self.buffers.get(buffer as usize) else {
            return;
        };
        let made = made.clone();
        let current = self.pass;
        self.group(current);
        if let Some(encoder) = self.encoder.as_mut() {
            encoder.clear_buffer(&made, 0, None);
        }
    }

    pub(crate) fn copy_view(&mut self, layer: crate::Layer, view: ViewTarget) {
        self.end_render();
        let from = final_target(view);
        let source = self.targets.iter().find(|(known, _)| *known == from).map(|(_, made)| made.texture.clone());
        let target = self.targets.iter().find(|(known, _)| *known == Target::Copy(layer, view)).map(|(_, made)| made.texture.clone());
        if let (Some(source), Some(target), Some(encoder)) = (source, target, self.encoder.as_mut()) {
            encoder.copy_texture_to_texture(source.as_image_copy(), target.as_image_copy(), source.size());
        }
    }

    pub(crate) fn record_readbacks(&mut self) {
        self.leave();
        let asked = std::mem::take(&mut self.asked);
        let (slots, most) = (self.readback_slots, self.readback_slot_bytes);
        for request in asked {
            let Some(Some(buffer)) = self.buffers.get(request.buffer as usize).map(|made| made.as_ref().map(|(buffer, _)| buffer.clone())) else {
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
            let once = most.is_some_and(|most| bytes > most);
            let idle = |slot: &Slot| !slot.once && slot.state.load(Ordering::Acquire) == 0 && slot.bytes >= bytes;
            if once || (!self.rings[ring].slots.iter().any(idle) && self.rings[ring].kept() < slots) {
                let staging = crate::memory::buffer(
                    &self.device,
                    &wgpu::BufferDescriptor {
                        label: Some("engine/readback"),
                        size: bytes.max(4),
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                );
                self.rings[ring].slots.push(Slot {
                    buffer: staging,
                    bytes,
                    state: Arc::new(AtomicU8::new(0)),
                    asked: None,
                    once,
                });
                if once {
                    self.overflows += 1;
                }
            }
            let found = if once {
                self.rings[ring].slots.last_mut()
            } else {
                self.rings[ring].slots.iter_mut().find(|slot| idle(slot))
            };
            let Some(slot) = found else {
                continue;
            };
            let Some(encoder) = self.encoder.as_mut() else {
                continue;
            };
            encoder.copy_buffer_to_buffer(&buffer, request.range.start, &slot.buffer, 0, bytes);
            if request.then == AfterRead::Clear {
                encoder.clear_buffer(&buffer, request.range.start, Some(bytes));
            }
            slot.state.store(1, Ordering::Release);
            slot.asked = Some((self.asked_frame, request.deliver));
        }
    }

    pub(crate) fn map_readbacks(&mut self) {
        for ring in &self.rings {
            for slot in &ring.slots {
                if slot.state.load(Ordering::Acquire) != 1 {
                    continue;
                }
                slot.state.store(2, Ordering::Release);
                let mapped = Arc::clone(&slot.state);
                slot.buffer.slice(..slot.bytes).map_async(
                    wgpu::MapMode::Read,
                    crate::platform::map_callback(move |result| {
                        mapped.store(if result.is_ok() { 3 } else { 0 }, Ordering::Release);
                    }),
                );
            }
        }
    }

    pub(crate) fn deliver(&mut self, world: &mut World) {
        #[cfg(not(target_arch = "wasm32"))]
        crate::platform::run_arrived_maps();
        for ring in &mut self.rings {
            for slot in &mut ring.slots {
                if slot.state.load(Ordering::Acquire) != 3 {
                    continue;
                }
                if let Some((frame, deliver)) = slot.asked.take()
                    && let Ok(view) = slot.buffer.slice(..slot.bytes).get_mapped_range()
                {
                    deliver(world, frame, ring.tag, &view);
                }
                slot.buffer.unmap();
                slot.state.store(0, Ordering::Release);
            }
            ring.slots.retain(|slot| !(slot.once && slot.state.load(Ordering::Acquire) == 0 && slot.asked.is_none()));
        }
    }
}

fn blendable(format: wgpu::TextureFormat) -> bool {
    format
        .sample_type(None, None)
        .is_some_and(|sample| matches!(sample, wgpu::TextureSampleType::Float { filterable: true }))
}

fn fragment_locations(text: &str, entry: &str) -> Option<Vec<u32>> {
    let at = text.find(&format!("fn {entry}("))?;
    let rest = &text[at..];
    let (arrow, brace) = (rest.find("->")?, rest.find('{')?);
    if arrow > brace {
        return None;
    }
    let returns = rest[arrow + 2..brace].trim();
    if returns.starts_with('@') {
        return Some(locations(returns));
    }
    let name: String = returns
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    let start = text.find(&format!("struct {name} {{"))?;
    let body = &text[start..];
    Some(locations(&body[..body.find('}')?]))
}

fn locations(text: &str) -> Vec<u32> {
    text.match_indices("@location(")
        .filter_map(|(at, found)| {
            let after = &text[at + found.len()..];
            after[..after.find(')')?].trim().parse().ok()
        })
        .collect()
}

pub(crate) enum Resolved {
    Buffer(wgpu::Buffer),
    Dynamic(wgpu::Buffer, u64),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
    Placeholder(Ty),
}

#[derive(Clone, Debug)]
pub(crate) enum AttachRaw {
    Declared,
    Depth(u32, Option<u32>),
    Color(u32, u32, Range<u32>),
    ColorDepth(u32, Option<u32>, u32),
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

fn placeholder(device: &wgpu::Device, ty: Ty) -> Placeholder {
    use crate::linker::Sample;
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
