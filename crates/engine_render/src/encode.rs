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
use crate::profiling::GpuProfiler;
use crate::registration::Installed;
use crate::{AfterRead, CameraUniform, DepthUse, PassNames, VertexLayout, ViewTarget};

mod compile;
mod copies;
mod draw;
mod passes;
mod views;

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

const PLACEHOLDER_BYTES: u64 = 4_096;

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
    pub(crate) once: bool,
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
    pub(crate) layered: bool,
    pub(crate) label: Option<u32>,
    pub(crate) set: Set,
}

pub(crate) struct Assets {
    pub(crate) meshes: crate::mesh_arena::GpuMeshes,
    pub(crate) materials: crate::materials::MaterialStore,
    pub(crate) impostors: crate::impostors::Impostors,
}

pub(crate) struct Delivered {
    frame: u64,
    tag: u32,
    bytes: Vec<u8>,
    deliver: fn(&mut World, u64, u32, &[u8]),
}

#[derive(Default)]
pub(crate) struct Queued {
    pub(crate) before: Option<wgpu::CommandEncoder>,
    pub(crate) delivered: Vec<Delivered>,
    pub(crate) refused: Option<Error>,
    pub(crate) cleared: Vec<u32>,
    pub(crate) ending: Vec<(Copied, Copied, Vec<crate::CopyRegion>)>,
    pub(crate) immediate: u32,
    pub(crate) staging: crate::staging::StagingRing,
}

#[derive(Default)]
pub(crate) struct Uploads {
    pub(crate) cameras: Vec<u8>,
    pub(crate) pieces: Option<Vec<u8>>,
    pub(crate) counted: u64,
    pub(crate) memory: Vec<(&'static str, u64)>,
}

#[derive(Default)]
pub(crate) struct Timed {
    pub(crate) asked: Option<&'static str>,
    pub(crate) open: Option<&'static str>,
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
    pub(crate) canvas_texture: Option<wgpu::Texture>,
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
    pub(crate) uploads: Uploads,
    pub(crate) frame_groups: Vec<([Option<wgpu::BindGroup>; 2], u64)>,
    pub(crate) own_groups: Vec<(Vec<[Option<wgpu::BindGroup>; 2]>, u64)>,
    pub(crate) filled: Vec<Filled>,
    pub(crate) wanted_textures: Vec<TextureHandle>,
    pub(crate) switches: Vec<(u32, bool)>,
    pub(crate) placeholders: Vec<((Ty, u64), Placeholder)>,
    pub(crate) sized: Vec<((Option<usize>, u32, u32), u64)>,
    pub(crate) encoder: Option<wgpu::CommandEncoder>,
    pub(crate) open: Option<Open>,
    pub(crate) computing: Option<(u32, wgpu::ComputePass<'static>)>,
    pub(crate) timed: Timed,
    pub(crate) queued: Queued,
    pub(crate) resolve: Option<(PassNames, &'static str)>,
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
    pub(crate) part_draws: Vec<u32>,
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
    IndirectAt { vertices: Vec<(u32, u64)>, indices: Option<u32>, args: u32, offset: u64 },
    Mesh { mesh: MeshHandle, material: Option<MaterialHandle>, instances: Range<u32>, streams: Vec<(u32, u64)> },
    MeshIndirect { mesh: MeshHandle, material: Option<MaterialHandle>, args: u32, offset: u64, streams: Vec<(u32, u64)> },
}

fn fault(path: String, code: ErrorCode) -> Error {
    Error::new(code, path)
}

fn bound_texture(bound: &Bound) -> Option<u32> {
    match bound {
        Bound::Texture(texture) | Bound::Slice { texture, .. } => Some(*texture),
        Bound::Buffer(_) | Bound::Window { .. } | Bound::Sampler(_) | Bound::Target(_) | Bound::Asset => None,
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
            canvas_texture: None,
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
            uploads: Uploads::default(),
            frame_groups,
            own_groups,
            filled: Vec::new(),
            wanted_textures: Vec::new(),
            switches: Vec::new(),
            placeholders: Vec::new(),
            sized: Vec::new(),
            encoder: None,
            open: None,
            computing: None,
            timed: Timed::default(),
            queued: Queued::default(),
            resolve: None,
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
            part_draws: vec![0; count],
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
        self.allocate_with(texture, None, size, mips);
    }

    pub(crate) fn allocate_as(&mut self, texture: u32, format: wgpu::TextureFormat, size: wgpu::Extent3d, mips: u32) {
        self.allocate_with(texture, Some(format), size, mips);
    }

    fn allocate_with(&mut self, texture: u32, format: Option<wgpu::TextureFormat>, size: wgpu::Extent3d, mips: u32) {
        let Some(decl) = self.machine.declared.textures.get(texture as usize) else {
            return;
        };
        let at = texture as usize;
        let wanted = format.unwrap_or(decl.format);
        let same = self.textures[at][0].as_ref().is_some_and(|made| made.texture.format() == wanted);
        if self.sizes[at] == Some((size, mips)) && same {
            return;
        }
        let mut usage = texture_usage(&self.machine, texture);
        if format.is_some() {
            if mips > 1 {
                usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
            }
            usage &= wanted.guaranteed_format_features(self.device.features()).allowed_usages;
        }
        let made = Made {
            label: decl.label,
            format: wanted,
            usage,
            dimension: self.dimensions.get(at).copied().unwrap_or(wgpu::TextureDimension::D2),
        };
        let doubled = self.doubled(texture);
        for held in self.textures[at].iter().flatten() {
            crate::memory::release_any(&held.texture);
        }
        self.textures[at] = [
            Some(Allocated::new(&self.device, &made, size, mips)),
            doubled.then(|| Allocated::new(&self.device, &made, size, mips)),
        ];
        self.sizes[at] = Some((size, mips));
        self.generation += 1;
    }

    pub(crate) fn grow(&mut self, texture: u32, size: wgpu::Extent3d, mips: u32) {
        let at = texture as usize;
        let Some(Some((known, levels))) = self.sizes.get(at).copied() else {
            self.allocate(texture, size, mips);
            return;
        };
        if (known, levels) == (size, mips) {
            self.allocate(texture, size, mips);
            return;
        }
        let previous = std::mem::replace(&mut self.textures[at], [None, None]);
        self.sizes[at] = None;
        self.allocate(texture, size, mips);
        if levels == mips {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/part-grow"),
            });
            for (old, new) in previous.iter().zip(&self.textures[at]) {
                let (Some(old), Some(new)) = (old, new) else {
                    continue;
                };
                for level in 0..old.mips.min(new.mips) {
                    let extent = wgpu::Extent3d {
                        width: (known.width.min(size.width) >> level).max(1),
                        height: (known.height.min(size.height) >> level).max(1),
                        depth_or_array_layers: old.layers().min(new.layers()),
                    };
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &old.texture,
                            mip_level: level,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &new.texture,
                            mip_level: level,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        extent,
                    );
                }
            }
            crate::gpu_calls::submit(&self.queue, [encoder.finish()]);
            self.queued.immediate += 1;
        }
        for held in previous.iter().flatten() {
            crate::memory::release_any(&held.texture);
        }
    }

    pub(crate) fn release(&mut self, texture: u32) {
        let at = texture as usize;
        let Some(held) = self.textures.get_mut(at) else {
            return;
        };
        if held.iter().all(Option::is_none) {
            return;
        }
        for made in held.iter().flatten() {
            crate::memory::release_any(&made.texture);
        }
        *held = [None, None];
        self.sizes[at] = None;
        self.generation += 1;
    }

    pub(crate) fn renew_buffer(&mut self, buffer: u32, bytes: u64) {
        self.release_buffer(buffer);
        self.allocate_buffer(buffer, bytes);
    }

    pub(crate) fn clear_buffer_range(&mut self, buffer: u32, range: Range<u64>) {
        let Some(Some((made, size))) = self.buffers.get(buffer as usize) else {
            let error = fault(self.path("clear_buffer_range"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        if range.start > range.end || range.end > *size || range.start % 4 != 0 || range.end % 4 != 0 {
            let error = fault(self.path("clear_buffer_range"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        let made = made.clone();
        let device = &self.device;
        let encoder = self.queued.before.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/part-clears"),
            })
        });
        encoder.clear_buffer(&made, range.start, Some(range.end - range.start));
    }

    pub(crate) fn release_buffer(&mut self, buffer: u32) {
        let Some(held) = self.buffers.get_mut(buffer as usize) else {
            return;
        };
        if let Some((made, _)) = held.take() {
            crate::memory::release_buffer(&made);
            self.generation += 1;
        }
    }

    pub(crate) fn release_part(&mut self, part: u32) {
        let textures: Vec<u32> = (0..self.machine.declared.textures.len() as u32)
            .filter(|&texture| self.machine.declared.textures[texture as usize].part == part)
            .collect();
        for texture in textures {
            self.release(texture);
        }
        let buffers: Vec<u32> = (0..self.machine.declared.buffers.len() as u32)
            .filter(|&buffer| self.machine.declared.buffers[buffer as usize].part == part)
            .collect();
        for buffer in buffers {
            self.release_buffer(buffer);
        }
    }

    pub(crate) fn mipmaps(&mut self, texture: u32, layers: Range<u32>) {
        let Some(made) = self.written_texture(texture).map(|made| made.texture.clone()) else {
            let error = fault(self.path("mipmaps"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        let device = &self.device;
        let encoder = self.queued.before.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/part-mips"),
            })
        });
        for layer in layers {
            self.assets.materials.mips().record_layer(device, encoder, &made, layer);
        }
    }

    pub(crate) fn deliver_bytes(&mut self, tag: u32, bytes: Vec<u8>, deliver: fn(&mut World, u64, u32, &[u8])) {
        self.queued.delivered.push(Delivered {
            frame: self.asked_frame,
            tag,
            bytes,
            deliver,
        });
    }

    pub(crate) fn refuse(&mut self, error: Error) {
        if self.queued.refused.is_none() {
            self.queued.refused = Some(error);
        }
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
        if let Some((held, _)) = &self.buffers[at] {
            crate::memory::release_buffer(held);
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

    pub(crate) fn write_staged(&mut self, buffer: u32, (bytes, stride): (&[u8], usize), ranges: &[Range<usize>]) -> bool {
        let Some(Some((made, size))) = self.buffers.get(buffer as usize) else {
            let error = fault(self.path("write_staged"), ErrorCode::InvalidValue);
            self.fault(error);
            return false;
        };
        let reach = ranges.iter().map(|range| range.end * stride).max().unwrap_or(0);
        if reach as u64 > *size || reach > bytes.len() || stride % 4 != 0 {
            let error = fault(self.path("write_staged"), ErrorCode::OutOfRange);
            self.fault(error);
            return false;
        }
        let made = made.clone();
        let device = &self.device;
        let encoder = self.queued.before.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/pose-staging-copies"),
            })
        });
        let staged = self.queued.staging.record(device, encoder, &made, (bytes, stride), ranges);
        staged == crate::staging::Staged::Recorded
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
}

pub(crate) type TimedDraw<'t> = Option<(&'t GpuProfiler, Box<dyn FnOnce() -> String + 't>)>;

fn timed_draw(render: &mut wgpu::RenderPass<'static>, timed: TimedDraw<'_>, draw: impl FnOnce(&mut wgpu::RenderPass<'static>)) {
    let Some((timer, label)) = timed else {
        draw(render);
        return;
    };
    let began = timer.begin_draw(render, label);
    draw(render);
    timer.end_draw(render, began);
}

fn blendable(format: wgpu::TextureFormat) -> bool {
    format
        .sample_type(None, None)
        .is_some_and(|sample| matches!(sample, wgpu::TextureSampleType::Float { filterable: true }))
}

fn fragment_outputs<'a>(text: &'a str, entry: &str) -> Option<&'a str> {
    let at = text.find(&format!("fn {entry}("))?;
    let rest = &text[at..];
    let (arrow, brace) = (rest.find("->")?, rest.find('{')?);
    if arrow > brace {
        return None;
    }
    let returns = rest[arrow + 2..brace].trim();
    if returns.starts_with('@') {
        return Some(returns);
    }
    let name: String = returns
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    let start = text.find(&format!("struct {name} {{"))?;
    let body = &text[start..];
    Some(&body[..body.find('}')?])
}

fn fragment_locations(text: &str, entry: &str) -> Option<Vec<u32>> {
    fragment_outputs(text, entry).map(locations)
}

fn fragment_kinds(text: &str, entry: &str) -> Option<Vec<(u32, wgpu::TextureSampleType)>> {
    let outputs = fragment_outputs(text, entry)?;
    let kinds = outputs.match_indices("@location(").map(|(at, _)| {
        let field = &outputs[at..];
        let field = &field[..field.find(',').unwrap_or(field.len())];
        if field.contains("u32") {
            wgpu::TextureSampleType::Uint
        } else if field.contains("i32") {
            wgpu::TextureSampleType::Sint
        } else {
            wgpu::TextureSampleType::Float { filterable: true }
        }
    });
    Some(locations(outputs).into_iter().zip(kinds).collect())
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
    Colors(Vec<(u32, u32)>),
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

fn placeholder(device: &wgpu::Device, (ty, bytes): (Ty, u64)) -> Placeholder {
    use crate::linker::Sample;
    let buffer = |usage| {
        Placeholder::Buffer(crate::memory::buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("engine/placeholder"),
                size: bytes.max(PLACEHOLDER_BYTES),
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
