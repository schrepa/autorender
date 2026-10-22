//! Octahedral impostors: an opaque draw group whose mesh has enough
//! triangles draws, where its instances are too small for its texels to
//! show, as cards blending the three frames around the view
//! (`meshes/impostors.wgsl`) of its mesh and material, and of its parts
//! with theirs, seen from a grid of
//! directions, one layer of four texture arrays. Plan batches' frames cover
//! the sphere, eight a side at 16 texels; placed objects', which stand on
//! the ground, the upper hemisphere, twelve a side, each at least as many
//! texels across as the pixels their cards span where they take over
//! (`LevelOfDetail::impostor_pixels`), so no card shows magnified there.
//! Layers hold each surface's color and coverage, normal and depth, and
//! how much of the sky it sees past its own mesh, which shades the sun on
//! cards as the mesh's own shadows would; the light is the scene's, live.
//! Cards read each frame where the line of sight meets the surface the
//! frames' depths hold, so the three show the same point as the view turns.
//!
//! Layers are records (`IMPOSTOR_RECORD`), keyed by the renderer's code,
//! the atlas's shape and format and the content of the mesh and of the
//! material ([`Contents`]): hosts hand back a group's record, which uploads
//! into its layer. Where none holds it the layer is baked on the GPU
//! (`bake`) and read back as its record (`records`), which hosts keep and
//! which then uploads as any record does: how the cook makes them. Atlases
//! hold the compressed family the device samples textures in, BC7 or else
//! ASTC 4 × 4, encoded on the CPU by the encoder every cooked texture is
//! (`engine_textures`); on devices sampling neither, RGBA8, at half the
//! texels where a layer would pass 512 across. Records exist per format:
//! the cook, on a desktop, makes BC7's, and other devices make and keep
//! their own (their materials' textures, of another family, key other
//! records anyway). Records baked on the GPU are cooked once, not compared
//! across GPUs, whose rasterizers differ in their last bits: what reads
//! them takes them as they are.

#![expect(
    clippy::expect_used,
    reason = "the atlas arrays grow before they are read"
)]

mod bake;
mod content;
mod records;

use engine_core::collections::{HashMap, HashSet};
use engine_core::glam::Vec3;
use engine_core::{Material, MaterialHandle, MeshHandle, RecordKey};

use crate::bindings::{IMPOSTOR_BAKE, IMPOSTOR_DRAW};
use crate::cached::Cached;
use crate::gpu_data::Variant;
use crate::gpu_layouts::constants::IMPOSTOR_FOLIAGE as FOLIAGE;
use crate::gpu_layouts::constants::IMPOSTOR_FRAMES_SHIFT as FRAMES_SHIFT;
pub(crate) use crate::gpu_layouts::constants::IMPOSTOR_HEMISPHERE as HEMISPHERE;
use crate::gpu_layouts::constants::IMPOSTOR_MASKED as MASKED;
use crate::gpu_layouts::constants::IMPOSTOR_ROOMS as ROOMS;
use crate::gpu_layouts::constants::VARIANT_FOLIAGE;
pub(crate) use crate::gpu_layouts::impostors::ImpostorRecord;
use crate::mesh_arena::MeshDraw;
use crate::mipmaps::MipGenerator;
use crate::scene_shading::SceneShading;
use crate::targets::Targets;
pub(crate) use content::Contents;

/// The kind of impostors' records.
const IMPOSTOR_RECORD: &str = "impostor";

/// Fewest triangles a mesh needs for an impostor: a card's pixels cost
/// about what fewer triangles would.
pub(crate) const IMPOSTOR_MIN_TRIANGLES: usize = 64;

/// The widest layer RGBA8 atlases hold: wider shapes hold half the texels
/// a frame.
const RGBA8_SIZE: u32 = 512;

/// Record bytes an atlas loads in a frame, past which the records hosts
/// handed back wait for the next: a pack answering for every impostor at
/// once spreads their uploads over frames.
const LOAD_BUDGET: usize = 16 << 20;

/// Frames along each side of placed objects' grids over the upper
/// hemisphere: every view falls among three of them, which cards read
/// where the line of sight meets the surfaces their depths hold.
const PLACED_FRAMES: u32 = 12;

/// Bytes the arrays take at first: room for up to 16 small layers, for as
/// few as one large one.
const FIRST_BYTES: u64 = 16 << 20;

/// How an atlas lays out its impostors: frames along each side of the grid,
/// texels across a frame of the first level, and whether its frames cover
/// only the upper hemisphere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AtlasShape {
    pub frames: u32,
    pub texels: u32,
    pub hemisphere: bool,
}

impl AtlasShape {
    /// Plan batches' impostors: eight frames a side of 16 texels over the
    /// sphere.
    pub(crate) const SPHERE: Self = Self {
        frames: 8,
        texels: 16,
        hemisphere: false,
    };

    /// Placed objects' impostors, whose cards take over where their
    /// bounding spheres span `pixels` of the viewport's height
    /// (`LevelOfDetail::impostor_pixels`): [`PLACED_FRAMES`] a side over
    /// the upper hemisphere, which views from below see from the horizon,
    /// each as many texels across as those pixels, rounded up to a power of
    /// two, so a card is not magnified where it takes over.
    pub(crate) fn placed(pixels: f32) -> Self {
        let texels = (pixels.ceil() as u32).clamp(2, 1 << 15);
        Self {
            frames: PLACED_FRAMES,
            texels: texels.next_power_of_two(),
            hemisphere: true,
        }
    }

    /// Texels across a layer's first level.
    pub(crate) fn size(self) -> u32 {
        self.frames * self.texels
    }

    /// Levels down to two texels a frame.
    pub(crate) fn levels(self) -> u32 {
        self.texels.trailing_zeros()
    }

    /// The flags of records of this shape.
    fn flags(self) -> u32 {
        self.texels | if self.hemisphere { HEMISPHERE } else { 0 } | (self.frames << FRAMES_SHIFT)
    }

    /// The shape an atlas holding `format` holds this shape's impostors at:
    /// compressed atlases' this one; RGBA8's with half the texels a frame
    /// where a layer would pass [`RGBA8_SIZE`] texels across, so atlases
    /// without compression keep to that memory.
    fn held(self, format: AtlasFormat) -> Self {
        match format {
            AtlasFormat::Rgba8 if self.size() > RGBA8_SIZE => Self {
                texels: self.texels / 2,
                ..self
            },
            AtlasFormat::Bc7 | AtlasFormat::Astc | AtlasFormat::Rgba8 => self,
        }
    }

    /// The shape an atlas holding `format` on `device` holds this shape's
    /// impostors at: [`Self::held`]'s, its frames' texels halved until a
    /// layer fits the device's textures and a bake's read back its buffers.
    fn held_on(self, device: &wgpu::Device, format: AtlasFormat) -> Self {
        let limits = device.limits();
        let mut shape = self.held(format);
        while (shape.size() > limits.max_texture_dimension_2d
            || records::readback_bytes(shape) > limits.max_buffer_size)
            && shape.texels > 2
        {
            shape.texels /= 2;
        }
        shape
    }
}

/// What an atlas's arrays hold: blocks of the compressed family the device
/// samples textures in (BC7, else ASTC 4 × 4), or RGBA8 texels on devices
/// sampling neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtlasFormat {
    Bc7,
    Astc,
    Rgba8,
}

/// Albedo and coverage, emissive color and mask (both decode sRGB), then
/// normal and depth, and roughness, metalness, occlusion and the sky the
/// surface sees past its mesh: what bakes draw, and RGBA8 atlases hold.
const FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8Unorm,
];
/// The arrays of BC7 atlases, decoding as [`FORMATS`] do.
const BC7_FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    wgpu::TextureFormat::Bc7RgbaUnorm,
    wgpu::TextureFormat::Bc7RgbaUnorm,
];
/// The arrays of ASTC atlases, decoding as [`FORMATS`] do.
const ASTC_FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    },
];

impl AtlasFormat {
    /// The format `device` holds atlases in: the family it samples cooked
    /// textures of (`Renderer::texture_sampling`), RGBA8 where neither.
    fn of(device: &wgpu::Device) -> Self {
        let features = device.features();
        if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            Self::Bc7
        } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
            Self::Astc
        } else {
            Self::Rgba8
        }
    }

    /// Its tag in records.
    fn tag(self) -> u32 {
        match self {
            Self::Rgba8 => 0,
            Self::Bc7 => 1,
            Self::Astc => 2,
        }
    }

    /// The family its blocks are encoded in; `None` for RGBA8 texels.
    fn family(self) -> Option<engine_textures::Family> {
        match self {
            Self::Bc7 => Some(engine_textures::Family::Bc7),
            Self::Astc => Some(engine_textures::Family::Astc),
            Self::Rgba8 => None,
        }
    }

    /// The bytes of each row of a level `size` texels a side, and its rows:
    /// of 4 × 4 blocks of 16 bytes in BC7 and ASTC, of texels in RGBA8.
    fn rows(self, size: u32) -> (u32, u32) {
        match self {
            Self::Bc7 | Self::Astc => (size.div_ceil(4) * 16, size.div_ceil(4)),
            Self::Rgba8 => (size * 4, size),
        }
    }

    /// Bytes of a level `size` texels a side.
    fn level_bytes(self, size: u32) -> usize {
        let (row, rows) = self.rows(size);
        row as usize * rows as usize
    }

    /// The arrays' texture formats.
    fn textures(self) -> [wgpu::TextureFormat; 4] {
        match self {
            Self::Bc7 => BC7_FORMATS,
            Self::Astc => ASTC_FORMATS,
            Self::Rgba8 => FORMATS,
        }
    }
}

impl ImpostorRecord {
    /// The record of a mesh with bounds `min` to `max` held in `layer` of
    /// an atlas at `shape`, whose material emits `emissive`, which is
    /// `masked` (alpha tested), shades as `foliage` and is divided into
    /// `rooms`.
    pub(crate) fn new(
        min: Vec3,
        max: Vec3,
        layer: u32,
        (emissive, masked, foliage, rooms): ([f32; 3], bool, bool, bool),
        shape: AtlasShape,
    ) -> Self {
        let center = (min + max) * 0.5;
        let half = (max - min) * 0.5;
        // A hair over the half diagonal keeps every vertex inside the frames,
        // and flat bounds still get a radius.
        let radius = engine_core::math::max(half.length() * 1.0001, 1.0e-6);
        Self {
            sphere: center.extend(radius).to_array(),
            extent: half.to_array(),
            layer,
            emissive,
            flags: shape.flags()
                | if masked { MASKED } else { 0 }
                | if foliage { FOLIAGE } else { 0 }
                | if rooms { ROOMS } else { 0 },
        }
    }

    /// The impostor's error in mesh units: a texel of its first level, which
    /// the card resolves no finer; 0 for the zeroed records of draw groups
    /// without an impostor.
    pub(crate) fn error(&self) -> f32 {
        let texels = self.flags & (HEMISPHERE - 1);
        if texels == 0 {
            return 0.0;
        }
        2.0 * self.sphere[3] / texels as f32
    }
}

/// Layers of the texture arrays: free ones are reused lowest first, and the
/// arrays grow by doubling up to `limit` layers.
#[derive(Debug, Default)]
pub(crate) struct Layers {
    free: Vec<u32>,
    used: u32,
}

impl Layers {
    /// A layer for a new impostor, or `None` when `limit` layers are in use.
    pub(crate) fn allocate(&mut self, limit: u32) -> Option<u32> {
        if let Some(index) = self
            .free
            .iter()
            .enumerate()
            .min_by_key(|(_, layer)| **layer)
            .map(|(index, _)| index)
        {
            return Some(self.free.swap_remove(index));
        }
        (self.used < limit).then(|| {
            self.used += 1;
            self.used - 1
        })
    }

    pub(crate) fn release(&mut self, layer: u32) {
        self.free.push(layer);
    }

    /// Layers the arrays must hold: every layer ever handed out.
    pub(crate) fn extent(&self) -> u32 {
        self.used
    }

    /// Layers holding impostors.
    pub(crate) fn live(&self) -> u32 {
        self.used - self.free.len() as u32
    }
}

/// The capacity arrays grow to for `needed` layers of `bytes` each: powers
/// of two from as many as [`FIRST_BYTES`] hold (one to 16), never beyond
/// `limit`.
pub(crate) fn grown_layers(needed: u32, bytes: u64, limit: u32) -> u32 {
    let first = (FIRST_BYTES / bytes.max(1)).clamp(1, 16) as u32;
    needed.max(first).next_power_of_two().min(limit).max(needed)
}

/// Whether what `variant` draws shades as foliage: masked, or of a foliage
/// material, as meshes do.
fn shades_as_foliage(variant: Variant) -> bool {
    variant.alpha_mask() || variant.bits() & VARIANT_FOLIAGE != 0
}

/// The most impostors an atlas holds on `device` within `budget`.
fn layer_limit(device: &wgpu::Device, budget: Option<u32>) -> u32 {
    device
        .limits()
        .max_texture_array_layers
        .min(budget.unwrap_or(u32::MAX))
}

/// The prelude's encoder, created with the first work recorded into it.
fn prelude<'e>(
    device: &wgpu::Device,
    encoder: &'e mut Option<wgpu::CommandEncoder>,
) -> &'e mut wgpu::CommandEncoder {
    encoder.get_or_insert_with(|| {
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/texture-mips"),
        })
    })
}

/// What an impostor depicts: a mesh drawn with a material.
pub(crate) type ImpostorKey = (MeshHandle, Option<MaterialHandle>);

struct Arrays {
    textures: [wgpu::Texture; 4],
    views: [wgpu::TextureView; 4],
    layers: u32,
}

struct Baked {
    record: ImpostorRecord,
    // The material it depicts, to supply one replaced in place anew.
    material: Option<Material>,
}

/// An impostor whose record hosts were asked for: the record, and the
/// material it depicts, to ask anew for one replaced in place.
#[derive(Clone, Copy)]
struct Supply {
    record: RecordKey,
    material: Option<Material>,
}

/// How an impostor stands this frame ([`ImpostorAtlas::supply`]).
pub(crate) enum Supplied {
    /// Resident: its record.
    Resident(ImpostorRecord),
    /// Coming: hosts have yet to answer for its record, or its bake to
    /// read back as one.
    Coming,
    /// Left out: the budget's impostors, or as many as the device's array
    /// layers, are resident.
    Refused,
}

impl Supplied {
    /// The record draws take: the impostor's while resident, zeroed
    /// otherwise.
    pub(crate) fn record(&self) -> ImpostorRecord {
        match self {
            Self::Resident(record) => *record,
            Self::Coming | Self::Refused => ImpostorRecord::default(),
        }
    }
}

/// What a bake draws: the mesh as the arena holds it, with its
/// material's group and pipeline variant, the bounds of everything it
/// draws, and the parts drawn with it, each with a material of its own.
pub(crate) struct BakeSource<'a> {
    pub draw: MeshDraw<'a>,
    pub extended: bool,
    pub group: &'a wgpu::BindGroup,
    pub variant: Variant,
    pub bounds: (Vec3, Vec3),
    pub material: Option<Material>,
    pub parts: Vec<BakePart<'a>>,
}

/// A part a bake draws with its mesh: its own mesh as the arena holds it,
/// with its material's group and pipeline variant.
pub(crate) struct BakePart<'a> {
    pub draw: MeshDraw<'a>,
    pub extended: bool,
    pub group: &'a wgpu::BindGroup,
    pub variant: Variant,
}

/// What supplying an impostor records with: the device, its queue, the
/// prelude's encoder (created with the first work it records), mip levels,
/// the mesh shader and the pipelines every atlas shares.
pub(crate) struct BakeGpu<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut Option<wgpu::CommandEncoder>,
    pub mips: &'a MipGenerator,
    pub shader: &'a wgpu::ShaderModule,
    pub pipelines: &'a mut ImpostorPipelines,
}

/// What every atlas shares: the bakes' and draws' layouts and pipelines,
/// and the draws' sampler.
pub(crate) struct ImpostorPipelines {
    bake_layout: crate::pipelines::GroupLayout,
    bake_pipeline_layout: wgpu::PipelineLayout,
    bake_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
    shade_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
    draw_layout: crate::pipelines::GroupLayout,
    draw_pipelines: HashMap<(Targets, bool), wgpu::RenderPipeline>,
    sampler: wgpu::Sampler,
    // Stands in for the frames' depths while bakes draw them.
    no_shade: Option<wgpu::TextureView>,
}

impl ImpostorPipelines {
    /// `material_layout` binds a material's maps (group 1 of bakes).
    pub(crate) fn new(
        device: &wgpu::Device,
        material_layout: &crate::pipelines::GroupLayout,
        empty_layout: &crate::pipelines::GroupLayout,
    ) -> Self {
        let bake_layout = crate::pipelines::layout(device, &IMPOSTOR_BAKE);
        let bake_pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::IMPOSTOR_BAKE_PIPELINES,
            &[empty_layout, material_layout, &bake_layout],
        );
        let draw_layout = crate::pipelines::layout(device, &IMPOSTOR_DRAW);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/impostor-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            bake_layout,
            bake_pipeline_layout,
            bake_pipelines: HashMap::default(),
            shade_pipelines: HashMap::default(),
            draw_layout,
            draw_pipelines: HashMap::default(),
            sampler,
            no_shade: None,
        }
    }

    /// The layout of the impostor draws' group 1.
    pub(crate) fn draw_layout(&self) -> &crate::pipelines::GroupLayout {
        &self.draw_layout
    }

    /// The draw pipeline into `targets`, shading local lights when `lit`,
    /// once inserted.
    pub(crate) fn pipeline(&self, targets: Targets, lit: bool) -> Option<&wgpu::RenderPipeline> {
        self.draw_pipelines.get(&(targets, lit))
    }

    pub(crate) fn insert_pipeline(
        &mut self,
        targets: Targets,
        lit: bool,
        pipeline: wgpu::RenderPipeline,
    ) {
        self.draw_pipelines.insert((targets, lit), pipeline);
    }

    /// Drops the draw pipelines of `targets`, built over layouts of a
    /// pass that is gone.
    pub(crate) fn forget_pipelines(&mut self, targets: Targets) {
        self.draw_pipelines.retain(|(kind, _), _| *kind != targets);
    }

    /// Drops every draw pipeline, specialized for the scene features
    /// shading looked up before.
    pub(crate) fn forget_shading(&mut self) {
        self.draw_pipelines.clear();
    }
}

pub(crate) struct ImpostorAtlas {
    // The shape its layers hold, and the format of its arrays.
    shape: AtlasShape,
    format: AtlasFormat,
    arrays: Option<Arrays>,
    targets: Option<bake::BakeTargets>,
    layers: Layers,
    baked: HashMap<ImpostorKey, Baked>,
    // Impostors whose records hosts were asked for, those asked for since
    // the last frame's drops, the asks not yet taken and the answers hosts
    // gave; read backs of records made for hosts to keep.
    supplies: HashMap<ImpostorKey, Supply>,
    supplied: HashSet<ImpostorKey>,
    wants: Vec<RecordKey>,
    answers: HashMap<RecordKey, Option<Vec<u8>>>,
    readbacks: Vec<records::Readback>,
    // Record bytes loaded since the last frame was submitted, and whether
    // the budget held an answer back.
    loaded: usize,
    deferred: bool,
    // While a session cooks, every impostor is supplied, whatever the
    // budget.
    cooking: bool,
    // The records last uploaded, and the group binding them with the
    // arrays, keyed by the generations of both.
    records: wgpu::Buffer,
    uploaded: Vec<ImpostorRecord>,
    group: Cached<(u64, u64), wgpu::BindGroup>,
    records_generation: u64,
    arrays_generation: u64,
    // Advances whenever an impostor becomes resident or is dropped, and
    // whenever hosts answer for a record or an answer waits a frame:
    // records carrying impostor errors are rebuilt then.
    generation: u64,
}

fn record_buffer(device: &wgpu::Device, records: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/impostor-records"),
        size: (records.max(1) * size_of::<ImpostorRecord>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl ImpostorAtlas {
    /// An atlas of impostors of `shape`, holding layers as `device` samples
    /// them best.
    pub(crate) fn new(device: &wgpu::Device, shape: AtlasShape) -> Self {
        let format = AtlasFormat::of(device);
        Self {
            shape: shape.held_on(device, format),
            format,
            arrays: None,
            targets: None,
            layers: Layers::default(),
            baked: HashMap::default(),
            supplies: HashMap::default(),
            supplied: HashSet::default(),
            wants: Vec::new(),
            answers: HashMap::default(),
            readbacks: Vec::new(),
            loaded: 0,
            deferred: false,
            cooking: false,
            records: record_buffer(device, 1),
            uploaded: Vec::new(),
            group: Cached::default(),
            records_generation: 0,
            arrays_generation: 0,
            generation: 0,
        }
    }

    /// While a session cooks: every impostor is supplied, whatever the
    /// budget, its record fetched or made.
    pub(crate) fn set_cooking(&mut self, cooking: bool) {
        self.cooking = cooking;
    }

    /// Holds impostors of `shape` from now on. Where `device` holds those
    /// at another shape than the atlas does, its impostors, arrays and
    /// draws' group go, with what it asked hosts for and what it was
    /// reading back, and draws supply them anew at that shape, whose
    /// records are others.
    pub(crate) fn fit(&mut self, device: &wgpu::Device, shape: AtlasShape) {
        let shape = shape.held_on(device, self.format);
        if shape == self.shape {
            return;
        }
        self.shape = shape;
        self.arrays = None;
        self.targets = None;
        self.group = Cached::default();
        self.layers = Layers::default();
        self.baked.clear();
        self.supplies.clear();
        self.supplied.clear();
        self.wants.clear();
        self.answers.clear();
        self.readbacks.clear();
        self.generation += 1;
    }

    /// Advances whenever an impostor becomes resident or is dropped, and
    /// whenever hosts answer for a record or an answer waits a frame.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Impostors resident.
    pub(crate) fn live(&self) -> u32 {
        self.layers.live()
    }

    /// GPU bytes of the arrays and records.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let layers = self.arrays.as_ref().map_or(0, |arrays| arrays.layers);
        u64::from(layers) * self.layer_bytes() + self.records.size()
    }

    /// GPU bytes of a layer: four arrays, a third more for the levels.
    fn layer_bytes(&self) -> u64 {
        self.format.level_bytes(self.shape.size()) as u64 * 4 * 4 / 3
    }

    /// Whether the impostor of `key` is resident.
    pub(crate) fn contains(&self, key: ImpostorKey) -> bool {
        self.baked.contains_key(&key)
    }

    /// The impostor of `key`, when resident for `material`.
    pub(crate) fn record(
        &self,
        key: ImpostorKey,
        material: Option<&Material>,
    ) -> Option<ImpostorRecord> {
        self.baked
            .get(&key)
            .filter(|baked| baked.material.as_ref() == material)
            .map(|baked| baked.record)
    }

    /// Records it asks hosts for since the last call.
    pub(crate) fn take_wants(&mut self, out: &mut Vec<RecordKey>) {
        out.append(&mut self.wants);
    }

    /// Whether it asked hosts for record `record`.
    pub(crate) fn asks(&self, record: RecordKey) -> bool {
        self.supplies.values().any(|supply| supply.record == record)
    }

    /// Hosts' answer for record `record`, which it asked for: its bytes,
    /// or `None` where no pack or store holds it.
    pub(crate) fn answer(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        self.answers.insert(record, bytes);
        self.generation += 1;
    }

    /// Records still to come: asked of hosts, or reading back.
    pub(crate) fn left(&self) -> u64 {
        (self.supplies.len() + self.readbacks.len()) as u64
    }

    /// The key of the record of this atlas's impostor of the mesh and the
    /// material whose contents are `content`.
    fn record_key(&self, content: (u128, u128)) -> RecordKey {
        RecordKey::new(
            &crate::SOURCE,
            IMPOSTOR_RECORD,
            &records::identity(self.shape, self.format, content),
        )
    }

    /// Drops impostors `keep` refuses and those whose material changed
    /// from `material`, freeing their layers, and what was asked of hosts
    /// for those `keep` refuses and for those no draw supplied since the
    /// last frame's drops: an impostor its draws no longer want waits for
    /// no answer (a cooking session's view moved past it), and one wanted
    /// again asks anew.
    pub(crate) fn retain(
        &mut self,
        mut keep: impl FnMut(&ImpostorKey) -> bool,
        mut material: impl FnMut(MaterialHandle) -> Option<Material>,
    ) {
        let layers = &mut self.layers;
        let before = self.baked.len();
        self.baked.retain(|key, baked| {
            let current = key.1.and_then(&mut material);
            let kept = keep(key) && baked.material == current;
            if !kept {
                layers.release(baked.record.layer);
            }
            kept
        });
        if self.baked.len() != before {
            self.generation += 1;
        }
        let supplied = std::mem::take(&mut self.supplied);
        let asked = self.supplies.len();
        self.supplies.retain(|key, _| keep(key));
        let kept = self.supplies.len();
        self.supplies.retain(|key, _| supplied.contains(key));
        if self.cooking && self.supplies.len() < kept {
            eprintln!(
                "Impostors no draw supplies any more stop waiting: {} of {asked}",
                kept - self.supplies.len()
            );
        }
        let supplies = &self.supplies;
        self.answers
            .retain(|record, _| supplies.values().any(|supply| supply.record == *record));
    }

    /// Drops every impostor when more than `budget` are resident: the
    /// draw groups first in order take theirs again up to it. A cooking
    /// session keeps them all.
    pub(crate) fn enforce_budget(&mut self, budget: Option<u32>) {
        if !self.cooking && budget.is_some_and(|budget| self.layers.live() > budget) {
            self.baked.clear();
            self.layers = Layers::default();
            self.generation += 1;
        }
    }

    /// The impostor of `key`, drawing `source`, whose mesh and material
    /// hold `content`: resident; coming, its record asked of hosts, which
    /// uploads into a layer once they hand it back, or, where none holds
    /// it, baked into `gpu`'s encoder and read back as its record, which
    /// then uploads the same way. Refused where `budget` impostors (none
    /// while a session cooks), or as many as the device's array layers,
    /// already are.
    pub(crate) fn supply(
        &mut self,
        mut gpu: BakeGpu<'_>,
        key: ImpostorKey,
        source: &BakeSource<'_>,
        content: (u128, u128),
        budget: Option<u32>,
    ) -> Supplied {
        if let Some(resident) = self.record(key, source.material.as_ref()) {
            return Supplied::Resident(resident);
        }
        self.supplied.insert(key);
        let budget = budget.filter(|_| !self.cooking);
        if self.layers.live() >= layer_limit(gpu.device, budget) {
            self.supplies.remove(&key);
            return Supplied::Refused;
        }
        // A material replaced in place is asked for anew.
        let asked = self
            .supplies
            .get(&key)
            .filter(|supply| supply.material.as_ref() == source.material.as_ref())
            .map(|supply| supply.record);
        let Some(record) = asked else {
            let record = self.record_key(content);
            self.wants.push(record);
            self.supplies.insert(
                key,
                Supply {
                    record,
                    material: source.material,
                },
            );
            return Supplied::Coming;
        };
        // A frame uploads records within a budget: the rest wait.
        if self.loaded >= LOAD_BUDGET && self.answers.get(&record).is_some_and(Option::is_some) {
            self.deferred = true;
            return Supplied::Coming;
        }
        let Some(answer) = self.answers.remove(&record) else {
            return Supplied::Coming;
        };
        // Impostors of meshes and materials holding the same contents share
        // a record: its bytes stay for the others still coming.
        if answer.is_some()
            && self
                .supplies
                .iter()
                .any(|(other, supply)| *other != key && supply.record == record)
        {
            self.answers.insert(record, answer.clone());
        }
        if let Some(bytes) = answer
            && self.load(&mut gpu, key, source, &bytes, budget)
        {
            self.supplies.remove(&key);
            return self.supplied(key, source);
        }
        // What no host held, or a record that does not load, bakes: its
        // record comes as the bake reads back, and answers this supply.
        self.bake(gpu, source, content);
        Supplied::Coming
    }

    /// The impostor of `key`, resident for `source`'s material.
    fn supplied(&self, key: ImpostorKey, source: &BakeSource<'_>) -> Supplied {
        self.record(key, source.material.as_ref())
            .map_or(Supplied::Refused, Supplied::Resident)
    }

    /// A layer for a new impostor, the arrays grown to hold it: `None`
    /// where `budget` impostors, or as many as the device's array layers,
    /// already are.
    fn layer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut Option<wgpu::CommandEncoder>,
        budget: Option<u32>,
    ) -> Option<u32> {
        if self.layers.live() >= layer_limit(device, budget) {
            return None;
        }
        let layer = self
            .layers
            .allocate(device.limits().max_texture_array_layers)?;
        self.ensure_layers(device, encoder, self.layers.extent());
        Some(layer)
    }

    /// `key` resident in `layer`, drawing `source`: its card dissolves as
    /// masked meshes do where its mesh or a part is masked, and shades as
    /// foliage where one is masked or of a foliage material, as they do.
    fn resident(&mut self, key: ImpostorKey, source: &BakeSource<'_>, layer: u32) {
        let emissive = source
            .material
            .as_ref()
            .map_or([0.0; 3], Material::emissive);
        let rooms = source
            .material
            .as_ref()
            .is_some_and(|material| material.windows().is_some());
        let variants = || {
            std::iter::once(source.variant)
                .chain(source.parts.iter().map(|part| part.variant))
        };
        let masked = variants().any(Variant::alpha_mask);
        let foliage = variants().any(shades_as_foliage);
        let record = ImpostorRecord::new(
            source.bounds.0,
            source.bounds.1,
            layer,
            (emissive, masked, foliage, rooms),
            self.shape,
        );
        self.baked.insert(
            key,
            Baked {
                record,
                material: source.material,
            },
        );
        self.generation += 1;
    }

    /// Frees the targets bakes render into until the next bake: a large
    /// atlas's would otherwise stay resident.
    pub(crate) fn release_targets(&mut self) {
        self.targets = None;
    }

    /// Grows the arrays to hold `layers`, copying the resident ones in the
    /// prelude's `encoder`.
    fn ensure_layers(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut Option<wgpu::CommandEncoder>,
        layers: u32,
    ) {
        if self
            .arrays
            .as_ref()
            .is_some_and(|arrays| arrays.layers >= layers)
        {
            return;
        }
        let capacity = grown_layers(
            layers,
            self.layer_bytes(),
            device.limits().max_texture_array_layers,
        );
        let (size, levels) = (self.shape.size(), self.shape.levels());
        let textures = self.format.textures().map(|format| {
            crate::targets::texture_2d(
                device,
                "engine/impostor-array",
                (size, size),
                capacity,
                levels,
                format,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
            )
        });
        if let Some(old) = &self.arrays {
            let encoder = prelude(device, encoder);
            for (from, to) in old.textures.iter().zip(&textures) {
                for level in 0..levels {
                    let size = size >> level;
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: from,
                            mip_level: level,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: to,
                            mip_level: level,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: size,
                            height: size,
                            depth_or_array_layers: old.layers,
                        },
                    );
                }
            }
        }
        let views = textures.each_ref().map(|texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/impostor-array-view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        });
        self.arrays = Some(Arrays {
            textures,
            views,
            layers: capacity,
        });
        self.arrays_generation += 1;
    }

    /// Uploads `records` (zeroed for draw groups without an impostor)
    /// when they changed, and builds the draws' group; returns whether
    /// there is one, not before any impostor is resident.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &ImpostorPipelines,
        records: &[ImpostorRecord],
    ) -> bool {
        let Some(arrays) = self.arrays.as_ref() else {
            return false;
        };
        if records != self.uploaded.as_slice() {
            let bytes = size_of_val(records) as u64;
            if bytes > self.records.size() {
                self.records = record_buffer(device, records.len().next_power_of_two());
                self.records_generation += 1;
            }
            if !records.is_empty() {
                queue.write_buffer(&self.records, 0, bytemuck::cast_slice(records));
            }
            self.uploaded.clear();
            self.uploaded.extend_from_slice(records);
        }
        let key = (self.records_generation, self.arrays_generation);
        if self.group.stale(&key) {
            let group = crate::pipelines::group(
                device,
                "engine/impostor-group",
                &IMPOSTOR_DRAW,
                &pipelines.draw_layout,
                &[
                    ("impostor_records", self.records.as_entire_binding()),
                    ("impostor_albedo", crate::pipelines::view(&arrays.views[0])),
                    (
                        "impostor_emissive",
                        crate::pipelines::view(&arrays.views[1]),
                    ),
                    (
                        "impostor_geometry",
                        crate::pipelines::view(&arrays.views[2]),
                    ),
                    ("impostor_surface", crate::pipelines::view(&arrays.views[3])),
                    (
                        "impostor_sampler",
                        crate::pipelines::sampler(&pipelines.sampler),
                    ),
                ],
            );
            self.group.set(key, group);
        }
        true
    }

    /// The draws' group, after [`Self::prepare`] built it.
    pub(crate) fn group(&self) -> Option<&wgpu::BindGroup> {
        self.group.get()
    }
}

/// A card's visible slot and draw group, per instance.
const CARD_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![2 => Uint32, 3 => Uint32];

/// An impostor draw pipeline: `layout` binds the camera, the impostors'
/// group, the poses (with previous poses on temporal frames) and, when
/// `lit`, the local lights; it shades a scene of `shading`'s features.
/// `canvas` is the surface's format.
pub(crate) fn draw_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    (targets, lit): (Targets, bool),
    shading: SceneShading,
    canvas: wgpu::TextureFormat,
    depth: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let vertex = crate::entries::impostor_vertex(targets);
    let fragment = crate::entries::impostor_fragment(targets, lit);
    let color_targets = targets.color_targets(canvas);
    let constants = shading.constants();
    let mut buffers = vec![Some(wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &CARD_ATTRIBUTES,
    })];
    // Debug views read the draw's record.
    if targets == Targets::Debug {
        buffers = crate::debug_view::with_record(&buffers);
    }
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label: "engine/impostors",
            layout,
            module: shader,
            vertex_constants: &[],
            vertex,
            buffers: &buffers,
            // Mirrored poses turn cards around.
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(crate::pipelines::Fragment {
                entry: fragment,
                constants: &constants,
                targets: &color_targets,
            }),
        },
    )
}
