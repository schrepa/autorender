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
use crate::entries::MeshEntries;
use crate::gpu_data::Variant;
use crate::gpu_layouts::constants::IMPOSTOR_FOLIAGE as FOLIAGE;
use crate::gpu_layouts::constants::IMPOSTOR_FRAMES_SHIFT as FRAMES_SHIFT;
pub use crate::gpu_layouts::constants::IMPOSTOR_HEMISPHERE as HEMISPHERE;
use crate::gpu_layouts::constants::IMPOSTOR_KINDS as KINDS;
use crate::gpu_layouts::constants::IMPOSTOR_MASKED as MASKED;
use crate::gpu_layouts::constants::VARIANT_FOLIAGE;
pub use crate::gpu_layouts::impostors::ImpostorRecord;
use crate::graph::Targets;
use crate::mesh_arena::MeshDraw;
use crate::mipmaps::MipGenerator;
use crate::parts::{Prelude, PreludeStage};
use crate::renderer::ShadingSet;
pub use content::Contents;

const IMPOSTOR_RECORD: &str = "impostor";

pub const IMPOSTOR_MIN_TRIANGLES: usize = 64;

const RGBA8_SIZE: u32 = 512;

const LOAD_BUDGET: usize = 16 << 20;

const PLACED_FRAMES: u32 = 12;

const FIRST_BYTES: u64 = 16 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasShape {
    pub frames: u32,
    pub texels: u32,
    pub hemisphere: bool,
}

impl AtlasShape {
    pub const SPHERE: Self = Self {
        frames: 8,
        texels: 16,
        hemisphere: false,
    };

    pub fn placed(pixels: f32) -> Self {
        let texels = (pixels.ceil() as u32).clamp(2, 1 << 15);
        Self {
            frames: PLACED_FRAMES,
            texels: texels.next_power_of_two(),
            hemisphere: true,
        }
    }

    pub fn size(self) -> u32 {
        self.frames * self.texels
    }

    pub fn levels(self) -> u32 {
        self.texels.trailing_zeros()
    }

    fn flags(self) -> u32 {
        self.texels | if self.hemisphere { HEMISPHERE } else { 0 } | (self.frames << FRAMES_SHIFT)
    }

    fn held(self, format: AtlasFormat) -> Self {
        match format {
            AtlasFormat::Rgba8 if self.size() > RGBA8_SIZE => Self {
                texels: self.texels / 2,
                ..self
            },
            AtlasFormat::Bc7 | AtlasFormat::Astc | AtlasFormat::Rgba8 => self,
        }
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtlasFormat {
    Bc7,
    Astc,
    Rgba8,
}

const FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8Unorm,
];
const BC7_FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    wgpu::TextureFormat::Bc7RgbaUnorm,
    wgpu::TextureFormat::Bc7RgbaUnorm,
];
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

    fn tag(self) -> u32 {
        match self {
            Self::Rgba8 => 0,
            Self::Bc7 => 1,
            Self::Astc => 2,
        }
    }

    fn family(self) -> Option<engine_textures::Family> {
        match self {
            Self::Bc7 => Some(engine_textures::Family::Bc7),
            Self::Astc => Some(engine_textures::Family::Astc),
            Self::Rgba8 => None,
        }
    }

    fn rows(self, size: u32) -> (u32, u32) {
        match self {
            Self::Bc7 | Self::Astc => (size.div_ceil(4) * 16, size.div_ceil(4)),
            Self::Rgba8 => (size * 4, size),
        }
    }

    fn level_bytes(self, size: u32) -> usize {
        let (row, rows) = self.rows(size);
        row as usize * rows as usize
    }

    fn textures(self) -> [wgpu::TextureFormat; 4] {
        match self {
            Self::Bc7 => BC7_FORMATS,
            Self::Astc => ASTC_FORMATS,
            Self::Rgba8 => FORMATS,
        }
    }
}

impl ImpostorRecord {
    pub fn new(
        min: Vec3,
        max: Vec3,
        layer: u32,
        (emissive, masked, foliage): ([f32; 3], bool, bool),
        shape: AtlasShape,
    ) -> Self {
        let center = (min + max) * 0.5;
        let half = (max - min) * 0.5;
        let radius = engine_core::math::max(half.length() * 1.0001, 1.0e-6);
        Self {
            sphere: center.extend(radius).to_array(),
            extent: half.to_array(),
            layer,
            emissive,
            flags: shape.flags()
                | if masked { MASKED } else { 0 }
                | if foliage { FOLIAGE } else { 0 },
        }
    }

    pub fn error(&self) -> f32 {
        let texels = self.flags & (HEMISPHERE - 1);
        if texels == 0 {
            return 0.0;
        }
        2.0 * self.sphere[3] / texels as f32
    }

    pub fn framed(&self) -> bool {
        self.flags & (HEMISPHERE - 1) != 0 && self.sphere[3] > 0.0
    }
}

#[derive(Debug, Default)]
pub struct Layers {
    free: Vec<u32>,
    used: u32,
}

impl Layers {
    pub fn allocate(&mut self, limit: u32) -> Option<u32> {
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

    pub fn release(&mut self, layer: u32) {
        self.free.push(layer);
    }

    pub fn extent(&self) -> u32 {
        self.used
    }

    pub fn live(&self) -> u32 {
        self.used - self.free.len() as u32
    }
}

pub fn grown_layers(needed: u32, bytes: u64, limit: u32) -> u32 {
    let first = (FIRST_BYTES / bytes.max(1)).clamp(1, 16) as u32;
    needed.max(first).next_power_of_two().min(limit).max(needed)
}

fn shades_as_foliage(variant: Variant) -> bool {
    variant.alpha_mask() || variant.bits() & VARIANT_FOLIAGE != 0
}

fn most_layers(device: &wgpu::Device) -> u32 {
    device.limits().max_texture_array_layers / KINDS
}

fn layer_limit(device: &wgpu::Device, budget: Option<u32>) -> u32 {
    most_layers(device).min(budget.unwrap_or(u32::MAX))
}

const MIPS_LABEL: &str = "engine/texture-mips";

pub trait MipEncoder {
    fn mips(&mut self, device: &wgpu::Device) -> &mut wgpu::CommandEncoder;
}

impl MipEncoder for Option<wgpu::CommandEncoder> {
    fn mips(&mut self, device: &wgpu::Device) -> &mut wgpu::CommandEncoder {
        self.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(MIPS_LABEL),
            })
        })
    }
}

impl MipEncoder for Prelude {
    fn mips(&mut self, device: &wgpu::Device) -> &mut wgpu::CommandEncoder {
        self.encoder(device, PreludeStage::Uploads, "pre_mips", MIPS_LABEL)
    }
}

pub type ImpostorKey = (MeshHandle, Option<MaterialHandle>);

struct Arrays {
    texture: wgpu::Texture,
    views: [wgpu::TextureView; 4],
    whole: wgpu::TextureView,
    layers: u32,
}

struct Baked {
    record: ImpostorRecord,
    material: Option<Material>,
}

#[derive(Clone, Copy)]
struct Supply {
    record: RecordKey,
    material: Option<Material>,
}

pub enum Supplied {
    Resident(ImpostorRecord),
    Coming,
    Refused,
}

impl Supplied {
    pub fn record(&self) -> ImpostorRecord {
        match self {
            Self::Resident(record) => *record,
            Self::Coming | Self::Refused => ImpostorRecord::default(),
        }
    }
}

pub struct BakeSource<'a> {
    pub draw: MeshDraw<'a>,
    pub extended: bool,
    pub group: &'a wgpu::BindGroup,
    pub variant: Variant,
    pub bounds: (Vec3, Vec3),
    pub material: Option<Material>,
    pub parts: Vec<BakePart<'a>>,
}

pub struct BakePart<'a> {
    pub draw: MeshDraw<'a>,
    pub extended: bool,
    pub group: &'a wgpu::BindGroup,
    pub variant: Variant,
}

pub struct BakeGpu<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a crate::gpu_calls::Queue,
    pub encoder: &'a mut dyn MipEncoder,
    pub mips: &'a MipGenerator,
    pub pipelines: &'a mut ImpostorPipelines,
    pub entries: &'a MeshEntries,
}

pub struct ImpostorPipelines {
    bake_layout: crate::pipelines::GroupLayout,
    bake_pipeline_layout: wgpu::PipelineLayout,
    bake_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
    shade_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
    draw_layout: crate::pipelines::GroupLayout,
    draw_pipelines: HashMap<(Targets, ShadingSet), wgpu::RenderPipeline>,
    sampler: wgpu::Sampler,
    no_shade: Option<wgpu::TextureView>,
}

impl ImpostorPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let bake_layout = crate::pipelines::layout(device, &IMPOSTOR_BAKE);
        let bake_pipeline_layout =
            crate::pipelines::layout_of(device, &crate::pipeline_layouts::IMPOSTOR_BAKE_PIPELINES);
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

    pub fn pipeline(&self, targets: Targets, shading: ShadingSet) -> Option<&wgpu::RenderPipeline> {
        self.draw_pipelines.get(&(targets, shading))
    }

    pub fn insert_pipeline(
        &mut self,
        targets: Targets,
        shading: ShadingSet,
        pipeline: wgpu::RenderPipeline,
    ) {
        self.draw_pipelines.insert((targets, shading), pipeline);
    }

    pub fn forget_pipelines(&mut self, targets: Targets) {
        self.draw_pipelines.retain(|(kind, _), _| *kind != targets);
    }
}

pub struct ImpostorAtlas {
    shape: AtlasShape,
    format: AtlasFormat,
    arrays: Option<Arrays>,
    targets: Option<bake::BakeTargets>,
    layers: Layers,
    baked: HashMap<ImpostorKey, Baked>,
    supplies: HashMap<ImpostorKey, Supply>,
    supplied: HashSet<ImpostorKey>,
    wants: Vec<RecordKey>,
    answers: HashMap<RecordKey, Option<Vec<u8>>>,
    readbacks: Vec<records::Readback>,
    loaded: usize,
    deferred: bool,
    cooking: bool,
    records: wgpu::Buffer,
    uploaded: Vec<ImpostorRecord>,
    group: Cached<(u64, u64), wgpu::BindGroup>,
    records_generation: u64,
    arrays_generation: u64,
    generation: u64,
}

fn record_buffer(device: &wgpu::Device, records: usize) -> wgpu::Buffer {
    crate::memory::buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("engine/impostor-records"),
            size: (records.max(1) * size_of::<ImpostorRecord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )
}

impl ImpostorAtlas {
    pub fn new(device: &wgpu::Device, shape: AtlasShape) -> Self {
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

    pub fn set_cooking(&mut self, cooking: bool) {
        self.cooking = cooking;
    }

    pub fn fit(&mut self, device: &wgpu::Device, shape: AtlasShape) {
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

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn live(&self) -> u32 {
        self.layers.live()
    }

    pub fn capacity_bytes(&self) -> u64 {
        let layers = self.arrays.as_ref().map_or(0, |arrays| arrays.layers);
        u64::from(layers) * self.layer_bytes() + self.records.size()
    }

    fn layer_bytes(&self) -> u64 {
        self.format.level_bytes(self.shape.size()) as u64 * 4 * 4 / 3
    }

    pub fn contains(&self, key: ImpostorKey) -> bool {
        self.baked.contains_key(&key)
    }

    pub fn record(
        &self,
        key: ImpostorKey,
        material: Option<&Material>,
    ) -> Option<ImpostorRecord> {
        self.baked
            .get(&key)
            .filter(|baked| baked.material.as_ref() == material)
            .map(|baked| baked.record)
    }

    pub fn take_wants(&mut self, out: &mut Vec<RecordKey>) {
        out.append(&mut self.wants);
    }

    pub fn asks(&self, record: RecordKey) -> bool {
        self.supplies.values().any(|supply| supply.record == record)
    }

    pub fn answer(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        self.answers.insert(record, bytes);
        self.generation += 1;
    }

    pub fn left(&self) -> u64 {
        (self.supplies.len() + self.readbacks.len()) as u64
    }

    fn record_key(&self, content: (u128, u128)) -> RecordKey {
        RecordKey::new(
            &crate::SOURCE,
            IMPOSTOR_RECORD,
            &records::identity(self.shape, self.format, content),
        )
    }

    pub fn retain(
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
            let record = serde_json::json!({
                "event": "impostor_supplies_dropped",
                "dropped": kept - self.supplies.len(),
                "asked": asked,
            });
            eprintln!("{record}");
        }
        let supplies = &self.supplies;
        self.answers
            .retain(|record, _| supplies.values().any(|supply| supply.record == *record));
    }

    pub fn enforce_budget(&mut self, budget: Option<u32>) {
        if !self.cooking && budget.is_some_and(|budget| self.layers.live() > budget) {
            self.baked.clear();
            self.layers = Layers::default();
            self.generation += 1;
        }
    }

    pub fn supply(
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
        if self.loaded >= LOAD_BUDGET && self.answers.get(&record).is_some_and(Option::is_some) {
            self.deferred = true;
            return Supplied::Coming;
        }
        let Some(answer) = self.answers.remove(&record) else {
            return Supplied::Coming;
        };
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
        self.bake(gpu, source, content);
        Supplied::Coming
    }

    fn supplied(&self, key: ImpostorKey, source: &BakeSource<'_>) -> Supplied {
        self.record(key, source.material.as_ref())
            .map_or(Supplied::Refused, Supplied::Resident)
    }

    fn layer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut dyn MipEncoder,
        budget: Option<u32>,
    ) -> Option<u32> {
        if self.layers.live() >= layer_limit(device, budget) {
            return None;
        }
        let layer = self.layers.allocate(most_layers(device))?;
        self.ensure_layers(device, encoder, self.layers.extent());
        Some(layer)
    }

    fn resident(&mut self, key: ImpostorKey, source: &BakeSource<'_>, layer: u32) {
        let emissive = source
            .material
            .as_ref()
            .map_or([0.0; 3], Material::emissive);
        let variants =
            || std::iter::once(source.variant).chain(source.parts.iter().map(|part| part.variant));
        let masked = variants().any(Variant::alpha_mask);
        let foliage = variants().any(shades_as_foliage);
        let record = ImpostorRecord::new(
            source.bounds.0,
            source.bounds.1,
            layer,
            (emissive, masked, foliage),
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

    pub fn release_targets(&mut self) {
        self.targets = None;
    }

    fn ensure_layers(&mut self, device: &wgpu::Device, encoder: &mut dyn MipEncoder, layers: u32) {
        if self
            .arrays
            .as_ref()
            .is_some_and(|arrays| arrays.layers >= layers)
        {
            return;
        }
        let capacity = grown_layers(layers, self.layer_bytes(), most_layers(device));
        let (size, levels) = (self.shape.size(), self.shape.levels());
        let formats = self.format.textures();
        let texture = crate::memory::texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("engine/impostor-array"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: KINDS * capacity,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: formats[0].remove_srgb_suffix(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[formats[0]],
            },
        );
        if let Some(old) = &self.arrays {
            let encoder = encoder.mips(device);
            for kind in 0..KINDS {
                for level in 0..levels {
                    let size = size >> level;
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &old.texture,
                            mip_level: level,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: kind * old.layers,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: level,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: kind * capacity,
                            },
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
        let views = std::array::from_fn(|kind| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/impostor-array-view"),
                format: Some(formats[kind]),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_array_layer: kind as u32 * capacity,
                array_layer_count: Some(capacity),
                ..Default::default()
            })
        });
        let whole = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/impostor-array-whole"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.arrays = Some(Arrays {
            texture,
            views,
            whole,
            layers: capacity,
        });
        self.arrays_generation += 1;
    }

    pub fn frames(&self) -> Option<(&wgpu::TextureView, u64)> {
        self.arrays
            .as_ref()
            .map(|arrays| (&arrays.whole, self.arrays_generation))
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
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

    pub fn group(&self) -> Option<&wgpu::BindGroup> {
        self.group.get()
    }
}

const CARD_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![2 => Uint32, 3 => Uint32];

pub fn draw_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    (targets, entries): (Targets, &MeshEntries),
    shading: &[(&'static str, f64)],
    canvas: wgpu::TextureFormat,
    depth: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let vertex = (entries.card_vertex)(targets);
    let fragment = (entries.card_fragment)(targets);
    let color_targets = targets.color_targets(canvas);
    let mut buffers = vec![Some(wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &CARD_ATTRIBUTES,
    })];
    if targets == Targets::Debug {
        buffers = crate::debug_view::with_record(&buffers);
    }
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label: "engine/impostors",
            layout,
            vertex_constants: &[],
            vertex,
            buffers: &buffers,
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
                constants: shading,
                targets: &color_targets,
            }),
        },
    )
}
