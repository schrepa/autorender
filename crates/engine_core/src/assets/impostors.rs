use std::collections::BTreeSet;
use std::sync::{Arc, Weak};

use bevy_ecs::change_detection::Mut;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Query, Res, SystemState};
use bevy_ecs::world::World;
use engine_assets::{
    AlphaMode, AssetLibrary, Cull, Filter, LightingModel, Material, MaterialHandle, MeshHandle,
    TextureRef, Wrap,
};
use glam::Vec3;

use super::AssetResidency;
use crate::collections::{HashMap, HashSet};
use crate::{
    Census, CookMode, Detail, Error, ErrorCode, Feed, FeedSettings, FrameBudget, GpuBits,
    IDLE_JOBS, Interest, Job, JobCx, JobExchange, JobId, JobKind, JobRef, Jobs, RecordKey,
    Records, Request, Requests, Slot, Source, VIEW_JOBS, WorkClock, content_hash, job_priority,
};

pub const IMPOSTOR_MIN_TRIANGLES: usize = 64;

const BAKES_IN_FLIGHT: usize = 4;

const BAKE_DELAY_MS: f64 = 1000.0;

const BLOCK_ROWS: u32 = 4;

#[crate::gpu_bits(crate = crate)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImpostorFlags {
    #[bits(16)]
    pub texels: u32,
    pub hemisphere: bool,
    pub masked: bool,
    #[bits(5, at = 19)]
    pub frames: u32,
}

pub mod impostor_atlas {
    crate::gpu_consts! {
        pub const IMPOSTOR_KINDS: u32 = 4;
    }
}

pub use impostor_atlas::IMPOSTOR_KINDS;

crate::gpu_struct! {
    pub struct ImpostorRecord {
        pub sphere: [f32; 4],
        pub extent: [f32; 3],
        pub layer: u32,
        pub emissive: [f32; 3],
        pub flags: u32,
    }
}

impl ImpostorRecord {
    pub fn new(
        min: Vec3,
        max: Vec3,
        layer: u32,
        (emissive, masked): ([f32; 3], bool),
        shape: AtlasShape,
    ) -> Self {
        let center = (min + max) * 0.5;
        let half = (max - min) * 0.5;
        let radius = crate::math::max(half.length() * 1.0001, 1.0e-6);
        Self {
            sphere: center.extend(radius).to_array(),
            extent: half.to_array(),
            layer,
            emissive,
            flags: ImpostorFlags {
                masked,
                ..shape.flags()
            }
            .pack(),
        }
    }

    fn texels(&self) -> u32 {
        ImpostorFlags::unpack(self.flags).map_or(0, |flags| flags.texels)
    }

    pub fn error(&self) -> f32 {
        let texels = self.texels();
        if texels == 0 {
            return 0.0;
        }
        2.0 * self.sphere[3] / texels as f32
    }

    pub fn framed(&self) -> bool {
        self.texels() != 0 && self.sphere[3] > 0.0
    }
}

const RGBA8_SIZE: u32 = 512;

const ATLAS_FRAMES: u32 = 12;

const FIRST_BYTES: u64 = 16 << 20;

const RECORD_MAGIC: &[u8; 4] = b"EIMP";

#[doc(hidden)]
pub const IMPOSTOR_RECORD_HEADER: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasShape {
    pub frames: u32,
    pub texels: u32,
    pub hemisphere: bool,
}

impl AtlasShape {
    fn of(pixels: f32) -> Self {
        let texels = (pixels.ceil() as u32).clamp(2, 1 << 15);
        Self {
            frames: ATLAS_FRAMES,
            texels: texels.next_power_of_two(),
            hemisphere: false,
        }
    }

    fn facing(self, hemisphere: bool) -> Self {
        Self { hemisphere, ..self }
    }

    pub fn size(self) -> u32 {
        self.frames * self.texels
    }

    pub fn levels(self) -> u32 {
        self.texels.trailing_zeros()
    }

    fn flags(self) -> ImpostorFlags {
        ImpostorFlags {
            texels: self.texels,
            hemisphere: self.hemisphere,
            masked: false,
            frames: self.frames,
        }
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

    fn held_on(self, device: &AtlasDevice) -> Self {
        let mut shape = self.held(device.format);
        while (shape.size() > device.max_texture_dimension
            || readback_bytes(shape, device.row_alignment) > device.max_buffer_size)
            && shape.texels > 2
        {
            shape.texels /= 2;
        }
        shape
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasFormat {
    Bc7,
    Astc,
    Rgba8,
}

impl AtlasFormat {
    #[doc(hidden)]
    pub fn tag(self) -> u32 {
        match self {
            Self::Rgba8 => 0,
            Self::Bc7 => 1,
            Self::Astc => 2,
        }
    }

    #[doc(hidden)]
    pub fn rows(self, size: u32) -> (u32, u32) {
        match self {
            Self::Bc7 | Self::Astc => (size.div_ceil(4) * 16, size.div_ceil(4)),
            Self::Rgba8 => (size * 4, size),
        }
    }

    #[doc(hidden)]
    pub fn level_bytes(self, size: u32) -> usize {
        let (row, rows) = self.rows(size);
        row as usize * rows as usize
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasDevice {
    pub format: AtlasFormat,
    pub max_texture_dimension: u32,
    pub max_buffer_size: u64,
    pub max_layers: u32,
    pub row_alignment: u32,
}

impl AtlasDevice {
    fn most_layers(&self) -> u32 {
        self.max_layers / IMPOSTOR_KINDS
    }

    fn layer_limit(&self, budget: Option<u32>) -> u32 {
        self.most_layers().min(budget.unwrap_or(u32::MAX))
    }
}

#[doc(hidden)]
pub fn impostor_level_rows(shape: AtlasShape, level: u32, alignment: u32) -> (u32, u32, u32) {
    let (row, rows) = AtlasFormat::Rgba8.rows(shape.size() >> level);
    (rows, row.next_multiple_of(alignment), row)
}

fn readback_bytes(shape: AtlasShape, alignment: u32) -> u64 {
    4 * (0..shape.levels())
        .map(|level| {
            let (rows, stride, _) = impostor_level_rows(shape, level, alignment);
            u64::from(stride) * u64::from(rows)
        })
        .sum::<u64>()
}

fn impostor_record_header(shape: AtlasShape, format: AtlasFormat) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(IMPOSTOR_RECORD_HEADER);
    bytes.extend_from_slice(RECORD_MAGIC);
    for word in [
        format.tag(),
        shape.frames,
        shape.texels,
        u32::from(shape.hemisphere),
        shape.levels(),
    ] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn record_bytes(shape: AtlasShape, format: AtlasFormat) -> usize {
    IMPOSTOR_RECORD_HEADER
        + 4 * (0..shape.levels())
            .map(|level| format.level_bytes(shape.size() >> level))
            .sum::<usize>()
}

#[doc(hidden)]
pub type ImpostorIdentity = [u8; 48];

fn identity(shape: AtlasShape, format: AtlasFormat, content: (u128, u128)) -> ImpostorIdentity {
    let mut bytes = [0_u8; 48];
    for (at, word) in [
        format.tag(),
        shape.frames,
        shape.texels,
        u32::from(shape.hemisphere),
    ]
    .into_iter()
    .enumerate()
    {
        bytes[at * 4..at * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes[16..32].copy_from_slice(&content.0.to_le_bytes());
    bytes[32..48].copy_from_slice(&content.1.to_le_bytes());
    bytes
}

static RENDER: Source = Source::generated("engine_render", false, crate::sources::RENDER, &[]);

#[doc(hidden)]
pub static IMPOSTOR: JobKind = JobKind::new("engine::impostor", &RENDER, made_on_the_gpu);

fn made_on_the_gpu(_: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs.engine::impostor").got("engine::impostor"))
}

#[doc(hidden)]
pub static IMPOSTOR_ASSEMBLY: JobKind =
    JobKind::new("engine::impostor_assembly", &RENDER, start_assembly);

const ASSEMBLY_HEADER: usize = IMPOSTOR_RECORD_HEADER + 4;

#[doc(hidden)]
pub fn impostor_assembly_input(
    shape: AtlasShape,
    format: AtlasFormat,
    alignment: u32,
    read: &[u8],
) -> Vec<u8> {
    let mut input = Vec::with_capacity(ASSEMBLY_HEADER + read.len());
    input.extend_from_slice(&impostor_record_header(shape, format));
    input.extend_from_slice(&alignment.to_le_bytes());
    input.extend_from_slice(read);
    input
}

fn input_word(input: &[u8], at: usize) -> Option<u32> {
    let word: [u8; 4] = input.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(word))
}

struct Assembly {
    input: Vec<u8>,
    shape: AtlasShape,
    family: Option<engine_textures::Family>,
    alignment: u32,
    out: Vec<u8>,
    region: u32,
    at: usize,
    row: u32,
}

impl Assembly {
    fn parse(input: &[u8]) -> Option<Self> {
        let format = match input_word(input, 4)? {
            0 => AtlasFormat::Rgba8,
            1 => AtlasFormat::Bc7,
            2 => AtlasFormat::Astc,
            _ => return None,
        };
        let shape = AtlasShape {
            frames: input_word(input, 8)?,
            texels: input_word(input, 12)?,
            hemisphere: input_word(input, 16)? != 0,
        };
        let alignment = input_word(input, IMPOSTOR_RECORD_HEADER)?;
        if input.len() as u64 != ASSEMBLY_HEADER as u64 + readback_bytes(shape, alignment) {
            return None;
        }
        let mut out = Vec::with_capacity(record_bytes(shape, format));
        out.extend_from_slice(&input[..IMPOSTOR_RECORD_HEADER]);
        Some(Self {
            input: input.to_vec(),
            shape,
            family: match format {
                AtlasFormat::Bc7 => Some(engine_textures::Family::Bc7),
                AtlasFormat::Astc => Some(engine_textures::Family::Astc),
                AtlasFormat::Rgba8 => None,
            },
            alignment,
            out,
            region: 0,
            at: ASSEMBLY_HEADER,
            row: 0,
        })
    }

    fn band(&mut self, level: u32) {
        let (rows, stride, row) = impostor_level_rows(self.shape, level, self.alignment);
        let band = match self.family {
            Some(_) => (rows - self.row).min(BLOCK_ROWS),
            None => rows,
        };
        let mut texels = Vec::with_capacity(row as usize * band as usize);
        for at in self.row..self.row + band {
            let start = self.at + at as usize * stride as usize;
            texels.extend_from_slice(&self.input[start..start + row as usize]);
        }
        match self.family {
            Some(family) => {
                let size = self.shape.size() >> level;
                let blocks = engine_textures::compress(size, band, &texels, family);
                self.out.extend_from_slice(&blocks);
            }
            None => self.out.extend_from_slice(&texels),
        }
        self.row += band;
        if self.row == rows {
            self.at += stride as usize * rows as usize;
            self.row = 0;
            self.region += 1;
        }
    }
}

impl Job for Assembly {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let levels = self.shape.levels();
        if self.region < IMPOSTOR_KINDS * levels {
            self.band(self.region % levels);
        }
        Ok((self.region >= IMPOSTOR_KINDS * levels).then(|| std::mem::take(&mut self.out)))
    }
}

fn start_assembly(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let job = Assembly::parse(input).ok_or_else(|| {
        Error::new(ErrorCode::InvalidRecord, "$.jobs.engine::impostor_assembly").got(input.len())
    })?;
    Ok(Box::new(job))
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ImpostorPolicy {
    pub enabled: bool,
    pub pixels: f32,
    pub max: Option<u32>,
}

impl Default for ImpostorPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            pixels: 128.0,
            max: None,
        }
    }
}

impl Slot for ImpostorPolicy {
    const NAME: &'static str = "impostor_policy";
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImpostorAsk {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub parts: Vec<(MeshHandle, MaterialHandle)>,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct ImpostorAsks {
    pub generation: u64,
    pub hemisphere: bool,
    pub asks: Vec<Option<ImpostorAsk>>,
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct ImpostorRecords {
    pub generation: u64,
    pub records: Vec<ImpostorRecord>,
    pub asked: u32,
    pub refused: u32,
    pub live: u32,
}

fn material_content(library: &AssetLibrary, material: Option<&Material>) -> u128 {
    let Some(material) = material else {
        return 0;
    };
    let mut bytes = Vec::with_capacity(256);
    floats(&mut bytes, &material.color());
    for (name, map) in material.maps() {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        let texture = library
            .texture(map.texture)
            .map_or(0, |texture| texture.identity(content_hash));
        bytes.extend_from_slice(&texture.to_le_bytes());
        texture_ref(&mut bytes, map);
    }
    for (slot, module) in material.slots() {
        bytes.extend_from_slice(slot.name().as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(module.as_bytes());
        bytes.push(0);
    }
    for (name, value) in material.values() {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        floats(&mut bytes, &[*value]);
    }
    let surface = material.surface();
    floats(
        &mut bytes,
        &[
            surface.roughness,
            surface.metalness,
            surface.emission,
            surface.ripples,
            material.normal_scale(),
            material.occlusion_strength(),
        ],
    );
    floats(&mut bytes, &material.emissive());
    floats(&mut bytes, &[material.transmission()]);
    match material.alpha_mode() {
        AlphaMode::Opaque => bytes.push(0),
        AlphaMode::Mask { cutoff } => {
            bytes.push(1);
            floats(&mut bytes, &[cutoff]);
        }
        AlphaMode::Blend => bytes.push(2),
        AlphaMode::Additive => bytes.push(3),
    }
    bytes.push(match material.cull() {
        Cull::Back => 0,
        Cull::Front => 1,
        Cull::None => 2,
    });
    match material.lighting() {
        LightingModel::Lit => bytes.push(0),
        LightingModel::Unlit => bytes.push(1),
        LightingModel::Named(name) => {
            bytes.push(2);
            bytes.extend_from_slice(name.as_bytes());
        }
    }
    content_hash(&bytes)
}

fn floats(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn words(bytes: &mut Vec<u8>, values: &[u32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn texture_ref(bytes: &mut Vec<u8>, map: &TextureRef) {
    let wrap = |wrap: Wrap| match wrap {
        Wrap::Repeat => 0,
        Wrap::MirroredRepeat => 1,
        Wrap::ClampToEdge => 2,
    };
    let filter = |filter: Filter| match filter {
        Filter::Nearest => 0,
        Filter::Linear => 1,
    };
    let sampler = map.sampler;
    bytes.extend_from_slice(&[
        wrap(sampler.wrap_u),
        wrap(sampler.wrap_v),
        filter(sampler.mag_filter),
        filter(sampler.min_filter),
        sampler.mipmap_filter.map_or(2, filter),
    ]);
    let transform = map.transform;
    floats(bytes, &transform.offset);
    floats(bytes, &[transform.rotation]);
    floats(bytes, &transform.scale);
    words(bytes, &[map.tex_coord]);
}

fn combined(first: u128, second: u128) -> u128 {
    let mut bytes = [0_u8; 32];
    bytes[..16].copy_from_slice(&first.to_le_bytes());
    bytes[16..].copy_from_slice(&second.to_le_bytes());
    content_hash(&bytes)
}

fn masked(material: Option<&Material>) -> bool {
    material.is_some_and(|material| matches!(material.alpha_mode(), AlphaMode::Mask { .. }))
}

#[derive(Debug, Default)]
struct Layers {
    free: Vec<u32>,
    used: u32,
}

impl Layers {
    fn allocate(&mut self, limit: u32) -> Option<u32> {
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

    fn release(&mut self, layer: u32) {
        self.free.push(layer);
    }

    fn extent(&self) -> u32 {
        self.used
    }

    fn live(&self) -> u32 {
        self.used - self.free.len() as u32
    }
}

fn grown_layers(needed: u32, bytes: u64, limit: u32) -> u32 {
    let first = (FIRST_BYTES / bytes.max(1)).clamp(1, 16) as u32;
    needed.max(first).next_power_of_two().min(limit).max(needed)
}

type ImpostorKey = (MeshHandle, Option<MaterialHandle>, bool);

struct Baked {
    record: ImpostorRecord,
    material: Option<Material>,
}

struct Supply {
    material: Option<Material>,
    interest: Interest<ImpostorIdentity>,
    fetch: Option<Interest<RecordKey>>,
    since: Option<f64>,
}

enum Supplied {
    Resident(ImpostorRecord),
    Coming,
    Refused,
}

impl Supplied {
    fn record(&self) -> ImpostorRecord {
        match self {
            Self::Resident(record) => *record,
            Self::Coming | Self::Refused => ImpostorRecord::default(),
        }
    }
}

struct Bake {
    material: Option<Material>,
    bounds: (Vec3, Vec3),
    content: (u128, u128),
    masked: bool,
    parts: Vec<(MeshHandle, MaterialHandle)>,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum AtlasWork {
    Reset {
        shape: AtlasShape,
    },
    Release,
    Grow {
        layers: u32,
    },
    Load {
        layer: u32,
        bytes: Weak<[u8]>,
    },
    Bake {
        identity: ImpostorIdentity,
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
        parts: Vec<(MeshHandle, MaterialHandle)>,
        bounds: (Vec3, Vec3),
        hemisphere: bool,
    },
    DropTargets,
}

struct Book {
    shape: AtlasShape,
    layers: Layers,
    capacity: u32,
    targets: bool,
    drawn: bool,
    baked: HashMap<ImpostorKey, Baked>,
    supplies: HashMap<ImpostorKey, Supply>,
    supplied: HashSet<ImpostorKey>,
    answers: Requests<ImpostorIdentity, Arc<[u8]>>,
    baking: BTreeSet<ImpostorIdentity>,
}

struct Supplying<'a, 'w, 's> {
    device: AtlasDevice,
    jobs: &'a mut Jobs<'w, 's>,
    log: &'a mut Feed<AtlasWork>,
    budget: &'a mut FrameBudget,
    cooking: bool,
    now: Option<f64>,
}

impl Supplying<'_, '_, '_> {
    fn lasted(&self, since: Option<f64>) -> bool {
        match (since, self.now) {
            (Some(since), Some(now)) => now - since >= BAKE_DELAY_MS,
            _ => true,
        }
    }
}

impl Book {
    fn new(shape: AtlasShape, device: &AtlasDevice) -> Self {
        Self {
            shape: shape.held_on(device),
            layers: Layers::default(),
            capacity: 0,
            targets: false,
            drawn: false,
            baked: HashMap::default(),
            supplies: HashMap::default(),
            supplied: HashSet::default(),
            answers: Requests::default(),
            baking: BTreeSet::new(),
        }
    }

    fn push(&mut self, cx: &mut Supplying<'_, '_, '_>, work: AtlasWork) {
        cx.log.push(work);
    }

    fn fit(&mut self, cx: &mut Supplying<'_, '_, '_>, shape: AtlasShape) {
        let shape = shape.held_on(&cx.device);
        if shape == self.shape {
            return;
        }
        self.restart(cx, shape);
    }

    fn restart(&mut self, cx: &mut Supplying<'_, '_, '_>, shape: AtlasShape) {
        self.shape = shape;
        self.capacity = 0;
        self.targets = false;
        self.drawn = false;
        self.layers = Layers::default();
        self.baked.clear();
        self.supplies.clear();
        self.supplied.clear();
        self.baking.clear();
        self.push(cx, AtlasWork::Reset { shape });
    }

    fn release(&mut self, cx: &mut Supplying<'_, '_, '_>) {
        if self.capacity == 0
            && !self.targets
            && self.baked.is_empty()
            && self.supplies.is_empty()
            && self.baking.is_empty()
            && !self.drawn
        {
            return;
        }
        self.capacity = 0;
        self.targets = false;
        self.drawn = false;
        self.layers = Layers::default();
        self.baked.clear();
        self.supplies.clear();
        self.supplied.clear();
        self.baking.clear();
        self.push(cx, AtlasWork::Release);
    }

    fn retain(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        keep: impl Fn(&ImpostorKey) -> bool,
        material: impl Fn(MaterialHandle) -> Option<Material>,
    ) {
        let layers = &mut self.layers;
        self.baked.retain(|key, baked| {
            let current = key.1.and_then(&material);
            let kept = keep(key) && baked.material == current;
            if !kept {
                layers.release(baked.record.layer);
            }
            kept
        });
        let supplied = std::mem::take(&mut self.supplied);
        let asked = self.supplies.len();
        let dropped: Vec<ImpostorKey> = self
            .supplies
            .keys()
            .filter(|&key| !keep(key))
            .copied()
            .collect();
        for key in dropped {
            self.supplies.remove(&key);
        }
        let kept = self.supplies.len();
        self.supplies.retain(|key, _| supplied.contains(key));
        if cx.cooking && self.supplies.len() < kept {
            let record = serde_json::json!({
                "event": "impostor_supplies_dropped",
                "dropped": kept - self.supplies.len(),
                "asked": asked,
            });
            eprintln!("{record}");
        }
    }

    fn enforce_budget(&mut self, budget: Option<u32>, cooking: bool) {
        if !cooking && budget.is_some_and(|budget| self.layers.live() > budget) {
            self.baked.clear();
            self.layers = Layers::default();
        }
    }

    fn record(&self, key: ImpostorKey, material: Option<&Material>) -> Option<ImpostorRecord> {
        self.baked
            .get(&key)
            .filter(|baked| baked.material.as_ref() == material)
            .map(|baked| baked.record)
    }

    fn supply(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        key: ImpostorKey,
        bake: &Bake,
        budget: Option<u32>,
    ) -> Supplied {
        if let Some(resident) = self.record(key, bake.material.as_ref()) {
            return Supplied::Resident(resident);
        }
        self.supplied.insert(key);
        let budget = budget.filter(|_| !cx.cooking);
        if self.layers.live() >= cx.device.layer_limit(budget) {
            self.supplies.remove(&key);
            return Supplied::Refused;
        }
        let asked = self
            .supplies
            .get(&key)
            .filter(|supply| supply.material.as_ref() == bake.material.as_ref())
            .map(|supply| (*supply.interest.key(), supply.since));
        let Some((identity, since)) = asked else {
            let identity = identity(self.shape.facing(key.2), cx.device.format, bake.content);
            let fetch = cx
                .jobs
                .fetch(&IMPOSTOR, &identity, job_priority(VIEW_JOBS, 0));
            let supply = Supply {
                material: bake.material,
                interest: self.answers.ask(identity),
                fetch: Some(fetch),
                since: cx.now,
            };
            self.supplies.insert(key, supply);
            return Supplied::Coming;
        };
        match self.answers.get(&identity) {
            Some(Request::Ready(bytes)) => {
                if !cx.budget.spend("impostors", bytes.len() as u64) {
                    return Supplied::Coming;
                }
                let bytes = Arc::clone(bytes);
                if self.load(cx, key, bake, &bytes, budget) {
                    self.supplies.remove(&key);
                    return self.supplied(key, bake);
                }
            }
            Some(Request::Failed(_)) => {
                self.supplies.remove(&key);
                return Supplied::Refused;
            }
            Some(Request::Missing) => {}
            Some(Request::Asked(_)) | None => return Supplied::Coming,
        }
        if self.baking.contains(&identity)
            || self.baking.len() >= BAKES_IN_FLIGHT
            || !cx.lasted(since)
        {
            return Supplied::Coming;
        }
        self.bake(cx, key, bake, identity);
        Supplied::Coming
    }

    fn supplied(&self, key: ImpostorKey, bake: &Bake) -> Supplied {
        self.record(key, bake.material.as_ref())
            .map_or(Supplied::Refused, Supplied::Resident)
    }

    fn layer_bytes(&self, device: &AtlasDevice) -> u64 {
        device.format.level_bytes(self.shape.size()) as u64 * 4 * 4 / 3
    }

    fn layer(&mut self, cx: &mut Supplying<'_, '_, '_>, budget: Option<u32>) -> Option<u32> {
        if self.layers.live() >= cx.device.layer_limit(budget) {
            return None;
        }
        let layer = self.layers.allocate(cx.device.most_layers())?;
        let extent = self.layers.extent();
        if self.capacity < extent {
            let capacity = grown_layers(
                extent,
                self.layer_bytes(&cx.device),
                cx.device.most_layers(),
            );
            self.capacity = capacity;
            self.push(cx, AtlasWork::Grow { layers: capacity });
        }
        Some(layer)
    }

    fn load(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        key: ImpostorKey,
        bake: &Bake,
        bytes: &Arc<[u8]>,
        budget: Option<u32>,
    ) -> bool {
        let shape = self.shape.facing(key.2);
        let header = impostor_record_header(shape, cx.device.format);
        if bytes.len() != record_bytes(shape, cx.device.format)
            || bytes.get(..IMPOSTOR_RECORD_HEADER) != Some(header.as_slice())
        {
            return false;
        }
        let Some(layer) = self.layer(cx, budget) else {
            return false;
        };
        self.push(
            cx,
            AtlasWork::Load {
                layer,
                bytes: Arc::downgrade(bytes),
            },
        );
        self.resident(key, bake, layer);
        true
    }

    fn resident(&mut self, key: ImpostorKey, bake: &Bake, layer: u32) {
        let emissive = bake.material.as_ref().map_or([0.0; 3], Material::emissive);
        let record = ImpostorRecord::new(
            bake.bounds.0,
            bake.bounds.1,
            layer,
            (emissive, bake.masked),
            self.shape.facing(key.2),
        );
        self.baked.insert(
            key,
            Baked {
                record,
                material: bake.material,
            },
        );
    }

    fn bake(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        key: ImpostorKey,
        bake: &Bake,
        identity: ImpostorIdentity,
    ) {
        self.targets = true;
        self.baking.insert(identity);
        self.answers.ask_again(&identity);
        self.push(
            cx,
            AtlasWork::Bake {
                identity,
                mesh: key.0,
                material: key.1,
                parts: bake.parts.clone(),
                bounds: bake.bounds,
                hemisphere: key.2,
            },
        );
    }

    fn release_targets(&mut self, cx: &mut Supplying<'_, '_, '_>) {
        if std::mem::take(&mut self.targets) {
            self.push(cx, AtlasWork::DropTargets);
        }
    }

    fn poll(&mut self, records: &Records, now: f64) {
        self.answers.settle(now);
        let answers = &mut self.answers;
        for supply in self.supplies.values_mut() {
            let Some(fetch) = supply.fetch.as_ref() else {
                continue;
            };
            let Some(found) = records.get(fetch) else {
                continue;
            };
            supply.fetch = None;
            let identity = *supply.interest.key();
            let Some(ticket) = answers.ticket(&identity) else {
                continue;
            };
            match found {
                Ok(bytes) => answers.ready(&identity, ticket, bytes),
                Err(_) => answers.miss(&identity, ticket),
            };
        }
    }

    fn answer(&mut self, identity: ImpostorIdentity, made: Option<Result<Arc<[u8]>, Error>>) {
        self.baking.remove(&identity);
        let Some(ticket) = self.answers.ticket(&identity) else {
            return;
        };
        match made {
            Some(Ok(bytes)) => self.answers.ready(&identity, ticket, bytes),
            Some(Err(error)) => self.answers.fail(&identity, ticket, error),
            None => self.answers.miss(&identity, ticket),
        };
    }

    fn left(&self) -> u64 {
        (self.supplies.len() + self.baking.len()) as u64
    }
}

#[doc(hidden)]
#[derive(Resource)]
pub struct ImpostorBooks {
    device: Option<AtlasDevice>,
    book: Option<Book>,
    log: Feed<AtlasWork>,
    missed: bool,
    read: Vec<(ImpostorIdentity, Option<Vec<u8>>)>,
    assembling: Vec<(JobId, ImpostorIdentity)>,
    made: Vec<(ImpostorIdentity, Option<Result<Arc<[u8]>, Error>>)>,
}

impl Default for ImpostorBooks {
    fn default() -> Self {
        Self {
            device: None,
            book: None,
            log: Feed::new(FeedSettings::default().impostor_work as usize),
            missed: false,
            read: Vec::new(),
            assembling: Vec::new(),
            made: Vec::new(),
        }
    }
}

impl ImpostorBooks {
    #[doc(hidden)]
    pub fn device(&self) -> Option<AtlasDevice> {
        self.device
    }

    fn assemble(&mut self, exchange: &mut JobExchange) {
        let empty = BTreeSet::new();
        let baking = self.book.as_ref().map_or(&empty, |book| &book.baking);
        for (identity, read) in self.read.drain(..) {
            if !baking.contains(&identity) {
                continue;
            }
            match read {
                Some(input) => {
                    let kind = JobRef::from(&IMPOSTOR_ASSEMBLY);
                    let job = exchange.submit(kind, input, job_priority(IDLE_JOBS, 0));
                    self.assembling.push((job, identity));
                }
                None => self.made.push((identity, None)),
            }
        }
        let made = &mut self.made;
        self.assembling.retain(|&(job, identity)| {
            if !baking.contains(&identity) {
                exchange.cancel(job);
                return false;
            }
            let Some(output) = exchange.take(job) else {
                return true;
            };
            let output = output.map_err(|message| {
                Error::new(ErrorCode::InvalidRecord, "$.jobs.engine::impostor_assembly")
                    .got(message)
            });
            made.push((identity, Some(output)));
            false
        });
    }

    #[doc(hidden)]
    pub fn contains(&self, mesh: MeshHandle, material: Option<MaterialHandle>) -> bool {
        self.book.as_ref().is_some_and(|book| {
            book.baked
                .keys()
                .any(|&(baked, held, _)| baked == mesh && held == material)
        })
    }

    #[doc(hidden)]
    pub fn log(&self) -> &Feed<AtlasWork> {
        &self.log
    }

    pub fn census(&self) -> Census {
        self.book
            .as_ref()
            .map_or_else(Census::default, |book| book.answers.census())
    }

    #[doc(hidden)]
    pub fn left(&self) -> u64 {
        self.book.as_ref().map_or(0, Book::left)
    }
}

#[doc(hidden)]
pub fn deliver_impostors(
    world: &mut World,
    missed: bool,
    made: Vec<(ImpostorIdentity, Option<Vec<u8>>)>,
) {
    if let Some(mut books) = world.get_resource_mut::<ImpostorBooks>() {
        books.missed |= missed;
        books.read.extend(made);
    }
}

pub(super) fn configure(world: &mut World, atlas: AtlasDevice) {
    let (log, assembling) = world
        .get_resource_mut::<ImpostorBooks>()
        .map_or_else(
            || (ImpostorBooks::default().log, Vec::new()),
            |mut books| {
                let mut log = books.log.clone();
                log.replace();
                (log, std::mem::take(&mut books.assembling))
            },
        );
    if let Some(mut exchange) = world.get_resource_mut::<JobExchange>() {
        for (job, _) in assembling {
            exchange.cancel(job);
        }
    }
    world.insert_resource(ImpostorBooks {
        device: Some(atlas),
        log,
        ..ImpostorBooks::default()
    });
}

type Inputs = (
    Jobs<'static, 'static>,
    Res<'static, Records>,
    Res<'static, AssetLibrary>,
    Res<'static, AssetResidency>,
    Query<'static, 'static, (Entity, &'static ImpostorAsks)>,
);

#[derive(Resource)]
struct ImpostorInputs(SystemState<Inputs>);

struct Frame {
    policy: ImpostorPolicy,
    detail: bool,
    cooking: bool,
    now: Option<f64>,
    seconds: f64,
}

type Sources<'a> = (&'a AssetLibrary, &'a AssetResidency);

pub(super) fn settle_impostors(world: &mut World) {
    if world
        .get_resource::<ImpostorBooks>()
        .is_none_or(|books| books.device.is_none())
    {
        return;
    }
    let size = world
        .get_resource::<FeedSettings>()
        .cloned()
        .unwrap_or_default()
        .impostor_work;
    world.resource_scope(|world, mut exchange: Mut<JobExchange>| {
        let mut books = world.resource_mut::<ImpostorBooks>();
        books.log.resize(size as usize);
        books.assemble(&mut exchange);
    });
    let cooking = world.resource::<CookMode>().cooking;
    let frame = Frame {
        policy: world
            .get_resource::<ImpostorPolicy>()
            .copied()
            .unwrap_or_default(),
        detail: world.resource::<Detail>().enabled,
        cooking,
        now: world.resource::<WorkClock>().now().filter(|_| !cooking),
        seconds: world.resource::<crate::FrameTime>().real_elapsed,
    };
    if !world.contains_resource::<ImpostorInputs>() {
        let state = SystemState::<Inputs>::new(world);
        world.insert_resource(ImpostorInputs(state));
    }
    let written = world.resource_scope(|world, mut inputs: Mut<ImpostorInputs>| {
        let written = world.resource_scope(|world, mut books: Mut<ImpostorBooks>| {
            world.resource_scope(|world, mut budget: Mut<FrameBudget>| {
                let Ok((mut jobs, records, library, residency, asks)) = inputs.0.get_mut(world)
                else {
                    return Default::default();
                };
                settle_books(
                    &mut books,
                    &frame,
                    (&mut jobs, &*records, &mut budget),
                    (&*library, &*residency),
                    &asks,
                )
            })
        });
        inputs.0.apply(world);
        written
    });
    for (entity, records) in written {
        let Ok(mut target) = world.get_entity_mut(entity) else {
            continue;
        };
        if !target.contains::<ImpostorRecords>() {
            target.insert(records);
            continue;
        }
        if let Some(mut held) = target.get_mut::<ImpostorRecords>()
            && *held != records
        {
            *held = records;
        }
    }
}

fn settle_books(
    books: &mut ImpostorBooks,
    frame: &Frame,
    (jobs, records, budget): (&mut Jobs<'_, '_>, &Records, &mut FrameBudget),
    sources: Sources<'_>,
    asks: &Query<'_, '_, (Entity, &'static ImpostorAsks)>,
) -> Vec<(Entity, ImpostorRecords)> {
    let Some(device) = books.device else {
        return Vec::new();
    };
    let made = std::mem::take(&mut books.made);
    let missed = std::mem::take(&mut books.missed);
    let ImpostorBooks {
        book: held, log, ..
    } = books;
    let mut cx = Supplying {
        device,
        jobs,
        log,
        budget,
        cooking: frame.cooking,
        now: frame.now,
    };
    if missed && let Some(book) = held.as_mut() {
        let shape = book.shape;
        book.restart(&mut cx, shape);
    }
    for (identity, made) in made {
        if let Some(Ok(bytes)) = &made {
            cx.jobs.keep(&IMPOSTOR, &identity, bytes.to_vec());
        }
        if let Some(book) = held.as_mut() {
            book.answer(identity, made);
        }
    }
    let asked: Vec<(Entity, &ImpostorAsks)> = asks.iter().collect();
    if asked.is_empty() {
        if let Some(book) = held.as_mut() {
            book.supplies.clear();
            book.poll(records, frame.seconds);
        }
        return Vec::new();
    }
    let created = held.is_none();
    let book = held.get_or_insert_with(|| Book::new(AtlasShape::of(frame.policy.pixels), &device));
    if created {
        let shape = book.shape;
        book.push(&mut cx, AtlasWork::Reset { shape });
    }
    book.poll(records, frame.seconds);
    settle(book, &mut cx, frame, sources, &asked)
}

fn keep_key(residency: &AssetResidency) -> impl Fn(&ImpostorKey) -> bool + '_ {
    move |key: &ImpostorKey| {
        let &(mesh, material, _) = key;
        residency.resident(mesh)
            && material.is_none_or(|handle| residency.material_resident(handle))
    }
}

fn current_material<'a>(
    (library, residency): Sources<'a>,
) -> impl Fn(MaterialHandle) -> Option<Material> + 'a {
    move |handle: MaterialHandle| {
        residency
            .material_resident(handle)
            .then(|| library.material(handle).cloned())
            .flatten()
    }
}

fn retain_book(book: &mut Book, cx: &mut Supplying<'_, '_, '_>, sources: Sources<'_>) {
    book.retain(cx, keep_key(sources.1), current_material(sources));
}

fn bake_of(
    (library, residency): Sources<'_>,
    ask: &ImpostorAsk,
    material: Option<Material>,
) -> Option<Bake> {
    let mesh = library.mesh(ask.mesh)?;
    let mut content = (
        mesh.identity(content_hash),
        material_content(library, material.as_ref()),
    );
    let bounds = mesh.bounds();
    let (mut low, mut high) = (bounds.min(), bounds.max());
    let mut masked_any = masked(material.as_ref());
    let mut parts = Vec::with_capacity(ask.parts.len());
    for &(part_mesh, part_material) in &ask.parts {
        if !residency.resident(part_mesh) {
            continue;
        }
        let (Some(finest), Some(value)) =
            (library.mesh(part_mesh), library.material(part_material))
        else {
            continue;
        };
        content = (
            combined(content.0, finest.identity(content_hash)),
            combined(content.1, material_content(library, Some(value))),
        );
        let bounds = finest.bounds();
        low = low.min(bounds.min());
        high = high.max(bounds.max());
        masked_any |= masked(Some(value));
        parts.push((part_mesh, part_material));
    }
    Some(Bake {
        material,
        bounds: (low, high),
        content,
        masked: masked_any,
        parts,
    })
}

fn supply_ask(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    sources: Sources<'_>,
    (ask, hemisphere): (&ImpostorAsk, bool),
    budget: Option<u32>,
) -> Option<Supplied> {
    if !sources.1.resident(ask.mesh) {
        return None;
    }
    let material = ask.material.and_then(current_material(sources));
    let key = (ask.mesh, ask.material, hemisphere);
    if let Some(resident) = book.record(key, material.as_ref()) {
        return Some(Supplied::Resident(resident));
    }
    let bake = bake_of(sources, ask, material)?;
    Some(book.supply(cx, key, &bake, budget))
}

fn supply_all(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    sources: Sources<'_>,
    (asked, enabled, budget): (&ImpostorAsks, bool, Option<u32>),
    settled: &mut ImpostorRecords,
) {
    for ask in &asked.asks {
        let supplied = match ask {
            Some(ask) if enabled => supply_ask(book, cx, sources, (ask, asked.hemisphere), budget),
            Some(_) | None => None,
        };
        let record = match supplied {
            Some(supplied) => {
                settled.asked += 1;
                settled.refused += u32::from(matches!(supplied, Supplied::Refused));
                supplied.record()
            }
            None => ImpostorRecord::default(),
        };
        settled.records.push(record);
    }
}

fn live_layers(records: &[ImpostorRecord]) -> u32 {
    records
        .iter()
        .filter(|record| record.framed())
        .map(|record| record.layer)
        .collect::<BTreeSet<u32>>()
        .len() as u32
}

fn settle(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    frame: &Frame,
    sources: Sources<'_>,
    asked: &[(Entity, &ImpostorAsks)],
) -> Vec<(Entity, ImpostorRecords)> {
    book.fit(cx, AtlasShape::of(frame.policy.pixels));
    retain_book(book, cx, sources);
    let shown = frame.detail && frame.policy.enabled;
    if !shown {
        book.release(cx);
    } else {
        book.enforce_budget(frame.policy.max, cx.cooking);
    }
    let mut written = Vec::with_capacity(asked.len());
    for (entity, asks) in asked {
        let mut settled = ImpostorRecords {
            generation: asks.generation,
            ..ImpostorRecords::default()
        };
        supply_all(
            book,
            cx,
            sources,
            (*asks, shown, frame.policy.max),
            &mut settled,
        );
        book.drawn |= book.capacity > 0 && !settled.records.is_empty();
        settled.live = live_layers(&settled.records);
        written.push((*entity, settled));
    }
    book.release_targets(cx);
    written
}
