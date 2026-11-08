use std::collections::BTreeSet;
use std::sync::{Arc, Weak};

use bevy_ecs::prelude::*;
use engine_core::collections::{HashMap, HashSet};
use engine_core::glam::Vec3;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Census, CookMode, Cull, Detail, DeviceLimits, Error,
    ErrorCode, Feed, Filter, FrameBudget, FrameTime, GpuBits, IDLE_JOBS, Interest, Job, JobCx,
    JobExchange, JobId, JobKind, JobRef, Jobs, LightingModel, Material, MaterialHandle,
    MeshHandle, RecordKey, Records, Request, Requests, TextureFamily, TextureRef, VIEW_JOBS,
    WorkClock, Wrap, compress_texture, content_hash, job_priority,
};
use engine_places::impostors::{
    ImpostorAsk, ImpostorAsks, ImpostorFlags, ImpostorRecord, ImpostorRecords, Impostors,
};
use engine_render::wgpu;

use crate::part::BakeReads;
use crate::settings::{ImpostorSettings, ImpostorStats};

pub(crate) const IMPOSTOR_KINDS: u32 = 4;

const BAKES_IN_FLIGHT: usize = 4;

const BAKE_DELAY_MS: f64 = 1000.0;

const BLOCK_ROWS: u32 = 4;

const RGBA8_SIZE: u32 = 512;

const ATLAS_FRAMES: u32 = 12;

const FIRST_BYTES: u64 = 16 << 20;

const RECORD_MAGIC: &[u8; 4] = b"EIMP";

const RECORD_HEADER: usize = 24;

pub(crate) const RECORD_START: usize = RECORD_HEADER + 4;

pub(crate) const ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

pub(crate) fn record_of(
    (min, max): (Vec3, Vec3),
    layer: u32,
    (emissive, thin): ([f32; 3], bool),
    shape: AtlasShape,
) -> ImpostorRecord {
    let center = (min + max) * 0.5;
    let half = (max - min) * 0.5;
    let radius = engine_core::math::max(half.length() * 1.0001, 1.0e-6);
    ImpostorRecord {
        sphere: center.extend(radius).to_array(),
        extent: half.to_array(),
        layer,
        emissive,
        flags: ImpostorFlags {
            thin,
            ..shape.flags()
        }
        .pack(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AtlasShape {
    pub(crate) frames: u32,
    pub(crate) texels: u32,
    pub(crate) hemisphere: bool,
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

    pub(crate) fn facing(self, hemisphere: bool) -> Self {
        Self { hemisphere, ..self }
    }

    pub(crate) fn size(self) -> u32 {
        self.frames * self.texels
    }

    pub(crate) fn levels(self) -> u32 {
        self.texels.trailing_zeros()
    }

    fn flags(self) -> ImpostorFlags {
        ImpostorFlags {
            texels: self.texels,
            hemisphere: self.hemisphere,
            thin: false,
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
            || readback_bytes(shape, ROW_ALIGNMENT) + 4 > device.max_buffer_size)
            && shape.texels > 2
        {
            shape.texels /= 2;
        }
        shape
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtlasFormat {
    Bc7,
    Astc,
    Rgba8,
}

impl AtlasFormat {
    pub(crate) fn tag(self) -> u32 {
        match self {
            Self::Rgba8 => 0,
            Self::Bc7 => 1,
            Self::Astc => 2,
        }
    }

    pub(crate) fn of_tag(tag: u32) -> Option<Self> {
        match tag {
            0 => Some(Self::Rgba8),
            1 => Some(Self::Bc7),
            2 => Some(Self::Astc),
            _ => None,
        }
    }

    pub(crate) fn rows(self, size: u32) -> (u32, u32) {
        match self {
            Self::Bc7 | Self::Astc => (size.div_ceil(4) * 16, size.div_ceil(4)),
            Self::Rgba8 => (size * 4, size),
        }
    }

    fn level_bytes(self, size: u32) -> usize {
        let (row, rows) = self.rows(size);
        row as usize * rows as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AtlasDevice {
    format: AtlasFormat,
    max_texture_dimension: u32,
    max_buffer_size: u64,
    max_layers: u32,
}

impl AtlasDevice {
    fn of(format: AtlasFormat, limits: &DeviceLimits) -> Self {
        Self {
            format,
            max_texture_dimension: limits.max_texture_dimension_2d,
            max_buffer_size: limits.storage_room,
            max_layers: limits.texture_layers,
        }
    }

    fn layer_limit(&self, budget: Option<u32>) -> u32 {
        self.max_layers.min(budget.unwrap_or(u32::MAX))
    }
}

pub(crate) fn level_rows(shape: AtlasShape, level: u32, alignment: u32) -> (u32, u32, u32) {
    let (row, rows) = AtlasFormat::Rgba8.rows(shape.size() >> level);
    (rows, row.next_multiple_of(alignment), row)
}

pub(crate) fn readback_bytes(shape: AtlasShape, alignment: u32) -> u64 {
    u64::from(IMPOSTOR_KINDS)
        * (0..shape.levels())
            .map(|level| {
                let (rows, stride, _) = level_rows(shape, level, alignment);
                u64::from(stride) * u64::from(rows)
            })
            .sum::<u64>()
}

fn record_header(shape: AtlasShape, format: AtlasFormat) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_HEADER);
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
    RECORD_START
        + IMPOSTOR_KINDS as usize
            * (0..shape.levels())
                .map(|level| format.level_bytes(shape.size() >> level))
                .sum::<usize>()
}

type ImpostorIdentity = [u8; 48];

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

pub(crate) static BAKE: JobKind = JobKind::new("impostors::bake", &crate::SOURCE, made_on_the_gpu);

fn made_on_the_gpu(_: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs.impostors::bake").got("impostors::bake"))
}

pub(crate) static ASSEMBLY: JobKind =
    JobKind::new("impostors::assembly", &crate::SOURCE, start_assembly);

const ASSEMBLY_HEADER: usize = RECORD_HEADER + 4;

fn assembly_input(shape: AtlasShape, format: AtlasFormat, read: &[u8]) -> Vec<u8> {
    let mut input = Vec::with_capacity(ASSEMBLY_HEADER + read.len());
    input.extend_from_slice(&record_header(shape, format));
    input.extend_from_slice(&ROW_ALIGNMENT.to_le_bytes());
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
    family: Option<TextureFamily>,
    alignment: u32,
    out: Vec<u8>,
    region: u32,
    at: usize,
    row: u32,
}

impl Assembly {
    fn parse(input: &[u8]) -> Option<Self> {
        let format = AtlasFormat::of_tag(input_word(input, 4)?)?;
        let shape = AtlasShape {
            frames: input_word(input, 8)?,
            texels: input_word(input, 12)?,
            hemisphere: input_word(input, 16)? != 0,
        };
        let alignment = input_word(input, RECORD_HEADER)?;
        let regions = readback_bytes(shape, alignment);
        if input.len() as u64 != ASSEMBLY_HEADER as u64 + regions + 4 {
            return None;
        }
        let thin = input_word(input, input.len() - 4)?;
        let mut out = Vec::with_capacity(record_bytes(shape, format));
        out.extend_from_slice(&input[..RECORD_HEADER]);
        out.extend_from_slice(&u32::from(thin != 0).to_le_bytes());
        Some(Self {
            input: input.to_vec(),
            shape,
            family: match format {
                AtlasFormat::Bc7 => Some(TextureFamily::Bc7),
                AtlasFormat::Astc => Some(TextureFamily::Astc),
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
        let (rows, stride, row) = level_rows(self.shape, level, self.alignment);
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
                let blocks = compress_texture(size, band, &texels, family);
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
        Error::new(ErrorCode::InvalidRecord, "$.jobs.impostors::assembly").got(input.len())
    })?;
    Ok(Box::new(job))
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
    parts: Vec<(MeshHandle, MaterialHandle)>,
}

#[derive(Clone, Debug)]
pub(crate) enum AtlasWork {
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
        serial: u32,
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
        parts: Vec<(MeshHandle, MaterialHandle)>,
        bounds: (Vec3, Vec3),
        hemisphere: bool,
    },
}

struct Book {
    shape: AtlasShape,
    layers: Layers,
    capacity: u32,
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
    serials: &'a mut Serials,
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

#[derive(Default)]
struct Serials {
    next: u32,
    baking: HashMap<u32, (ImpostorIdentity, AtlasShape)>,
}

impl Serials {
    fn take(&mut self, identity: ImpostorIdentity, shape: AtlasShape) -> u32 {
        self.next = self.next.wrapping_add(1) % crate::part::MISSED;
        self.baking.insert(self.next, (identity, shape));
        self.next
    }
}

impl Book {
    fn new(shape: AtlasShape, device: &AtlasDevice) -> Self {
        Self {
            shape: shape.held_on(device),
            layers: Layers::default(),
            capacity: 0,
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
        self.drawn = false;
        self.layers = Layers::default();
        self.baked.clear();
        self.supplies.clear();
        self.supplied.clear();
        self.baking.clear();
        cx.serials.baking.clear();
        self.push(cx, AtlasWork::Reset { shape });
    }

    fn release(&mut self, cx: &mut Supplying<'_, '_, '_>) {
        if self.capacity == 0
            && self.baked.is_empty()
            && self.supplies.is_empty()
            && self.baking.is_empty()
            && !self.drawn
        {
            return;
        }
        self.capacity = 0;
        self.drawn = false;
        self.layers = Layers::default();
        self.baked.clear();
        self.supplies.clear();
        self.supplied.clear();
        self.baking.clear();
        cx.serials.baking.clear();
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
            let fetch = cx.jobs.fetch(&BAKE, &identity, job_priority(VIEW_JOBS, 0));
            let supply = Supply {
                material: bake.material.clone(),
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
        let layer = self.layers.allocate(cx.device.max_layers)?;
        let extent = self.layers.extent();
        if self.capacity < extent {
            let capacity = grown_layers(
                extent,
                self.layer_bytes(&cx.device),
                cx.device.max_layers,
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
        let header = record_header(shape, cx.device.format);
        if bytes.len() != record_bytes(shape, cx.device.format)
            || bytes.get(..RECORD_HEADER) != Some(header.as_slice())
        {
            return false;
        }
        let thin = input_word(bytes, RECORD_HEADER).is_some_and(|word| word != 0);
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
        self.resident(key, bake, (layer, thin));
        true
    }

    fn resident(&mut self, key: ImpostorKey, bake: &Bake, (layer, thin): (u32, bool)) {
        let emissive = bake.material.as_ref().map_or([0.0; 3], Material::emissive);
        let record = record_of(bake.bounds, layer, (emissive, thin), self.shape.facing(key.2));
        self.baked.insert(
            key,
            Baked {
                record,
                material: bake.material.clone(),
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
        self.baking.insert(identity);
        self.answers.ask_again(&identity);
        let serial = cx.serials.take(identity, self.shape.facing(key.2));
        self.push(
            cx,
            AtlasWork::Bake {
                serial,
                mesh: key.0,
                material: key.1,
                parts: bake.parts.clone(),
                bounds: bake.bounds,
                hemisphere: key.2,
            },
        );
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

    fn answer(&mut self, identity: ImpostorIdentity, made: Made) {
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

type Made = Option<Result<Arc<[u8]>, Error>>;

#[derive(Resource)]
pub(crate) struct ImpostorBooks {
    format: Option<AtlasFormat>,
    book: Option<Book>,
    log: Feed<AtlasWork>,
    serials: Serials,
    missed: bool,
    read: Vec<(ImpostorIdentity, Option<Vec<u8>>)>,
    assembling: Vec<(JobId, ImpostorIdentity)>,
    made: Vec<(ImpostorIdentity, Made)>,
}

impl Default for ImpostorBooks {
    fn default() -> Self {
        Self {
            format: None,
            book: None,
            log: Feed::new(ImpostorSettings::default().work as usize),
            serials: Serials::default(),
            missed: false,
            read: Vec::new(),
            assembling: Vec::new(),
            made: Vec::new(),
        }
    }
}

impl ImpostorBooks {
    pub(crate) fn log(&self) -> &Feed<AtlasWork> {
        &self.log
    }

    fn configure(&mut self, format: AtlasFormat, exchange: Option<&mut JobExchange>) {
        if let Some(exchange) = exchange {
            for (job, _) in self.assembling.drain(..) {
                exchange.cancel(job);
            }
        }
        self.log.replace();
        self.format = Some(format);
        self.book = None;
        self.serials = Serials::default();
        self.missed = false;
        self.read.clear();
        self.assembling.clear();
        self.made.clear();
    }

    fn arrived(&mut self, serial: u32, bytes: Vec<u8>) {
        let (Some(format), Some((identity, shape))) =
            (self.format, self.serials.baking.remove(&serial))
        else {
            return;
        };
        let expected = readback_bytes(shape, ROW_ALIGNMENT) + 4;
        let input = (bytes.len() as u64 == expected).then(|| assembly_input(shape, format, &bytes));
        self.read.push((identity, input));
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
                    let kind = JobRef::from(&ASSEMBLY);
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
                Error::new(ErrorCode::InvalidRecord, "$.jobs.impostors::assembly").got(message)
            });
            made.push((identity, Some(output)));
            false
        });
    }

    fn census(&self) -> Census {
        self.book
            .as_ref()
            .map_or_else(Census::default, |book| book.answers.census())
    }

    fn left(&self) -> u64 {
        self.book.as_ref().map_or(0, Book::left)
    }
}

pub(crate) fn assemble(
    settings: Res<ImpostorSettings>,
    mut books: ResMut<ImpostorBooks>,
    mut reads: ResMut<BakeReads>,
    mut exchange: Option<ResMut<JobExchange>>,
) {
    if let Some(format) = reads.format.take().and_then(AtlasFormat::of_tag) {
        books.configure(format, exchange.as_deref_mut());
    }
    books.missed |= std::mem::take(&mut reads.missed);
    books.log.resize(settings.work as usize);
    for (serial, bytes) in std::mem::take(&mut reads.read) {
        books.arrived(serial, bytes);
    }
    if let Some(exchange) = exchange.as_deref_mut() {
        books.assemble(exchange);
    }
}

pub(crate) fn policy(
    settings: Res<ImpostorSettings>,
    mut impostors: ResMut<Impostors>,
) {
    impostors.set_if_neq(Impostors {
        enabled: settings.enabled,
        pixels: settings.pixels,
    });
}

struct Frame {
    enabled: bool,
    pixels: f32,
    max: Option<u32>,
    detail: bool,
    cooking: bool,
    now: Option<f64>,
    seconds: f64,
}

type Sources<'a> = (&'a AssetLibrary, &'a AssetResidency);

pub(crate) fn settle(
    (settings, detail, cook, clock, time, limits): (
        Res<ImpostorSettings>,
        Res<Detail>,
        Res<CookMode>,
        Res<WorkClock>,
        Res<FrameTime>,
        Res<DeviceLimits>,
    ),
    (library, residency, records): (Res<AssetLibrary>, Res<AssetResidency>, Res<Records>),
    (mut books, mut budget): (ResMut<ImpostorBooks>, ResMut<FrameBudget>),
    mut jobs: Jobs,
    asks: Query<(Entity, &ImpostorAsks)>,
    mut held: Query<&mut ImpostorRecords>,
    mut stats: ResMut<ImpostorStats>,
) {
    let cooking = cook.cooking;
    let frame = Frame {
        enabled: settings.enabled,
        pixels: settings.pixels,
        max: settings.max,
        detail: detail.enabled,
        cooking,
        now: clock.now().filter(|_| !cooking),
        seconds: time.real_elapsed,
    };
    let books = &mut *books;
    let written = settle_books(
        books,
        (&frame, &limits),
        (&mut jobs, &records, &mut budget),
        (&library, &residency),
        &asks,
    );
    for (entity, records) in written {
        if let Ok(mut current) = held.get_mut(entity) {
            current.set_if_neq(records);
        }
    }
    let census = books.census();
    stats.set_if_neq(ImpostorStats {
        pending: u32::try_from(books.left()).unwrap_or(u32::MAX),
        failed: census.failed,
    });
}

fn settle_books(
    books: &mut ImpostorBooks,
    (frame, limits): (&Frame, &DeviceLimits),
    (jobs, records, budget): (&mut Jobs<'_, '_>, &Records, &mut FrameBudget),
    sources: Sources<'_>,
    asks: &Query<'_, '_, (Entity, &ImpostorAsks)>,
) -> Vec<(Entity, ImpostorRecords)> {
    let Some(format) = books.format else {
        return Vec::new();
    };
    let device = AtlasDevice::of(format, limits);
    let made = std::mem::take(&mut books.made);
    let missed = std::mem::take(&mut books.missed);
    let ImpostorBooks {
        book: held,
        log,
        serials,
        ..
    } = books;
    let mut cx = Supplying {
        device,
        jobs,
        log,
        serials,
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
            cx.jobs.keep(&BAKE, &identity, bytes.to_vec());
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
    let book =
        held.get_or_insert_with(|| Book::new(AtlasShape::of(frame.pixels), &device));
    if created {
        let shape = book.shape;
        book.push(&mut cx, AtlasWork::Reset { shape });
    }
    book.poll(records, frame.seconds);
    settle_book(book, &mut cx, frame, sources, &asked)
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
        parts.push((part_mesh, part_material));
    }
    Some(Bake {
        material,
        bounds: (low, high),
        content,
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

fn settle_book(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    frame: &Frame,
    sources: Sources<'_>,
    asked: &[(Entity, &ImpostorAsks)],
) -> Vec<(Entity, ImpostorRecords)> {
    book.fit(cx, AtlasShape::of(frame.pixels));
    retain_book(book, cx, sources);
    let shown = frame.detail && frame.enabled;
    if !shown {
        book.release(cx);
    } else {
        book.enforce_budget(frame.max, cx.cooking);
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
            (*asks, shown, frame.max),
            &mut settled,
        );
        book.drawn |= book.capacity > 0 && !settled.records.is_empty();
        settled.live = live_layers(&settled.records);
        written.push((*entity, settled));
    }
    written
}
