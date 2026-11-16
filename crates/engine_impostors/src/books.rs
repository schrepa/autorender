use std::collections::BTreeSet;
use std::sync::{Arc, Weak};

use bevy_ecs::prelude::*;
use bevy_ecs::world::World;
use engine_core::collections::{HashMap, HashSet};
use engine_core::glam::Vec3;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Background, Census, CookAsks, Cull, Detail,
    DeviceLimits, Error, ErrorCode, Feed, Filter, FrameBudget, FrameTime, GeometryHandle, GpuBits,
    Hold, IDLE_JOBS, Job, JobCx, JobKind, Jobs, Landed, LightingModel, Material, MaterialHandle,
    Records, Request, Requests, Settling, Switched, TextureRef, VIEW_JOBS, Wrap, content_hash,
    gpu_struct, job_priority,
};
use engine_places_rendering::impostors::{
    ImpostorAsk, ImpostorAsks, ImpostorFlags, ImpostorRecord, ImpostorRecords,
};
use engine_render::wgpu;

use crate::part::BakeReads;
use crate::settings::{ImpostorSettings, ImpostorStats};

pub(crate) const BASE_KINDS: u32 = 3;

const BAKES_IN_FLIGHT: usize = 4;

const ATLAS_FRAMES: u32 = 12;

const RECORD_MAGIC: &[u8; 4] = b"EIMP";

const RECORD_HEADER: usize = 24;

const RECORD_EMISSIVE: usize = 20;

pub(crate) const RECORD_START: usize = RECORD_HEADER + 4;

pub(crate) const ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

pub(crate) fn kinds(emissive: bool) -> u32 {
    BASE_KINDS + u32::from(emissive)
}

fn emits(material: Option<&Material>) -> bool {
    material.is_some_and(|material| material.emissive() != [0.0; 3])
}

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

    fn held_on(self, device: &AtlasDevice) -> Self {
        let mut shape = self;
        while (shape.size() > device.max_texture_dimension
            || readback_bytes(shape, ROW_ALIGNMENT, true) + 4 > device.max_buffer_size)
            && shape.texels > 2
        {
            shape.texels /= 2;
        }
        shape
    }
}

pub(crate) fn rows(size: u32) -> (u32, u32) {
    (size * 4, size)
}

fn level_bytes(size: u32) -> usize {
    let (row, rows) = rows(size);
    row as usize * rows as usize
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AtlasDevice {
    max_texture_dimension: u32,
    max_buffer_size: u64,
    max_layers: u32,
}

impl AtlasDevice {
    fn of(limits: &DeviceLimits) -> Self {
        Self {
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
    let (row, rows) = rows(shape.size() >> level);
    (rows, row.next_multiple_of(alignment), row)
}

pub(crate) fn readback_bytes(shape: AtlasShape, alignment: u32, emissive: bool) -> u64 {
    u64::from(kinds(emissive))
        * (0..shape.levels())
            .map(|level| {
                let (rows, stride, _) = level_rows(shape, level, alignment);
                u64::from(stride) * u64::from(rows)
            })
            .sum::<u64>()
}

fn record_header(shape: AtlasShape, emissive: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_HEADER);
    bytes.extend_from_slice(RECORD_MAGIC);
    for word in [
        shape.frames,
        shape.texels,
        u32::from(shape.hemisphere),
        shape.levels(),
        u32::from(emissive),
    ] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn record_bytes(shape: AtlasShape, emissive: bool) -> usize {
    RECORD_START
        + kinds(emissive) as usize
            * (0..shape.levels())
                .map(|level| level_bytes(shape.size() >> level))
                .sum::<usize>()
}

type ImpostorIdentity = [u8; 44];

fn identity(shape: AtlasShape, content: (u128, u128)) -> ImpostorIdentity {
    let mut bytes = [0_u8; 44];
    for (at, word) in [shape.frames, shape.texels, u32::from(shape.hemisphere)]
        .into_iter()
        .enumerate()
    {
        bytes[at * 4..at * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes[12..28].copy_from_slice(&content.0.to_le_bytes());
    bytes[28..44].copy_from_slice(&content.1.to_le_bytes());
    bytes
}

pub(crate) static BAKE: JobKind = JobKind::new("impostors::bake", &crate::SOURCE, made_on_the_gpu);

fn made_on_the_gpu(_: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs.impostors::bake").got("impostors::bake"))
}

pub(crate) static ASSEMBLY: JobKind =
    JobKind::new("impostors::assembly", &crate::SOURCE, start_assembly);

const ASSEMBLY_HEADER: usize = RECORD_HEADER + 4;

fn assembly_input(shape: AtlasShape, emissive: bool, read: &[u8]) -> Vec<u8> {
    let mut input = Vec::with_capacity(ASSEMBLY_HEADER + read.len());
    input.extend_from_slice(&record_header(shape, emissive));
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
    kinds: u32,
    alignment: u32,
    out: Vec<u8>,
    region: u32,
    at: usize,
}

impl Assembly {
    fn parse(input: &[u8]) -> Option<Self> {
        let shape = AtlasShape {
            frames: input_word(input, 4)?,
            texels: input_word(input, 8)?,
            hemisphere: input_word(input, 12)? != 0,
        };
        let emissive = input_word(input, RECORD_EMISSIVE)? != 0;
        let alignment = input_word(input, RECORD_HEADER)?;
        let regions = readback_bytes(shape, alignment, emissive);
        if input.len() as u64 != ASSEMBLY_HEADER as u64 + regions + 4 {
            return None;
        }
        let thin = input_word(input, input.len() - 4)?;
        let mut out = Vec::with_capacity(record_bytes(shape, emissive));
        out.extend_from_slice(&input[..RECORD_HEADER]);
        out.extend_from_slice(&u32::from(thin != 0).to_le_bytes());
        Some(Self {
            input: input.to_vec(),
            shape,
            kinds: kinds(emissive),
            alignment,
            out,
            region: 0,
            at: ASSEMBLY_HEADER,
        })
    }

    fn level(&mut self, level: u32) {
        let (rows, stride, row) = level_rows(self.shape, level, self.alignment);
        for at in 0..rows {
            let start = self.at + at as usize * stride as usize;
            self.out
                .extend_from_slice(&self.input[start..start + row as usize]);
        }
        self.at += stride as usize * rows as usize;
        self.region += 1;
    }

    fn blank(&self) -> bool {
        let (rows, stride, row) = level_rows(self.shape, 0, self.alignment);
        (0..rows as usize).all(|at| {
            let start = ASSEMBLY_HEADER + at * stride as usize;
            self.input[start..start + row as usize]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|texel| texel[3] == 0)
        })
    }
}

impl Job for Assembly {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let levels = self.shape.levels();
        if self.region == 0 && self.blank() {
            return Err(
                Error::new(ErrorCode::RenderFailed, "$.jobs.impostors::assembly").got("blank")
            );
        }
        if self.region < self.kinds * levels {
            self.level(self.region % levels);
        }
        Ok((self.region >= self.kinds * levels).then(|| std::mem::take(&mut self.out)))
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
        bytes.extend_from_slice(slot.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(module.as_bytes());
        bytes.push(0);
    }
    for (name, value) in material.values() {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        floats(&mut bytes, &[*value]);
    }
    for (name, value) in material.surface() {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        floats(&mut bytes, &[*value]);
    }
    floats(&mut bytes, &[material.emission(), material.normal_scale()]);
    floats(&mut bytes, &material.emissive());
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

type ImpostorKey = (GeometryHandle, Option<MaterialHandle>, bool);

struct Baked {
    record: ImpostorRecord,
    material: Option<Material>,
    content: (u128, u128),
}

struct Supply {
    material: Option<Material>,
    identity: ImpostorIdentity,
    _held: Hold,
    fetch: Option<Background<Arc<[u8]>>>,
}

enum Supplied {
    Resident(ImpostorRecord),
    Coming,
    Waiting,
    Refused,
}

impl Supplied {
    fn record(&self) -> ImpostorRecord {
        match self {
            Self::Resident(record) => *record,
            Self::Coming | Self::Waiting | Self::Refused => ImpostorRecord::default(),
        }
    }
}

struct Bake<'a> {
    material: Option<&'a Material>,
    bounds: (Vec3, Vec3),
    content: (u128, u128),
    parts: Vec<(GeometryHandle, MaterialHandle)>,
}

#[derive(Clone, Debug)]
pub(crate) enum AtlasWork {
    Reset {
        shape: AtlasShape,
    },
    Release,
    Grow {
        layers: u32,
        emissive: bool,
    },
    Load {
        layer: u32,
        bytes: Weak<[u8]>,
        emissive: bool,
    },
    Bake {
        serial: u32,
        mesh: GeometryHandle,
        material: Option<MaterialHandle>,
        parts: Vec<(GeometryHandle, MaterialHandle)>,
        bounds: (Vec3, Vec3),
        hemisphere: bool,
        emissive: bool,
    },
}

struct Book {
    shape: AtlasShape,
    layers: Layers,
    capacity: u32,
    emissive: bool,
    wanted: HashSet<ImpostorKey>,
    wanting: HashSet<ImpostorKey>,
    drawn: bool,
    baked: HashMap<ImpostorKey, Baked>,
    supplies: HashMap<ImpostorKey, Supply>,
    supplied: HashSet<ImpostorKey>,
    answers: Requests<ImpostorIdentity, Arc<[u8]>>,
    baking: BTreeSet<ImpostorIdentity>,
    waiting: u32,
}

struct Supplying<'a, 'w, 's> {
    device: AtlasDevice,
    jobs: &'a mut Jobs<'w, 's>,
    log: &'a mut Feed<AtlasWork>,
    serials: &'a mut Serials,
    budget: &'a mut FrameBudget,
    cooking: bool,
}

#[derive(Default)]
struct Serials {
    next: u32,
    baking: HashMap<u32, (ImpostorIdentity, AtlasShape, bool)>,
}

impl Serials {
    fn take(&mut self, identity: ImpostorIdentity, shape: AtlasShape, emissive: bool) -> u32 {
        self.next = self.next.wrapping_add(1) % crate::part::MISSED;
        self.baking.insert(self.next, (identity, shape, emissive));
        self.next
    }
}

impl Book {
    fn new(shape: AtlasShape, device: &AtlasDevice) -> Self {
        Self {
            shape: shape.held_on(device),
            layers: Layers::default(),
            capacity: 0,
            emissive: false,
            wanted: HashSet::default(),
            wanting: HashSet::default(),
            drawn: false,
            baked: HashMap::default(),
            supplies: HashMap::default(),
            supplied: HashSet::default(),
            answers: Requests::default(),
            baking: BTreeSet::new(),
            waiting: 0,
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
        self.emissive = false;
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
        self.emissive = false;
        self.drawn = false;
        self.layers = Layers::default();
        self.baked.clear();
        self.supplies.clear();
        self.supplied.clear();
        self.baking.clear();
        cx.serials.baking.clear();
        self.push(cx, AtlasWork::Release);
    }

    fn retain<'m>(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        keep: impl Fn(&ImpostorKey) -> bool,
        material: impl Fn(MaterialHandle) -> Option<&'m Material>,
    ) {
        let layers = &mut self.layers;
        self.baked.retain(|key, baked| {
            let current = key.1.and_then(&material);
            let kept = keep(key) && baked.material.as_ref() == current;
            if !kept {
                layers.release(baked.record.layer);
            }
            kept
        });
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
        let supplied = &self.supplied;
        self.supplies.retain(|key, _| supplied.contains(key));
        self.supplied.clear();
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

    fn record(&self, key: ImpostorKey, bake: &Bake<'_>) -> Option<ImpostorRecord> {
        self.baked
            .get(&key)
            .filter(|baked| baked.material.as_ref() == bake.material)
            .filter(|baked| baked.content == bake.content)
            .map(|baked| baked.record)
    }

    fn supply(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        key: ImpostorKey,
        bake: &Bake<'_>,
        budget: Option<u32>,
    ) -> Supplied {
        if let Some(resident) = self.record(key, bake) {
            return Supplied::Resident(resident);
        }
        if let Some(stale) = self.baked.remove(&key) {
            self.layers.release(stale.record.layer);
        }
        self.supplied.insert(key);
        let budget = budget.filter(|_| !cx.cooking);
        if self.layers.live() >= cx.device.layer_limit(budget) {
            self.supplies.remove(&key);
            return Supplied::Refused;
        }
        let identity = identity(self.shape.facing(key.2), bake.content);
        let asked = self
            .supplies
            .get(&key)
            .filter(|supply| supply.material.as_ref() == bake.material)
            .filter(|supply| supply.identity == identity);
        if asked.is_none() {
            let fetch = cx
                .jobs
                .look(&BAKE, &identity, job_priority(VIEW_JOBS, 0), Ok);
            let supply = Supply {
                material: bake.material.cloned(),
                identity,
                _held: self.answers.ask(identity).into(),
                fetch: Some(fetch),
            };
            self.supplies.insert(key, supply);
            return Supplied::Coming;
        }
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
            Some(Request::Failed(_)) => return Supplied::Refused,
            Some(Request::Missing) => {}
            Some(Request::Asked(_)) | None => return Supplied::Coming,
        }
        if self.baking.contains(&identity) || self.baking.len() >= BAKES_IN_FLIGHT {
            return Supplied::Coming;
        }
        self.bake(cx, key, bake, identity);
        Supplied::Coming
    }

    fn supplied(&self, key: ImpostorKey, bake: &Bake<'_>) -> Supplied {
        self.record(key, bake)
            .map_or(Supplied::Refused, Supplied::Resident)
    }

    fn layer(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        (key, emissive): (ImpostorKey, bool),
        budget: Option<u32>,
    ) -> Option<u32> {
        let limit = cx.device.layer_limit(budget);
        if self.layers.live() >= limit {
            return None;
        }
        let layer = self.layers.allocate(cx.device.max_layers)?;
        let extent = self.layers.extent();
        let grown = self.capacity < extent || (emissive && !self.emissive);
        if self.capacity < extent {
            let coming = self
                .wanted
                .iter()
                .filter(|wanted| **wanted != key && !self.baked.contains_key(*wanted))
                .count();
            let room = limit - self.layers.live();
            self.capacity = extent + u32::try_from(coming).unwrap_or(u32::MAX).min(room);
        }
        self.emissive |= emissive;
        if grown {
            let work = AtlasWork::Grow {
                layers: self.capacity,
                emissive: self.emissive,
            };
            self.push(cx, work);
        }
        Some(layer)
    }

    fn load(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        key: ImpostorKey,
        bake: &Bake<'_>,
        bytes: &Arc<[u8]>,
        budget: Option<u32>,
    ) -> bool {
        let shape = self.shape.facing(key.2);
        let emissive = emits(bake.material);
        let header = record_header(shape, emissive);
        if bytes.len() != record_bytes(shape, emissive)
            || bytes.get(..RECORD_HEADER) != Some(header.as_slice())
        {
            return false;
        }
        let thin = input_word(bytes, RECORD_HEADER).is_some_and(|word| word != 0);
        let Some(layer) = self.layer(cx, (key, emissive), budget) else {
            return false;
        };
        self.push(
            cx,
            AtlasWork::Load {
                layer,
                bytes: Arc::downgrade(bytes),
                emissive,
            },
        );
        self.resident(key, bake, (layer, thin));
        true
    }

    fn resident(&mut self, key: ImpostorKey, bake: &Bake<'_>, (layer, thin): (u32, bool)) {
        let emissive = bake.material.map_or([0.0; 3], Material::emissive);
        let record = record_of(
            bake.bounds,
            layer,
            (emissive, thin),
            self.shape.facing(key.2),
        );
        self.baked.insert(
            key,
            Baked {
                record,
                material: bake.material.cloned(),
                content: bake.content,
            },
        );
    }

    fn bake(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        key: ImpostorKey,
        bake: &Bake<'_>,
        identity: ImpostorIdentity,
    ) {
        self.baking.insert(identity);
        self.answers.ask_again(&identity);
        let emissive = emits(bake.material);
        let serial = cx
            .serials
            .take(identity, self.shape.facing(key.2), emissive);
        self.push(
            cx,
            AtlasWork::Bake {
                serial,
                mesh: key.0,
                material: key.1,
                parts: bake.parts.clone(),
                bounds: bake.bounds,
                hemisphere: key.2,
                emissive,
            },
        );
    }

    fn poll(&mut self, records: &Records, now: f64) {
        self.answers.settle(now);
        let answers = &mut self.answers;
        for supply in self.supplies.values_mut() {
            let Some(fetch) = supply.fetch.as_mut() else {
                continue;
            };
            let found = match fetch.poll(records) {
                Landed::Waiting => continue,
                Landed::Ready(bytes) => Some(Arc::clone(bytes)),
                Landed::Failed(_) => None,
            };
            supply.fetch = None;
            let identity = supply.identity;
            let Some(ticket) = answers.ticket(&identity) else {
                continue;
            };
            match found {
                Some(bytes) => answers.ready(&identity, ticket, bytes),
                None => answers.miss(&identity, ticket),
            };
        }
    }

    fn answer(&mut self, identity: ImpostorIdentity, made: Made) {
        self.baking.remove(&identity);
        let Some(ticket) = self.answers.ticket(&identity) else {
            return;
        };
        match made {
            Ok(bytes) => self.answers.ready(&identity, ticket, bytes),
            Err(error) => self.answers.fail(&identity, ticket, error),
        };
    }

    fn left(&self) -> u64 {
        let waiting = self
            .supplies
            .values()
            .filter(|supply| {
                !matches!(self.answers.get(&supply.identity), Some(Request::Failed(_)))
            })
            .count();
        (waiting + self.baking.len()) as u64 + u64::from(self.waiting)
    }

    fn unsettled(&self) -> u32 {
        let waiting = self
            .supplies
            .values()
            .filter(|supply| {
                matches!(
                    self.answers.get(&supply.identity),
                    Some(Request::Ready(_) | Request::Missing)
                )
            })
            .count();
        u32::try_from(waiting)
            .unwrap_or(u32::MAX)
            .saturating_add(self.waiting)
    }
}

type Made = Result<Arc<[u8]>, Error>;

fn bake_failed(got: impl Into<serde_json::Value>) -> Error {
    Error::new(ErrorCode::RenderFailed, "$.impostors.bake").got(got.into())
}

#[derive(Resource)]
pub(crate) struct ImpostorBooks {
    installed: bool,
    everything: Requests<(), ()>,
    book: Option<Book>,
    log: Feed<AtlasWork>,
    serials: Serials,
    missed: bool,
    read: Vec<(ImpostorIdentity, Result<Vec<u8>, Error>)>,
    assembling: Vec<(Background<Arc<[u8]>>, ImpostorIdentity)>,
    made: Vec<(ImpostorIdentity, Made)>,
    settling: Settling,
    settled: ImpostorRecords,
    layers: Vec<u32>,
}

impl Default for ImpostorBooks {
    fn default() -> Self {
        Self {
            installed: false,
            everything: Requests::default(),
            book: None,
            log: Feed::new(ImpostorSettings::default().work as usize),
            serials: Serials::default(),
            missed: false,
            read: Vec::new(),
            assembling: Vec::new(),
            made: Vec::new(),
            settling: Settling::default(),
            settled: ImpostorRecords::default(),
            layers: Vec::new(),
        }
    }
}

impl ImpostorBooks {
    pub(crate) fn log(&self) -> &Feed<AtlasWork> {
        &self.log
    }

    fn configure(&mut self) {
        self.log.replace();
        self.installed = true;
        self.book = None;
        self.serials = Serials::default();
        self.missed = false;
        self.read.clear();
        self.assembling.clear();
        self.made.clear();
    }

    fn arrived(&mut self, serial: u32, read: Option<Vec<u8>>) {
        if !self.installed {
            return;
        }
        let Some((identity, shape, emissive)) = self.serials.baking.remove(&serial) else {
            return;
        };
        let expected = readback_bytes(shape, ROW_ALIGNMENT, emissive) + 4;
        let input = match read {
            Some(bytes) if bytes.len() as u64 == expected => {
                Ok(assembly_input(shape, emissive, &bytes))
            }
            Some(bytes) => Err(bake_failed(bytes.len())),
            None => Err(bake_failed("lost")),
        };
        self.read.push((identity, input));
    }

    fn assemble(&mut self, jobs: &mut Jobs<'_, '_>, records: &Records) {
        let empty = BTreeSet::new();
        let baking = self.book.as_ref().map_or(&empty, |book| &book.baking);
        for (identity, read) in self.read.drain(..) {
            if !baking.contains(&identity) {
                continue;
            }
            match read {
                Ok(input) => {
                    let job = jobs.run(&ASSEMBLY, input, job_priority(IDLE_JOBS, 0), Ok);
                    self.assembling.push((job, identity));
                }
                Err(error) => self.made.push((identity, Err(error))),
            }
        }
        let made = &mut self.made;
        self.assembling.retain_mut(|(job, identity)| {
            let identity = *identity;
            if !baking.contains(&identity) {
                return false;
            }
            let output = match job.poll(records) {
                Landed::Waiting => return true,
                Landed::Ready(bytes) => Ok(Arc::clone(bytes)),
                Landed::Failed(error) => Err(error.clone()),
            };
            made.push((identity, output));
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
    (settings, records): (Res<ImpostorSettings>, Res<Records>),
    mut books: ResMut<ImpostorBooks>,
    mut reads: ResMut<BakeReads>,
    mut jobs: Jobs,
) {
    if std::mem::take(&mut reads.installed) {
        books.configure();
    }
    books.missed |= std::mem::take(&mut reads.missed);
    books.log.resize(settings.work as usize);
    for (serial, bytes) in std::mem::take(&mut reads.read) {
        books.arrived(serial, bytes);
    }
    books.assemble(&mut jobs, &records);
}

pub(crate) fn cook(world: &World, asks: &mut CookAsks<'_>) {
    let Some(books) = world.get_resource::<ImpostorBooks>() else {
        return;
    };
    asks.hold(books.everything.ask_later(()));
    asks.left(books.left());
}

gpu_struct! {
    #[derive(Resource)]
    pub struct ImpostorPiece {
        pub cutoff: [f32; 4],
    }
}

pub(crate) fn policy(settings: Res<ImpostorSettings>, mut piece: ResMut<ImpostorPiece>) {
    piece.set_if_neq(ImpostorPiece {
        cutoff: [settings.cutoff, 0.0, 0.0, 0.0],
    });
}

struct Frame {
    switched_on: bool,
    pixels: f32,
    max: Option<u32>,
    least: usize,
    detail: bool,
    cooking: bool,
    seconds: f64,
}

type Sources<'a> = (&'a AssetLibrary, &'a AssetResidency);

pub(crate) fn settle(
    (settings, switch, detail, limits): (
        Res<ImpostorSettings>,
        Switched<ImpostorSettings>,
        Res<Detail>,
        Res<DeviceLimits>,
    ),
    (time, records): (Res<FrameTime>, Res<Records>),
    (library, residency): (Res<AssetLibrary>, Res<AssetResidency>),
    (mut books, mut budget, mut stats): (
        ResMut<ImpostorBooks>,
        ResMut<FrameBudget>,
        ResMut<ImpostorStats>,
    ),
    mut jobs: Jobs,
    asks: Query<(Entity, &ImpostorAsks)>,
    mut held: Query<&mut ImpostorRecords>,
) {
    books.everything.settle(time.real_elapsed);
    if let Some(ticket) = books.everything.ticket(&()) {
        books.everything.ready(&(), ticket, ());
    }
    let cooking = books.everything.interests(&()) > 0;
    let frame = Frame {
        switched_on: switch.on(),
        pixels: settings.pixels,
        max: settings.max,
        least: settings.min_triangles as usize,
        detail: detail.chosen_error().is_some(),
        cooking,
        seconds: time.real_elapsed,
    };
    let books = &mut *books;
    settle_books(
        books,
        (&frame, &limits),
        (&mut jobs, &records, &mut budget),
        (&library, &residency),
        (&asks, &mut held),
    );
    let unsettled = books.book.as_ref().map_or(0, Book::unsettled);
    books.settling.left(unsettled);
    let next = ImpostorStats {
        failed: books.census().failed,
        ..stats.clone()
    };
    stats.set_if_neq(next);
}

fn settle_books(
    books: &mut ImpostorBooks,
    (frame, limits): (&Frame, &DeviceLimits),
    (jobs, records, budget): (&mut Jobs<'_, '_>, &Records, &mut FrameBudget),
    sources: Sources<'_>,
    (asks, written): (&Query<'_, '_, (Entity, &ImpostorAsks)>, &mut Written<'_, '_, '_>),
) {
    if !books.installed {
        return;
    }
    let device = AtlasDevice::of(limits);
    let made = std::mem::take(&mut books.made);
    let missed = std::mem::take(&mut books.missed);
    let ImpostorBooks {
        book: held,
        log,
        serials,
        settled,
        layers,
        ..
    } = books;
    let mut cx = Supplying {
        device,
        jobs,
        log,
        serials,
        budget,
        cooking: frame.cooking,
    };
    if missed && let Some(book) = held.as_mut() {
        let shape = book.shape;
        book.restart(&mut cx, shape);
    }
    for (identity, made) in made {
        if let Ok(bytes) = &made {
            cx.jobs.keep(&BAKE, &identity, bytes.to_vec());
        }
        if let Some(book) = held.as_mut() {
            book.answer(identity, made);
        }
    }
    if asks.is_empty() {
        if let Some(book) = held.as_mut() {
            book.supplies.clear();
            book.poll(records, frame.seconds);
        }
        return;
    }
    let created = held.is_none();
    let book = held.get_or_insert_with(|| Book::new(AtlasShape::of(frame.pixels), &device));
    if created {
        let shape = book.shape;
        book.push(&mut cx, AtlasWork::Reset { shape });
    }
    book.poll(records, frame.seconds);
    settle_book(
        book,
        (&mut cx, frame),
        sources,
        (asks, written),
        (settled, layers),
    );
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
) -> impl Fn(MaterialHandle) -> Option<&'a Material> + 'a {
    move |handle: MaterialHandle| {
        residency
            .material_resident(handle)
            .then(|| library.material(handle))
            .flatten()
    }
}

fn drawable((library, residency): Sources<'_>, handle: MaterialHandle) -> bool {
    residency.material_resident(handle)
        && library.material(handle).is_some_and(|material| {
            material
                .textures()
                .all(|texture| residency.texture_resident(texture))
        })
}

fn retain_book(book: &mut Book, cx: &mut Supplying<'_, '_, '_>, sources: Sources<'_>) {
    book.retain(cx, keep_key(sources.1), current_material(sources));
}

fn bake_of<'a>(
    (library, residency): Sources<'_>,
    ask: &ImpostorAsk,
    material: Option<&'a Material>,
) -> Option<Bake<'a>> {
    let mesh = library.mesh(ask.mesh)?;
    let mut content = (
        mesh.identity(content_hash),
        material_content(library, material),
    );
    let bounds = mesh.bounds();
    let (mut low, mut high) = (bounds.min(), bounds.max());
    let mut parts = Vec::with_capacity(ask.parts.len());
    for &(part_mesh, part_material) in &ask.parts {
        if !residency.resident(part_mesh) || !drawable((library, residency), part_material) {
            return None;
        }
        let (Some(finest), Some(value)) =
            (library.mesh(part_mesh), library.material(part_material))
        else {
            return None;
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
    (budget, least): (Option<u32>, usize),
) -> Option<Supplied> {
    if !sources.1.resident(ask.mesh) {
        return Some(Supplied::Waiting);
    }
    if sources
        .0
        .mesh(ask.mesh)
        .is_none_or(|mesh| mesh.indices().len() / 3 < least)
    {
        return None;
    }
    let key = (ask.mesh, ask.material, hemisphere);
    let supplied = supply_key(book, cx, sources, (ask, key), budget);
    if matches!(supplied, Supplied::Coming | Supplied::Waiting) {
        book.wanting.insert(key);
    }
    Some(supplied)
}

fn supply_key(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    sources: Sources<'_>,
    (ask, key): (&ImpostorAsk, ImpostorKey),
    budget: Option<u32>,
) -> Supplied {
    if ask
        .material
        .is_some_and(|handle| !drawable(sources, handle))
    {
        return Supplied::Waiting;
    }
    let material = ask.material.and_then(current_material(sources));
    let Some(bake) = bake_of(sources, ask, material) else {
        return Supplied::Waiting;
    };
    book.supply(cx, key, &bake, budget)
}

fn supply_all(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    sources: Sources<'_>,
    (asked, enabled, limits): (&ImpostorAsks, bool, (Option<u32>, usize)),
    settled: &mut ImpostorRecords,
) {
    for ask in &asked.asks {
        let supplied = match ask {
            Some(ask) if enabled => supply_ask(book, cx, sources, (ask, asked.hemisphere), limits),
            Some(_) | None => None,
        };
        let record = match supplied {
            Some(Supplied::Waiting) => {
                book.waiting = book.waiting.saturating_add(1);
                ImpostorRecord::default()
            }
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

fn live_layers(records: &[ImpostorRecord], layers: &mut Vec<u32>) -> u32 {
    layers.clear();
    layers.extend(
        records
            .iter()
            .filter(|record| record.framed())
            .map(|record| record.layer),
    );
    layers.sort_unstable();
    layers.dedup();
    layers.len() as u32
}

type Written<'w, 's, 'd> = Query<'w, 's, &'d mut ImpostorRecords>;

fn write_records(written: &mut Written<'_, '_, '_>, entity: Entity, settled: &ImpostorRecords) {
    let Ok(mut current) = written.get_mut(entity) else {
        return;
    };
    if *current == *settled {
        return;
    }
    let current = &mut *current;
    current.generation = settled.generation;
    current.records.clone_from(&settled.records);
    current.asked = settled.asked;
    current.refused = settled.refused;
    current.live = settled.live;
}

fn settle_book(
    book: &mut Book,
    (cx, frame): (&mut Supplying<'_, '_, '_>, &Frame),
    sources: Sources<'_>,
    (asked, written): (&Query<'_, '_, (Entity, &ImpostorAsks)>, &mut Written<'_, '_, '_>),
    (settled, layers): (&mut ImpostorRecords, &mut Vec<u32>),
) {
    book.fit(cx, AtlasShape::of(frame.pixels));
    book.waiting = 0;
    std::mem::swap(&mut book.wanted, &mut book.wanting);
    book.wanting.clear();
    retain_book(book, cx, sources);
    let shown = frame.detail && frame.switched_on;
    if !shown {
        book.release(cx);
    } else {
        book.enforce_budget(frame.max, cx.cooking);
    }
    for (entity, asks) in asked {
        settled.generation = asks.generation;
        settled.records.clear();
        settled.asked = 0;
        settled.refused = 0;
        supply_all(
            book,
            cx,
            sources,
            (asks, shown, (frame.max, frame.least)),
            settled,
        );
        book.drawn |= book.capacity > 0 && !settled.records.is_empty();
        settled.live = live_layers(&settled.records, layers);
        write_records(written, entity, settled);
    }
}
