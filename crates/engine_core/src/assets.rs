use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::change_detection::{DetectChanges, Mut, Ref};
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::query::QueryState;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Query, Res, SystemState};
use bevy_ecs::world::World;
use engine_assets::{
    AlphaMode, AssetLibrary, Filter, Material, MaterialHandle, Mesh, MeshHandle, MeshLods, Texture,
    TextureFormat, TextureHandle, TextureRef, Wrap,
};
pub use engine_textures::solid_block;
use glam::Vec3;

use crate::collections::{HashMap, HashSet};
use crate::{
    CookMode, Detail, Error, ErrorCode, Job, JobCx, JobKind, Jobs, PresentationState, Records,
    Slot, Source, Ticket, VIEW_JOBS, content_hash, job_priority,
};

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct AssetUses {
    pub meshes: Vec<MeshHandle>,
    pub materials: Vec<MaterialHandle>,
    pub textures: Vec<TextureHandle>,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshStreams {
    Plain = 0,
    Extended = 1,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshPlacement {
    pub first_index: u32,
    pub index_count: u32,
    pub streams: MeshStreams,
    pub base_vertex: u32,
    pub vertex_count: u32,
}

#[derive(Clone, Debug, Default)]
struct Space {
    capacity: u32,
    free: Vec<(u32, u32)>,
}

impl Space {
    fn allocate(&mut self, length: u32) -> Option<u32> {
        let slot = self.free.iter().position(|&(_, free)| free >= length)?;
        let (start, free) = self.free[slot];
        if free == length {
            self.free.remove(slot);
        } else {
            self.free[slot] = (start + length, free - length);
        }
        Some(start)
    }

    fn release(&mut self, start: u32, length: u32) {
        if length == 0 {
            return;
        }
        let slot = self.free.partition_point(|&(free, _)| free < start);
        self.free.insert(slot, (start, length));
        if slot + 1 < self.free.len() {
            let (next_start, next_length) = self.free[slot + 1];
            if start + length == next_start {
                self.free[slot].1 += next_length;
                self.free.remove(slot + 1);
            }
        }
        if slot > 0 {
            let (previous_start, previous_length) = self.free[slot - 1];
            if previous_start + previous_length == start {
                self.free[slot - 1].1 += self.free[slot].1;
                self.free.remove(slot);
            }
        }
    }

    fn grow(&mut self, capacity: u32) {
        let old = self.capacity;
        self.capacity = capacity;
        self.release(old, capacity - old);
    }
}

fn grown_capacity(space: &Space, length: u32, bytes: u64, limit: u64) -> Option<u32> {
    let needed = u64::from(space.capacity) + u64::from(length);
    let grown = (u64::from(space.capacity) * 2).max(needed).max(1024);
    let fits = (limit / bytes).min(u64::from(u32::MAX));
    let capacity = grown.min(fits);
    (capacity >= needed).then_some(capacity as u32)
}

const INDEX_BYTES: u64 = 4;

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaShape {
    pub vertex: u64,
    pub extra: u64,
    pub limit: u64,
}

#[derive(Clone, Debug)]
struct Arena {
    shape: ArenaShape,
    classes: [Space; 2],
    indices: Space,
}

impl Arena {
    fn new(shape: ArenaShape) -> Self {
        Self {
            shape,
            classes: [Space::default(), Space::default()],
            indices: Space::default(),
        }
    }

    fn widest(&self, streams: MeshStreams) -> u64 {
        match streams {
            MeshStreams::Plain => self.shape.vertex,
            MeshStreams::Extended => self.shape.vertex.max(self.shape.extra),
        }
    }

    fn insert(
        &mut self,
        streams: MeshStreams,
        vertex_count: u32,
        index_count: u32,
    ) -> Option<MeshPlacement> {
        let limit = self.shape.limit;
        let first_index = match self.indices.allocate(index_count) {
            Some(start) => start,
            None => {
                let grown = grown_capacity(&self.indices, index_count, INDEX_BYTES, limit)?;
                self.indices.grow(grown);
                self.indices.allocate(index_count)?
            }
        };
        let widest = self.widest(streams);
        let space = &mut self.classes[streams as usize];
        let base_vertex = match space.allocate(vertex_count) {
            Some(start) => Some(start),
            None => grown_capacity(space, vertex_count, widest, limit).and_then(|grown| {
                space.grow(grown);
                space.allocate(vertex_count)
            }),
        };
        let Some(base_vertex) = base_vertex else {
            self.indices.release(first_index, index_count);
            return None;
        };
        Some(MeshPlacement {
            first_index,
            index_count,
            streams,
            base_vertex,
            vertex_count,
        })
    }

    fn remove(&mut self, placement: MeshPlacement) {
        self.indices
            .release(placement.first_index, placement.index_count);
        self.classes[placement.streams as usize]
            .release(placement.base_vertex, placement.vertex_count);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidentMesh {
    placement: Option<MeshPlacement>,
    extended: bool,
    index_count: u32,
    byte_len: u64,
    lods: Vec<(u32, u32, f32)>,
    plan: Vec<(u32, u32, f32)>,
    casters: Vec<(u32, u32)>,
    without_levels: bool,
}

impl ResidentMesh {
    pub fn first_index(&self) -> u32 {
        self.placement.map_or(0, |placement| placement.first_index)
    }

    pub fn base_vertex(&self) -> i32 {
        self.placement
            .map_or(0, |placement| placement.base_vertex as i32)
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    pub fn extended(&self) -> bool {
        self.extended
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub fn lods(&self) -> &[(u32, u32, f32)] {
        &self.lods
    }

    pub fn plan(&self) -> &[(u32, u32, f32)] {
        &self.plan
    }

    pub fn casters(&self) -> &[(u32, u32)] {
        &self.casters
    }

    pub fn caster_range(&self, level: usize, own: (u32, u32)) -> (u32, u32) {
        self.casters.get(level).copied().unwrap_or(own)
    }

    pub fn without_levels(&self) -> bool {
        self.without_levels
    }

    #[doc(hidden)]
    pub fn placement(&self) -> Option<MeshPlacement> {
        self.placement
    }
}

fn plan_levels(lods: &[(u32, u32, f32)]) -> Vec<(u32, u32, f32)> {
    let first = lods.first().copied().filter(|&(_, count, _)| count > 0);
    let empty = lods.last().copied().filter(|&(_, count, _)| count == 0);
    first.into_iter().chain(empty).collect()
}

fn resident_mesh(mesh: &Mesh, lods: Option<&MeshLods>, arena: &mut Arena) -> ResidentMesh {
    let extended = mesh.colors().is_some() || mesh.uv1().is_some();
    let streams = if extended {
        MeshStreams::Extended
    } else {
        MeshStreams::Plain
    };
    let total = mesh.indices().len() + lods.map_or(0, |lods| lods.indices.len());
    let placement = u32::try_from(total)
        .ok()
        .and_then(|count| arena.insert(streams, mesh.vertices().len() as u32, count));
    let (levels, casters) = match lods {
        Some(lods) => (
            lods.levels
                .iter()
                .map(|level| (level.first_index, level.index_count, level.error))
                .collect(),
            lods.casters.clone(),
        ),
        None => (Vec::new(), Vec::new()),
    };
    ResidentMesh {
        placement,
        extended,
        index_count: mesh.indices().len() as u32,
        byte_len: mesh.byte_len() as u64,
        plan: plan_levels(&levels),
        lods: levels,
        casters,
        without_levels: lods.is_none(),
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Named {
    meshes: Vec<MeshHandle>,
    materials: BTreeSet<MaterialHandle>,
    textures: BTreeSet<TextureHandle>,
    library: ((u64, u64), u64),
}

#[derive(Debug, Default)]
struct Table {
    meshes: BTreeMap<MeshHandle, ResidentMesh>,
    materials: BTreeMap<MaterialHandle, u64>,
    textures: BTreeSet<TextureHandle>,
    arena: Option<Arena>,
    generation: u64,
    materials_generation: u64,
    layout: u64,
    named: Named,
}

impl Table {
    fn settle(&mut self, named: Named, library: &AssetLibrary) {
        let Some(arena) = self.arena.as_mut() else {
            return;
        };
        let settled = |handle: MeshHandle| {
            library.mesh(handle).is_some() && library.mesh_lods(handle).is_some()
        };
        let wanted: BTreeSet<MeshHandle> = named
            .meshes
            .iter()
            .copied()
            .filter(|&handle| settled(handle))
            .collect();
        let resident = self.meshes.len();
        self.meshes.retain(|handle, mesh| {
            let keep = wanted.contains(handle);
            if !keep && let Some(placement) = mesh.placement {
                arena.remove(placement);
            }
            keep
        });
        let mut changed = self.meshes.len() != resident;
        if changed {
            self.layout += 1;
        }
        for &handle in &named.meshes {
            if !wanted.contains(&handle) || self.meshes.contains_key(&handle) {
                continue;
            }
            let (Some(mesh), Some(lods)) = (library.mesh(handle), library.mesh_lods(handle)) else {
                continue;
            };
            let held = resident_mesh(mesh, lods.map(|lods| &**lods), arena);
            self.meshes.insert(handle, held);
            self.layout += 1;
            changed = true;
        }
        if changed {
            self.generation = self.generation.wrapping_add(1);
        }
        let materials: BTreeMap<MaterialHandle, u64> = named
            .materials
            .iter()
            .filter_map(|&handle| {
                let material = library.material(handle)?;
                material
                    .textures()
                    .all(|texture| library.texture(texture).is_some())
                    .then(|| (handle, library.material_version(handle).unwrap_or(0)))
            })
            .collect();
        let mut textures: BTreeSet<TextureHandle> = named
            .textures
            .iter()
            .copied()
            .filter(|&texture| library.texture(texture).is_some())
            .collect();
        for &handle in materials.keys() {
            if let Some(material) = library.material(handle) {
                textures.extend(material.textures());
            }
        }
        if materials != self.materials || textures != self.textures {
            self.materials = materials;
            self.textures = textures;
            self.materials_generation = self.materials_generation.wrapping_add(1);
        }
        self.named = named;
    }
}

#[derive(Resource)]
#[component(immutable)]
pub struct AssetResidency {
    table: Table,
}

impl AssetResidency {
    pub(crate) fn empty() -> Self {
        Self {
            table: Table::default(),
        }
    }

    pub fn resident(&self, mesh: MeshHandle) -> bool {
        self.table.meshes.contains_key(&mesh)
    }

    pub fn generation(&self) -> u64 {
        self.table.generation
    }

    pub fn mesh(&self, mesh: MeshHandle) -> Option<&ResidentMesh> {
        self.table.meshes.get(&mesh)
    }

    pub fn material_resident(&self, material: MaterialHandle) -> bool {
        self.table.materials.contains_key(&material)
    }

    pub fn texture_resident(&self, texture: TextureHandle) -> bool {
        self.table.textures.contains(&texture)
    }

    #[doc(hidden)]
    pub fn meshes(&self) -> impl Iterator<Item = (MeshHandle, &ResidentMesh)> + '_ {
        self.table
            .meshes
            .iter()
            .map(|(&handle, mesh)| (handle, mesh))
    }

    #[doc(hidden)]
    pub fn materials(&self) -> impl Iterator<Item = (MaterialHandle, u64)> + '_ {
        self.table
            .materials
            .iter()
            .map(|(&handle, &version)| (handle, version))
    }

    #[doc(hidden)]
    pub fn textures(&self) -> impl Iterator<Item = TextureHandle> + '_ {
        self.table.textures.iter().copied()
    }

    #[doc(hidden)]
    pub fn materials_generation(&self) -> u64 {
        self.table.materials_generation
    }

    #[doc(hidden)]
    pub fn layout(&self) -> u64 {
        self.table.layout
    }

    #[doc(hidden)]
    pub fn capacities(&self) -> Option<([u32; 2], u32)> {
        self.table.arena.as_ref().map(|arena| {
            (
                [arena.classes[0].capacity, arena.classes[1].capacity],
                arena.indices.capacity,
            )
        })
    }
}

#[derive(Resource)]
struct AssetScratch {
    uses: QueryState<&'static AssetUses>,
}

fn names(world: &mut World) -> Named {
    if !world.contains_resource::<AssetScratch>() {
        let uses = world.query::<&'static AssetUses>();
        world.insert_resource(AssetScratch { uses });
    }
    world.resource_scope(|world, mut scratch: Mut<AssetScratch>| {
        let mut named = Named::default();
        let mut seen = BTreeSet::new();
        for uses in scratch.uses.iter(world) {
            for &mesh in &uses.meshes {
                if seen.insert(mesh) {
                    named.meshes.push(mesh);
                }
            }
            named.materials.extend(uses.materials.iter().copied());
            named.textures.extend(uses.textures.iter().copied());
        }
        let library = world.resource::<AssetLibrary>();
        named.library = (library.revision(), library.material_revision());
        named
    })
}

fn settle_residency(world: &mut World) {
    if world
        .get_resource::<AssetResidency>()
        .is_none_or(|residency| residency.table.arena.is_none())
    {
        return;
    }
    let named = names(world);
    if world.resource::<AssetResidency>().table.named == named {
        return;
    }
    let Some(mut table) = world
        .modify_resource(|residency: &mut AssetResidency| std::mem::take(&mut residency.table))
        .ok()
        .flatten()
    else {
        return;
    };
    table.settle(named, world.resource::<AssetLibrary>());
    let _ = world.modify_resource(|residency: &mut AssetResidency| residency.table = table);
}

pub const IMPOSTOR_MIN_TRIANGLES: usize = 64;

pub mod impostor_flags {
    crate::gpu_consts! {
        pub const IMPOSTOR_HEMISPHERE: u32 = 0x10000;
        pub const IMPOSTOR_MASKED: u32 = 0x20000;
        pub const IMPOSTOR_FRAMES_SHIFT: u32 = 19;
        pub const IMPOSTOR_FOLIAGE: u32 = 0x1000000;
        pub const IMPOSTOR_KINDS: u32 = 4;
    }
}

pub use impostor_flags::IMPOSTOR_KINDS;

use impostor_flags::{
    IMPOSTOR_FOLIAGE, IMPOSTOR_FRAMES_SHIFT, IMPOSTOR_HEMISPHERE, IMPOSTOR_MASKED,
};

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
        (emissive, masked, foliage): ([f32; 3], bool, bool),
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
            flags: shape.flags()
                | if masked { IMPOSTOR_MASKED } else { 0 }
                | if foliage { IMPOSTOR_FOLIAGE } else { 0 },
        }
    }

    pub fn error(&self) -> f32 {
        let texels = self.flags & (IMPOSTOR_HEMISPHERE - 1);
        if texels == 0 {
            return 0.0;
        }
        2.0 * self.sphere[3] / texels as f32
    }

    pub fn framed(&self) -> bool {
        self.flags & (IMPOSTOR_HEMISPHERE - 1) != 0 && self.sphere[3] > 0.0
    }
}

const RGBA8_SIZE: u32 = 512;

const LOAD_BUDGET: usize = 16 << 20;

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
    pub const SPHERE: Self = Self {
        frames: 8,
        texels: 16,
        hemisphere: false,
    };

    pub fn size(self) -> u32 {
        self.frames * self.texels
    }

    pub fn levels(self) -> u32 {
        self.texels.trailing_zeros()
    }

    fn flags(self) -> u32 {
        self.texels
            | if self.hemisphere {
                IMPOSTOR_HEMISPHERE
            } else {
                0
            }
            | (self.frames << IMPOSTOR_FRAMES_SHIFT)
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

#[doc(hidden)]
pub fn impostor_record_header(shape: AtlasShape, format: AtlasFormat) -> Vec<u8> {
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

static RENDER: Source = Source::generated("engine_render", crate::sources::RENDER, &[]);

#[doc(hidden)]
pub static IMPOSTOR: JobKind = JobKind::new("engine::impostor", &RENDER, made_on_the_gpu);

fn made_on_the_gpu(_: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs.engine::impostor").got("engine::impostor"))
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
            enabled: true,
            pixels: 128.0,
            max: None,
        }
    }
}

impl Slot for ImpostorPolicy {
    const NAME: &'static str = "impostor_policy";
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtlasRole {
    Scene,
    Placed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImpostorAsk {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub parts: Vec<(MeshHandle, MaterialHandle)>,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct ImpostorAsks {
    pub role: AtlasRole,
    pub shape: AtlasShape,
    pub asks: Vec<Option<ImpostorAsk>>,
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct ImpostorRecords {
    pub records: Vec<ImpostorRecord>,
    pub asked: u32,
    pub refused: u32,
    pub live: u32,
}

#[derive(Default)]
struct Contents {
    meshes: HashMap<MeshHandle, u128>,
    textures: HashMap<TextureHandle, u128>,
}

impl Contents {
    fn mesh(&mut self, handle: MeshHandle, mesh: &Mesh) -> u128 {
        *self.meshes.entry(handle).or_insert_with(|| {
            let mut bytes = Vec::with_capacity(mesh.byte_len() + 16);
            words(
                &mut bytes,
                &[mesh.vertices().len() as u32, mesh.indices().len() as u32],
            );
            for vertex in mesh.vertices() {
                floats(&mut bytes, &vertex.position);
                floats(&mut bytes, &vertex.normal);
                floats(&mut bytes, &vertex.uv);
            }
            words(&mut bytes, mesh.indices());
            bytes.push(u8::from(mesh.colors().is_some()));
            for color in mesh.colors().unwrap_or_default() {
                floats(&mut bytes, color);
            }
            bytes.push(u8::from(mesh.uv1().is_some()));
            for uv in mesh.uv1().unwrap_or_default() {
                floats(&mut bytes, uv);
            }
            content_hash(&bytes)
        })
    }

    fn material<'a>(
        &mut self,
        material: Option<&Material>,
        texture: impl Fn(TextureHandle) -> Option<&'a Texture>,
    ) -> u128 {
        let Some(material) = material else {
            return 0;
        };
        let mut bytes = Vec::with_capacity(256);
        floats(&mut bytes, &material.color());
        for map in material.maps() {
            let Some(map) = map else {
                bytes.push(0);
                continue;
            };
            bytes.push(1);
            bytes.extend_from_slice(&self.texture(map.texture, &texture).to_le_bytes());
            texture_ref(&mut bytes, &map);
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
        match material.alpha_mode() {
            AlphaMode::Opaque => bytes.push(0),
            AlphaMode::Mask { cutoff } => {
                bytes.push(1);
                floats(&mut bytes, &[cutoff]);
            }
        }
        bytes.push(u8::from(material.is_double_sided()));
        floats(&mut bytes, &[material.world_tiling().unwrap_or(-1.0)]);
        match material.top_layer() {
            Some(top) => {
                bytes.push(1);
                floats(&mut bytes, &top.color);
                floats(&mut bytes, &[top.coverage, top.roughness]);
            }
            None => bytes.push(0),
        }
        match material.grounding() {
            Some(grounding) => {
                bytes.push(1);
                floats(&mut bytes, &[grounding.height]);
            }
            None => bytes.push(0),
        }
        content_hash(&bytes)
    }

    fn texture<'a>(
        &mut self,
        handle: TextureHandle,
        texture: &impl Fn(TextureHandle) -> Option<&'a Texture>,
    ) -> u128 {
        if let Some(&content) = self.textures.get(&handle) {
            return content;
        }
        let Some(found) = texture(handle) else {
            return 0;
        };
        let mut bytes = Vec::with_capacity(found.byte_len() + 12);
        let format = match found.format() {
            TextureFormat::Rgba8 => 0,
            TextureFormat::Bc7 => 1,
            TextureFormat::Etc2Rgba8 => 2,
            TextureFormat::Astc4x4 => 3,
        };
        words(&mut bytes, &[found.width(), found.height(), format]);
        for level in found.levels() {
            bytes.extend_from_slice(level);
        }
        let content = content_hash(&bytes);
        self.textures.insert(handle, content);
        content
    }

    fn retain(
        &mut self,
        mut mesh: impl FnMut(MeshHandle) -> bool,
        mut texture: impl FnMut(TextureHandle) -> bool,
    ) {
        self.meshes.retain(|&handle, _| mesh(handle));
        self.textures.retain(|&handle, _| texture(handle));
    }
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

fn shades_as_foliage(material: Option<&Material>) -> bool {
    material.is_some_and(|material| {
        matches!(material.alpha_mode(), AlphaMode::Mask { .. }) || material.is_foliage()
    })
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

type ImpostorKey = (MeshHandle, Option<MaterialHandle>);

struct Baked {
    record: ImpostorRecord,
    material: Option<Material>,
}

struct Supply {
    identity: ImpostorIdentity,
    material: Option<Material>,
    ticket: Option<Ticket>,
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
    foliage: bool,
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
        bytes: Arc<[u8]>,
    },
    Bake {
        identity: ImpostorIdentity,
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
        parts: Vec<(MeshHandle, MaterialHandle)>,
        bounds: (Vec3, Vec3),
    },
    DropTargets,
}

type SettleKey = (u64, u64, u64, ImpostorPolicyKey, bool, bool, u64, u32);

type ImpostorPolicyKey = (bool, u32, Option<u32>);

struct Book {
    shape: AtlasShape,
    layers: Layers,
    capacity: u32,
    targets: bool,
    drawn: bool,
    baked: HashMap<ImpostorKey, Baked>,
    supplies: HashMap<ImpostorKey, Supply>,
    supplied: HashSet<ImpostorKey>,
    answers: HashMap<ImpostorIdentity, Option<Arc<[u8]>>>,
    baking: BTreeSet<ImpostorIdentity>,
    loaded: usize,
    deferred: bool,
    generation: u64,
    contents: Contents,
    key: Option<SettleKey>,
    records: ImpostorRecords,
    work: Vec<(u64, AtlasWork)>,
}

struct Supplying<'a, 'w, 's> {
    device: AtlasDevice,
    jobs: &'a mut Jobs<'w, 's>,
    next: &'a mut u64,
    cooking: bool,
}

impl Supplying<'_, '_, '_> {
    fn cancel(&mut self, supply: Supply) {
        if let Some(ticket) = supply.ticket {
            self.jobs.cancel(ticket);
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
            answers: HashMap::default(),
            baking: BTreeSet::new(),
            loaded: 0,
            deferred: false,
            generation: 0,
            contents: Contents::default(),
            key: None,
            records: ImpostorRecords::default(),
            work: Vec::new(),
        }
    }

    fn push(&mut self, cx: &mut Supplying<'_, '_, '_>, work: AtlasWork) {
        *cx.next += 1;
        self.work.push((*cx.next, work));
    }

    fn drop_supplies(&mut self, cx: &mut Supplying<'_, '_, '_>) {
        for (_, supply) in self.supplies.drain() {
            cx.cancel(supply);
        }
    }

    fn fit(&mut self, cx: &mut Supplying<'_, '_, '_>, shape: AtlasShape) {
        let shape = shape.held_on(&cx.device);
        if shape == self.shape {
            return;
        }
        self.shape = shape;
        self.capacity = 0;
        self.targets = false;
        self.drawn = false;
        self.layers = Layers::default();
        self.baked.clear();
        self.drop_supplies(cx);
        self.supplied.clear();
        self.answers.clear();
        self.baking.clear();
        self.generation += 1;
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
        self.drop_supplies(cx);
        self.supplied.clear();
        self.answers.clear();
        self.baking.clear();
        self.generation += 1;
        self.push(cx, AtlasWork::Release);
    }

    fn retain(
        &mut self,
        cx: &mut Supplying<'_, '_, '_>,
        keep: impl Fn(&ImpostorKey) -> bool,
        material: impl Fn(MaterialHandle) -> Option<Material>,
    ) {
        let layers = &mut self.layers;
        let before = self.baked.len();
        self.baked.retain(|key, baked| {
            let current = key.1.and_then(&material);
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
        let dropped: Vec<ImpostorKey> = self
            .supplies
            .keys()
            .filter(|&key| !keep(key))
            .copied()
            .collect();
        for key in dropped {
            if let Some(supply) = self.supplies.remove(&key) {
                cx.cancel(supply);
            }
        }
        let kept = self.supplies.len();
        let unsupplied: Vec<ImpostorKey> = self
            .supplies
            .keys()
            .filter(|&key| !supplied.contains(key))
            .copied()
            .collect();
        for key in unsupplied {
            if let Some(supply) = self.supplies.remove(&key) {
                cx.cancel(supply);
            }
        }
        if cx.cooking && self.supplies.len() < kept {
            let record = serde_json::json!({
                "event": "impostor_supplies_dropped",
                "dropped": kept - self.supplies.len(),
                "asked": asked,
            });
            eprintln!("{record}");
        }
        let supplies = &self.supplies;
        self.answers
            .retain(|identity, _| supplies.values().any(|supply| supply.identity == *identity));
    }

    fn enforce_budget(&mut self, budget: Option<u32>, cooking: bool) {
        if !cooking && budget.is_some_and(|budget| self.layers.live() > budget) {
            self.baked.clear();
            self.layers = Layers::default();
            self.generation += 1;
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
            if let Some(supply) = self.supplies.remove(&key) {
                cx.cancel(supply);
            }
            return Supplied::Refused;
        }
        let asked = self
            .supplies
            .get(&key)
            .filter(|supply| supply.material.as_ref() == bake.material.as_ref())
            .map(|supply| supply.identity);
        let Some(identity) = asked else {
            let identity = identity(self.shape, cx.device.format, bake.content);
            let ticket = cx
                .jobs
                .fetch(&IMPOSTOR, &identity, job_priority(VIEW_JOBS, 0));
            let supply = Supply {
                identity,
                material: bake.material,
                ticket: Some(ticket),
            };
            if let Some(replaced) = self.supplies.insert(key, supply) {
                cx.cancel(replaced);
            }
            return Supplied::Coming;
        };
        if self.loaded >= LOAD_BUDGET && self.answers.get(&identity).is_some_and(Option::is_some) {
            self.deferred = true;
            return Supplied::Coming;
        }
        let Some(answer) = self.answers.remove(&identity) else {
            return Supplied::Coming;
        };
        if answer.is_some()
            && self
                .supplies
                .iter()
                .any(|(other, supply)| *other != key && supply.identity == identity)
        {
            self.answers.insert(identity, answer.clone());
        }
        if let Some(bytes) = answer
            && self.load(cx, key, bake, &bytes, budget)
        {
            if let Some(supply) = self.supplies.remove(&key) {
                cx.cancel(supply);
            }
            return self.supplied(key, bake);
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
        let header = impostor_record_header(self.shape, cx.device.format);
        if bytes.len() != record_bytes(self.shape, cx.device.format)
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
                bytes: Arc::clone(bytes),
            },
        );
        self.loaded += bytes.len();
        self.resident(key, bake, layer);
        true
    }

    fn resident(&mut self, key: ImpostorKey, bake: &Bake, layer: u32) {
        let emissive = bake.material.as_ref().map_or([0.0; 3], Material::emissive);
        let record = ImpostorRecord::new(
            bake.bounds.0,
            bake.bounds.1,
            layer,
            (emissive, bake.masked, bake.foliage),
            self.shape,
        );
        self.baked.insert(
            key,
            Baked {
                record,
                material: bake.material,
            },
        );
        self.generation += 1;
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
        self.push(
            cx,
            AtlasWork::Bake {
                identity,
                mesh: key.0,
                material: key.1,
                parts: bake.parts.clone(),
                bounds: bake.bounds,
            },
        );
    }

    fn release_targets(&mut self, cx: &mut Supplying<'_, '_, '_>) {
        if std::mem::take(&mut self.targets) {
            self.push(cx, AtlasWork::DropTargets);
        }
    }

    fn poll(&mut self, cx: &mut Supplying<'_, '_, '_>, records: &Records) {
        let mut answered = Vec::new();
        for (key, supply) in &self.supplies {
            let Some(ticket) = supply.ticket.as_ref() else {
                continue;
            };
            let Some(found) = records.get(ticket) else {
                continue;
            };
            answered.push((*key, supply.identity, found.ok()));
        }
        for (key, identity, bytes) in answered {
            if let Some(ticket) = self
                .supplies
                .get_mut(&key)
                .and_then(|supply| supply.ticket.take())
            {
                cx.jobs.cancel(ticket);
            }
            self.answers.insert(identity, bytes);
            self.generation += 1;
        }
    }

    fn answer(&mut self, identity: ImpostorIdentity, bytes: Option<Arc<[u8]>>) {
        self.baking.remove(&identity);
        self.answers.insert(identity, bytes);
        self.generation += 1;
    }

    fn left(&self) -> u64 {
        (self.supplies.len() + self.baking.len()) as u64
    }
}

#[doc(hidden)]
#[derive(Resource, Default)]
pub struct ImpostorBooks {
    device: Option<AtlasDevice>,
    books: BTreeMap<AtlasRole, Book>,
    next: u64,
    acked: u64,
    made: Vec<(AtlasRole, ImpostorIdentity, Option<Vec<u8>>)>,
}

impl ImpostorBooks {
    #[doc(hidden)]
    pub fn device(&self) -> Option<AtlasDevice> {
        self.device
    }

    #[doc(hidden)]
    pub fn shape(&self, role: AtlasRole) -> Option<AtlasShape> {
        self.books.get(&role).map(|book| book.shape)
    }

    #[doc(hidden)]
    pub fn records(&self, role: AtlasRole) -> &[ImpostorRecord] {
        self.books
            .get(&role)
            .map(|book| book.records.records.as_slice())
            .unwrap_or_default()
    }

    #[doc(hidden)]
    pub fn live(&self, role: AtlasRole) -> u32 {
        self.books.get(&role).map_or(0, |book| book.layers.live())
    }

    #[doc(hidden)]
    pub fn contains(
        &self,
        role: AtlasRole,
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
    ) -> bool {
        self.books
            .get(&role)
            .is_some_and(|book| book.baked.contains_key(&(mesh, material)))
    }

    #[doc(hidden)]
    pub fn work(&self) -> impl Iterator<Item = (u64, AtlasRole, &AtlasWork)> + '_ {
        let mut work: Vec<(u64, AtlasRole, &AtlasWork)> = self
            .books
            .iter()
            .flat_map(|(&role, book)| book.work.iter().map(move |(at, work)| (*at, role, work)))
            .collect();
        work.sort_by_key(|&(at, _, _)| at);
        work.into_iter()
    }

    #[doc(hidden)]
    pub fn left(&self) -> u64 {
        self.books.values().map(Book::left).sum()
    }
}

#[doc(hidden)]
pub fn deliver_impostors(
    world: &mut World,
    done: u64,
    made: Vec<(AtlasRole, ImpostorIdentity, Option<Vec<u8>>)>,
) {
    if let Some(mut books) = world.get_resource_mut::<ImpostorBooks>() {
        books.acked = books.acked.max(done);
        books.made.extend(made);
    }
}

pub(crate) fn install(world: &mut World) {
    world.insert_resource(AssetResidency::empty());
    world.init_resource::<ImpostorBooks>();
}

#[doc(hidden)]
pub fn configure_assets(world: &mut World, arena: ArenaShape, atlas: AtlasDevice) {
    let _ = world.modify_resource(|residency: &mut AssetResidency| {
        let table = &mut residency.table;
        table.meshes.clear();
        table.materials.clear();
        table.textures.clear();
        table.arena = Some(Arena::new(arena));
        table.named = Named::default();
        table.generation = table.generation.wrapping_add(1);
        table.materials_generation = table.materials_generation.wrapping_add(1);
        table.layout += 1;
    });
    let next = world
        .get_resource::<ImpostorBooks>()
        .map_or(0, |books| books.next);
    world.insert_resource(ImpostorBooks {
        device: Some(atlas),
        books: BTreeMap::new(),
        next,
        acked: next,
        made: Vec::new(),
    });
}

#[doc(hidden)]
pub fn assets_pending(world: &World) -> u64 {
    world
        .get_resource::<ImpostorBooks>()
        .map_or(0, ImpostorBooks::left)
}

#[doc(hidden)]
pub fn settle_assets(world: &mut World) {
    {
        let _profile = engine_profile::span("render", "validate_and_retire");
        settle_residency(world);
    }
    settle_impostors(world);
}

type Inputs = (
    Jobs<'static, 'static>,
    Res<'static, Records>,
    Res<'static, AssetLibrary>,
    Res<'static, AssetResidency>,
    Query<'static, 'static, (Entity, Ref<'static, ImpostorAsks>)>,
);

#[derive(Resource)]
struct ImpostorInputs(SystemState<Inputs>);

struct Frame {
    policy: ImpostorPolicy,
    detail: bool,
    cooking: bool,
    gpu: bool,
}

type Sources<'a> = (&'a AssetLibrary, &'a AssetResidency);

fn settle_impostors(world: &mut World) {
    if world
        .get_resource::<ImpostorBooks>()
        .is_none_or(|books| books.device.is_none())
    {
        return;
    }
    let frame = Frame {
        policy: world
            .get_resource::<ImpostorPolicy>()
            .copied()
            .unwrap_or_default(),
        detail: world.resource::<Detail>().enabled,
        cooking: world.resource::<CookMode>().cooking,
        gpu: world.resource::<PresentationState>().gpu_pending(),
    };
    if !world.contains_resource::<ImpostorInputs>() {
        let state = SystemState::<Inputs>::new(world);
        world.insert_resource(ImpostorInputs(state));
    }
    let written = world.resource_scope(|world, mut inputs: Mut<ImpostorInputs>| {
        let written = world.resource_scope(|world, mut books: Mut<ImpostorBooks>| {
            let Ok((mut jobs, records, library, residency, asks)) = inputs.0.get_mut(world) else {
                return Default::default();
            };
            settle_books(
                &mut books,
                &frame,
                (&mut jobs, &*records),
                (&*library, &*residency),
                &asks,
            )
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

fn settle_key(
    book: &Book,
    frame: &Frame,
    (library, residency): Sources<'_>,
    scene_live: u32,
) -> SettleKey {
    (
        residency.generation(),
        residency.materials_generation(),
        book.generation,
        (
            frame.policy.enabled,
            frame.policy.pixels.to_bits(),
            frame.policy.max,
        ),
        frame.detail,
        frame.gpu,
        library.material_revision(),
        scene_live,
    )
}

fn settle_books(
    books: &mut ImpostorBooks,
    frame: &Frame,
    (jobs, records): (&mut Jobs<'_, '_>, &Records),
    sources: Sources<'_>,
    asks: &Query<'_, '_, (Entity, Ref<'static, ImpostorAsks>)>,
) -> Vec<(Entity, ImpostorRecords)> {
    let Some(device) = books.device else {
        return Vec::new();
    };
    let acked = books.acked;
    let made = std::mem::take(&mut books.made);
    let ImpostorBooks {
        books: held, next, ..
    } = books;
    let mut cx = Supplying {
        device,
        jobs,
        next,
        cooking: frame.cooking,
    };
    for book in held.values_mut() {
        book.work.retain(|(at, _)| *at > acked);
    }
    for (role, identity, bytes) in made {
        if let Some(bytes) = &bytes {
            let ticket = cx.jobs.keep(&IMPOSTOR, &identity, bytes.clone());
            cx.jobs.cancel(ticket);
        }
        if let Some(book) = held.get_mut(&role) {
            book.answer(identity, bytes.map(Arc::from));
        }
    }
    let mut served = BTreeMap::new();
    for (entity, asked) in asks.iter() {
        served.entry(asked.role).or_insert((entity, asked));
    }
    let mut written = Vec::new();
    for (role, (entity, asked)) in served {
        let scene_live = match role {
            AtlasRole::Scene => 0,
            AtlasRole::Placed => held
                .get(&AtlasRole::Scene)
                .map_or(0, |book| book.layers.live()),
        };
        let created = !held.contains_key(&role);
        let book = held
            .entry(role)
            .or_insert_with(|| Book::new(asked.shape, &device));
        if created {
            let shape = book.shape;
            book.push(&mut cx, AtlasWork::Reset { shape });
        }
        book.poll(&mut cx, records);
        book.loaded = 0;
        if std::mem::take(&mut book.deferred) {
            book.generation += 1;
        }
        if !asked.is_changed() && book.key == Some(settle_key(book, frame, sources, scene_live)) {
            continue;
        }
        let settled = match role {
            AtlasRole::Scene => settle_scene(book, &mut cx, frame, sources, &asked),
            AtlasRole::Placed => settle_placed(book, &mut cx, frame, sources, &asked, scene_live),
        };
        book.records = settled;
        book.key = Some(settle_key(book, frame, sources, scene_live));
        written.push((entity, book.records.clone()));
    }
    written
}

fn keep_key(residency: &AssetResidency) -> impl Fn(&ImpostorKey) -> bool + '_ {
    move |key: &ImpostorKey| {
        let &(mesh, material) = key;
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
            .then(|| library.material(handle).copied())
            .flatten()
    }
}

fn retain_book(book: &mut Book, cx: &mut Supplying<'_, '_, '_>, sources: Sources<'_>) {
    book.retain(cx, keep_key(sources.1), current_material(sources));
}

fn retain_contents(book: &mut Book, residency: &AssetResidency) {
    book.contents.retain(
        |mesh| residency.resident(mesh),
        |texture| residency.texture_resident(texture),
    );
}

fn bake_of(
    book: &mut Book,
    (library, residency): Sources<'_>,
    ask: &ImpostorAsk,
    material: Option<Material>,
) -> Option<Bake> {
    let mesh = library.mesh(ask.mesh)?;
    let textures = |handle: TextureHandle| library.texture(handle).map(|texture| &**texture);
    let mut content = (
        book.contents.mesh(ask.mesh, mesh),
        book.contents.material(material.as_ref(), textures),
    );
    let bounds = mesh.bounds();
    let (mut low, mut high) = (bounds.min(), bounds.max());
    let mut masked_any = masked(material.as_ref());
    let mut foliage_any = shades_as_foliage(material.as_ref());
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
            combined(content.0, book.contents.mesh(part_mesh, finest)),
            combined(content.1, book.contents.material(Some(value), textures)),
        );
        let bounds = finest.bounds();
        low = low.min(bounds.min());
        high = high.max(bounds.max());
        masked_any |= masked(Some(value));
        foliage_any |= shades_as_foliage(Some(value));
        parts.push((part_mesh, part_material));
    }
    Some(Bake {
        material,
        bounds: (low, high),
        content,
        masked: masked_any,
        foliage: foliage_any,
        parts,
    })
}

fn supply_ask(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    sources: Sources<'_>,
    ask: &ImpostorAsk,
    budget: Option<u32>,
) -> Option<Supplied> {
    if !sources.1.resident(ask.mesh) {
        return None;
    }
    let material = ask.material.and_then(current_material(sources));
    let key = (ask.mesh, ask.material);
    if let Some(resident) = book.record(key, material.as_ref()) {
        return Some(Supplied::Resident(resident));
    }
    let bake = bake_of(book, sources, ask, material)?;
    Some(book.supply(cx, key, &bake, budget))
}

fn supply_all(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    sources: Sources<'_>,
    (asks, enabled, budget): (&[Option<ImpostorAsk>], bool, Option<u32>),
    settled: &mut ImpostorRecords,
) {
    for ask in asks {
        let supplied = match ask {
            Some(ask) if enabled => supply_ask(book, cx, sources, ask, budget),
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

fn settle_scene(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    frame: &Frame,
    sources: Sources<'_>,
    asked: &ImpostorAsks,
) -> ImpostorRecords {
    let mut settled = ImpostorRecords::default();
    book.fit(cx, asked.shape);
    retain_book(book, cx, sources);
    retain_contents(book, sources.1);
    if !(frame.detail && frame.policy.enabled) {
        book.release(cx);
        settled.live = book.layers.live();
        return settled;
    }
    if !frame.gpu {
        settled.live = book.layers.live();
        return settled;
    }
    book.enforce_budget(frame.policy.max, cx.cooking);
    supply_all(
        book,
        cx,
        sources,
        (&asked.asks, true, frame.policy.max),
        &mut settled,
    );
    book.drawn |= book.capacity > 0 && !settled.records.is_empty();
    settled.live = book.layers.live();
    settled
}

fn settle_placed(
    book: &mut Book,
    cx: &mut Supplying<'_, '_, '_>,
    frame: &Frame,
    sources: Sources<'_>,
    asked: &ImpostorAsks,
    scene_live: u32,
) -> ImpostorRecords {
    let mut settled = ImpostorRecords::default();
    let enabled = frame.gpu && frame.detail && frame.policy.enabled;
    book.fit(cx, asked.shape);
    retain_contents(book, sources.1);
    retain_book(book, cx, sources);
    let budget = frame
        .policy
        .max
        .map(|budget| budget.saturating_sub(scene_live));
    book.enforce_budget(budget, cx.cooking);
    supply_all(
        book,
        cx,
        sources,
        (&asked.asks, enabled, budget),
        &mut settled,
    );
    book.release_targets(cx);
    book.drawn |= book.capacity > 0 && !settled.records.is_empty();
    settled.live = book.layers.live();
    settled
}
