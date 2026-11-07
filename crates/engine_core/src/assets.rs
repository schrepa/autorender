use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_assets::{Asset, AssetLibrary, MaterialHandle, Mesh, MeshHandle, MeshLods, TextureHandle};
pub use engine_textures::solid_block;

use crate::{Census, Feed, Interest, Request, Requests};

mod impostors;

pub use impostors::*;

#[derive(Component, Clone, Debug, Default)]
pub struct AssetUses {
    assets: Vec<Asset>,
    _held: Vec<Interest<Asset>>,
}

impl AssetUses {
    pub fn new(residency: &AssetResidency, mut assets: Vec<Asset>) -> Self {
        assets.sort_unstable();
        assets.dedup();
        let held = assets.iter().map(|&asset| residency.hold(asset)).collect();
        Self {
            assets,
            _held: held,
        }
    }

    pub fn assets(&self) -> &[Asset] {
        &self.assets
    }
}

impl PartialEq for AssetUses {
    fn eq(&self, other: &Self) -> bool {
        self.assets == other.assets
    }
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

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshResidence {
    Arena(MeshPlacement),
    Own,
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
    placement: MeshResidence,
    revision: u64,
    extended: bool,
    index_count: u32,
    byte_len: u64,
    lods: Vec<(u32, u32, f32)>,
    plan: Vec<(u32, u32, f32)>,
    without_levels: bool,
}

impl ResidentMesh {
    pub fn first_index(&self) -> u32 {
        match self.placement {
            MeshResidence::Arena(placement) => placement.first_index,
            MeshResidence::Own => 0,
        }
    }

    pub fn base_vertex(&self) -> i32 {
        match self.placement {
            MeshResidence::Arena(placement) => placement.base_vertex as i32,
            MeshResidence::Own => 0,
        }
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    pub fn revision(&self) -> u64 {
        self.revision
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

    pub fn without_levels(&self) -> bool {
        self.without_levels
    }

    #[doc(hidden)]
    pub fn placement(&self) -> MeshResidence {
        self.placement
    }
}

fn plan_levels(lods: &[(u32, u32, f32)]) -> Vec<(u32, u32, f32)> {
    let first = lods.first().copied().filter(|&(_, count, _)| count > 0);
    let empty = lods.last().copied().filter(|&(_, count, _)| count == 0);
    first.into_iter().chain(empty).collect()
}

fn resident_mesh(
    (mesh, lods): (&Mesh, Option<&MeshLods>),
    revision: u64,
    arena: &mut Arena,
) -> ResidentMesh {
    let extended = mesh.colors().is_some() || mesh.uv1().is_some();
    let streams = if extended {
        MeshStreams::Extended
    } else {
        MeshStreams::Plain
    };
    let total = mesh.indices().len() + lods.map_or(0, |lods| lods.indices.len());
    let placement = u32::try_from(total)
        .ok()
        .and_then(|count| arena.insert(streams, mesh.vertices().len() as u32, count))
        .map_or(MeshResidence::Own, MeshResidence::Arena);
    let levels: Vec<(u32, u32, f32)> = lods.map_or_else(Vec::new, |lods| {
        lods.levels
            .iter()
            .map(|level| (level.first_index, level.index_count, level.error))
            .collect()
    });
    ResidentMesh {
        placement,
        revision,
        extended,
        index_count: mesh.indices().len() as u32,
        byte_len: mesh.byte_len() as u64,
        plan: plan_levels(&levels),
        lods: levels,
        without_levels: lods.is_none(),
    }
}

#[derive(Debug)]
pub enum Resident {
    Mesh(ResidentMesh),
    Material {
        revision: u64,
        textures: Vec<Interest<Asset>>,
    },
    Texture {
        revision: u64,
    },
}

impl Resident {
    pub fn revision(&self) -> u64 {
        match self {
            Self::Mesh(mesh) => mesh.revision,
            Self::Material { revision, .. } | Self::Texture { revision } => *revision,
        }
    }
}

enum Resolved {
    Ready(Resident),
    Waiting,
    Gone,
}

#[derive(Resource)]
#[component(immutable)]
pub struct AssetResidency {
    wanted: Requests<Asset, Resident>,
    arena: Option<Arena>,
    placed: Feed<MeshHandle>,
    seen: Option<(u64, u64)>,
    revision: u64,
}

impl AssetResidency {
    pub(crate) fn empty() -> Self {
        Self {
            wanted: Requests::default(),
            arena: None,
            placed: Feed::new(crate::FeedSettings::default().mesh_residency as usize),
            seen: None,
            revision: 0,
        }
    }

    pub fn hold(&self, asset: impl Into<Asset>) -> Interest<Asset> {
        self.wanted.ask_later(asset.into())
    }

    pub fn state(&self, asset: impl Into<Asset>) -> Option<&Request<Resident>> {
        self.wanted.get(&asset.into())
    }

    pub fn census(&self) -> Census {
        self.wanted.census()
    }

    pub fn holders(&self, asset: impl Into<Asset>) -> u32 {
        self.wanted.interests(&asset.into())
    }

    pub fn resident(&self, mesh: MeshHandle) -> bool {
        self.mesh(mesh).is_some()
    }

    pub fn placed(&self) -> &Feed<MeshHandle> {
        &self.placed
    }

    pub fn mesh(&self, mesh: MeshHandle) -> Option<&ResidentMesh> {
        let Some(Request::Ready(Resident::Mesh(resident))) = self.wanted.get(&Asset::Mesh(mesh))
        else {
            return None;
        };
        Some(resident)
    }

    pub fn material_resident(&self, material: MaterialHandle) -> bool {
        matches!(
            self.wanted.get(&Asset::Material(material)),
            Some(Request::Ready(Resident::Material { .. }))
        )
    }

    pub fn texture_resident(&self, texture: TextureHandle) -> bool {
        matches!(
            self.wanted.get(&Asset::Texture(texture)),
            Some(Request::Ready(Resident::Texture { .. }))
        )
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    #[doc(hidden)]
    pub fn meshes(&self) -> impl Iterator<Item = (MeshHandle, &ResidentMesh)> + '_ {
        self.wanted
            .iter()
            .filter_map(|(asset, request)| {
                if let (Asset::Mesh(handle), Request::Ready(Resident::Mesh(mesh))) = (asset, request) {
                    Some((*handle, mesh))
                } else {
                    None
                }
            })
    }

    #[doc(hidden)]
    pub fn materials(&self) -> impl Iterator<Item = (MaterialHandle, u64)> + '_ {
        self.wanted
            .iter()
            .filter_map(|(asset, request)| {
                if let (Asset::Material(handle), Request::Ready(Resident::Material { revision, .. })) =
                    (asset, request)
                {
                    Some((*handle, *revision))
                } else {
                    None
                }
            })
    }

    #[doc(hidden)]
    pub fn textures(&self) -> impl Iterator<Item = TextureHandle> + '_ {
        self.wanted
            .iter()
            .filter_map(|(asset, request)| {
                if let (Asset::Texture(handle), Request::Ready(Resident::Texture { .. })) =
                    (asset, request)
                {
                    Some(*handle)
                } else {
                    None
                }
            })
    }

    #[doc(hidden)]
    pub fn capacities(&self) -> Option<([u32; 2], u32)> {
        self.arena.as_ref().map(|arena| {
            (
                [arena.classes[0].capacity, arena.classes[1].capacity],
                arena.indices.capacity,
            )
        })
    }

    fn release(&mut self, asset: Asset, request: &Request<Resident>) {
        if let (Asset::Mesh(handle), Request::Ready(Resident::Mesh(mesh))) = (asset, request) {
            if let (MeshResidence::Arena(placement), Some(arena)) = (mesh.placement, self.arena.as_mut())
            {
                arena.remove(placement);
            }
            self.placed.push(handle);
        }
        self.revision = self.revision.wrapping_add(1);
    }

    fn resolve(&mut self, asset: Asset, library: &AssetLibrary) -> Resolved {
        let Some(revision) = library.revision_of(asset) else {
            return Resolved::Gone;
        };
        match asset {
            Asset::Mesh(handle) => {
                let (Some(mesh), Some(lods), Some(arena)) = (
                    library.mesh(handle),
                    library.mesh_lods(handle),
                    self.arena.as_mut(),
                ) else {
                    return Resolved::Waiting;
                };
                let resident = resident_mesh((mesh, lods.map(|lods| &**lods)), revision, arena);
                self.placed.push(handle);
                Resolved::Ready(Resident::Mesh(resident))
            }
            Asset::Material(handle) => {
                let Some(material) = library.material(handle) else {
                    return Resolved::Waiting;
                };
                if material
                    .textures()
                    .any(|texture| library.texture(texture).is_none())
                {
                    return Resolved::Waiting;
                }
                let textures = material
                    .textures()
                    .map(|texture| self.wanted.ask_later(Asset::Texture(texture)))
                    .collect();
                Resolved::Ready(Resident::Material { revision, textures })
            }
            Asset::Texture(handle) => match library.texture(handle) {
                Some(_) => Resolved::Ready(Resident::Texture { revision }),
                None => Resolved::Waiting,
            },
        }
    }

    fn settle(&mut self, library: &AssetLibrary, now: f64) {
        let mut dirty = self.seen != Some(library.revision());
        while self.wanted.changed() || dirty {
            dirty = false;
            for (asset, request) in self.wanted.settle(now) {
                self.release(asset, &request);
            }
            let stale: Vec<Asset> = self
                .wanted
                .iter()
                .filter(|&(asset, request)| match request {
                    Request::Ready(resident) => {
                        library.revision_of(*asset) != Some(resident.revision())
                    }
                    Request::Asked(_) => true,
                    Request::Missing | Request::Failed(_) => library.revision_of(*asset).is_some(),
                })
                .map(|(asset, _)| *asset)
                .collect();
            for asset in stale {
                self.restate(asset, library);
            }
        }
        self.seen = Some(library.revision());
    }

    fn restate(&mut self, asset: Asset, library: &AssetLibrary) {
        let held = self.wanted.get(&asset);
        let was_ready = matches!(held, Some(Request::Ready(_)));
        let placed = if let Some(Request::Ready(Resident::Mesh(mesh))) = held {
            Some(mesh.placement)
        } else {
            None
        };
        if let (Some(MeshResidence::Arena(placement)), Some(arena)) = (placed, self.arena.as_mut()) {
            arena.remove(placement);
        }
        let resolved = self.resolve(asset, library);
        let Some(ticket) = self.wanted.ask_again(&asset) else {
            return;
        };
        match resolved {
            Resolved::Ready(resident) => {
                self.wanted.ready(&asset, ticket, resident);
                self.revision = self.revision.wrapping_add(1);
            }
            Resolved::Waiting => {
                if was_ready {
                    self.revision = self.revision.wrapping_add(1);
                }
            }
            Resolved::Gone => {
                self.wanted.miss(&asset, ticket);
                self.revision = self.revision.wrapping_add(1);
            }
        }
    }

    fn unplace(&mut self) {
        let placed: Vec<Asset> = self.meshes().map(|(handle, _)| Asset::Mesh(handle)).collect();
        for asset in placed {
            self.wanted.ask_again(&asset);
        }
    }
}

fn settle_residency(world: &mut World) {
    let size = world
        .get_resource::<crate::FeedSettings>()
        .map_or(crate::FeedSettings::default().mesh_residency, |feeds| {
            feeds.mesh_residency
        });
    let now = world
        .get_resource::<crate::FrameTime>()
        .map_or(0.0, |time| time.real_elapsed);
    let Some(mut residency) = world
        .modify_resource(|held: &mut AssetResidency| {
            std::mem::replace(held, AssetResidency::empty())
        })
        .ok()
        .flatten()
    else {
        return;
    };
    residency.placed.resize(size as usize);
    if let Some(library) = world.get_resource::<AssetLibrary>() {
        residency.settle(library, now);
    }
    let _ = world.modify_resource(|held: &mut AssetResidency| *held = residency);
}

pub(crate) fn install(world: &mut World) {
    world.insert_resource(AssetResidency::empty());
    world.init_resource::<ImpostorBooks>();
}

#[doc(hidden)]
pub fn configure_assets(world: &mut World, arena: ArenaShape, atlas: AtlasDevice) {
    let _ = world.modify_resource(|residency: &mut AssetResidency| {
        residency.unplace();
        residency.arena = Some(Arena::new(arena));
        residency.placed.replace();
        residency.seen = None;
        residency.revision = residency.revision.wrapping_add(1);
    });
    impostors::configure(world, atlas);
}

#[doc(hidden)]
pub fn settle_assets(world: &mut World) {
    {
        let _profile = engine_profile::span("render", "validate_and_retire");
        settle_residency(world);
    }
    crate::drawn::resolve(world);
    impostors::settle_impostors(world);
}
