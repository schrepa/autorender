use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_assets::{
    Asset, AssetLibrary, AttributeFormat, GeometryHandle, GeometryStream, MaterialHandle, Mesh,
    TextureHandle,
};

use crate::{Census, Error, ErrorCode, Expected, Feed, Interest, Request, Requests};

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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassStream {
    Vertex,
    Extra,
    Named(String, AttributeFormat),
}

impl ClassStream {
    pub fn stride(&self) -> u64 {
        match self {
            Self::Vertex => GeometryStream::Vertex.stride(),
            Self::Extra => GeometryStream::Extra.stride(),
            Self::Named(_, format) => format.bytes(),
        }
    }

    pub fn serves(&self, stream: GeometryStream) -> bool {
        match (self, stream) {
            (Self::Vertex, GeometryStream::Vertex) | (Self::Extra, GeometryStream::Extra) => true,
            (Self::Named(name, format), GeometryStream::Named { shape, .. }) => {
                name == shape.name && *format == shape.format
            }
            (Self::Vertex, GeometryStream::Extra | GeometryStream::Named { .. })
            | (Self::Extra, GeometryStream::Vertex | GeometryStream::Named { .. })
            | (Self::Named(..), GeometryStream::Vertex | GeometryStream::Extra) => false,
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeometryClass {
    pub streams: Vec<ClassStream>,
}

impl GeometryClass {
    pub fn of(mesh: &Mesh) -> Self {
        let mut streams = vec![ClassStream::Vertex];
        if mesh.colors().is_some() || mesh.uv1().is_some() || !mesh.streams().is_empty() {
            streams.push(ClassStream::Extra);
        }
        streams.extend(
            mesh.streams()
                .iter()
                .map(|stream| ClassStream::Named(stream.name().to_owned(), stream.format())),
        );
        Self { streams }
    }

    pub fn extended(&self) -> bool {
        self.streams.contains(&ClassStream::Extra)
    }

    fn widest(&self) -> u64 {
        self.streams
            .iter()
            .map(ClassStream::stride)
            .max()
            .unwrap_or(1)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshPlacement {
    pub first_index: u32,
    pub index_count: u32,
    pub class: u32,
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
    pub limit: u64,
}

#[derive(Clone, Debug)]
struct Arena {
    shape: ArenaShape,
    classes: Vec<(GeometryClass, Space)>,
    indices: Space,
}

impl Arena {
    fn new(shape: ArenaShape) -> Self {
        Self {
            shape,
            classes: Vec::new(),
            indices: Space::default(),
        }
    }

    fn class_of(&mut self, class: GeometryClass) -> u32 {
        let at = match self.classes.iter().position(|(held, _)| *held == class) {
            Some(at) => at,
            None => {
                self.classes.push((class, Space::default()));
                self.classes.len() - 1
            }
        };
        at as u32
    }

    fn insert(&mut self, class: u32, vertex_count: u32, index_count: u32) -> Option<MeshPlacement> {
        let limit = self.shape.limit;
        let first_index = match self.indices.allocate(index_count) {
            Some(start) => start,
            None => {
                let grown = grown_capacity(&self.indices, index_count, INDEX_BYTES, limit)?;
                self.indices.grow(grown);
                self.indices.allocate(index_count)?
            }
        };
        let (held, space) = &mut self.classes[class as usize];
        let widest = held.widest();
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
            class,
            base_vertex,
            vertex_count,
        })
    }

    fn remove(&mut self, placement: MeshPlacement) {
        self.indices
            .release(placement.first_index, placement.index_count);
        if let Some((_, space)) = self.classes.get_mut(placement.class as usize) {
            space.release(placement.base_vertex, placement.vertex_count);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidentMesh {
    placement: MeshResidence,
    revision: u64,
    class: u32,
    extended: bool,
    index_count: u32,
    byte_len: u64,
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

    #[doc(hidden)]
    pub fn class(&self) -> u32 {
        self.class
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    #[doc(hidden)]
    pub fn placement(&self) -> MeshResidence {
        self.placement
    }
}

fn resident_mesh(mesh: &Mesh, revision: u64, arena: &mut Arena) -> ResidentMesh {
    let class = GeometryClass::of(mesh);
    let extended = class.extended();
    let class = arena.class_of(class);
    let total = mesh.indices().len() + mesh.lods().map_or(0, |lods| lods.indices.len());
    let placement = u32::try_from(total)
        .ok()
        .and_then(|count| arena.insert(class, mesh.vertices().len() as u32, count))
        .map_or(MeshResidence::Own, MeshResidence::Arena);
    ResidentMesh {
        placement,
        revision,
        class,
        extended,
        index_count: mesh.indices().len() as u32,
        byte_len: mesh.byte_len() as u64,
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
    placed: Feed<GeometryHandle>,
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

    pub fn refusal(&self, asset: impl Into<Asset>, path: &str) -> Option<Error> {
        let gone = || Error::new(ErrorCode::UnknownReference, path).expected(Expected::Live);
        match self.state(asset)? {
            Request::Failed(error) => Some(error.clone()),
            Request::Missing => Some(gone()),
            Request::Asked(_) | Request::Ready(_) => None,
        }
    }

    pub fn made_material(
        &self,
        made: Option<&Result<MaterialHandle, Error>>,
        path: &str,
    ) -> Result<Option<MaterialHandle>, Error> {
        match made {
            None => Ok(None),
            Some(Err(error)) => Err(error.clone()),
            Some(&Ok(handle)) if self.material_resident(handle) => Ok(Some(handle)),
            Some(&Ok(handle)) => self.refusal(handle, path).map_or(Ok(None), Err),
        }
    }

    pub fn census(&self) -> Census {
        self.wanted.census()
    }

    pub fn holders(&self, asset: impl Into<Asset>) -> u32 {
        self.wanted.interests(&asset.into())
    }

    pub fn resident(&self, mesh: GeometryHandle) -> bool {
        self.mesh(mesh).is_some()
    }

    pub fn placed(&self) -> &Feed<GeometryHandle> {
        &self.placed
    }

    pub fn mesh(&self, mesh: GeometryHandle) -> Option<&ResidentMesh> {
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
    pub fn meshes(&self) -> impl Iterator<Item = (GeometryHandle, &ResidentMesh)> + '_ {
        self.wanted.iter().filter_map(|(asset, request)| {
            if let (Asset::Mesh(handle), Request::Ready(Resident::Mesh(mesh))) = (asset, request) {
                Some((*handle, mesh))
            } else {
                None
            }
        })
    }

    #[doc(hidden)]
    pub fn materials(&self) -> impl Iterator<Item = (MaterialHandle, u64)> + '_ {
        self.wanted.iter().filter_map(|(asset, request)| {
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
        self.wanted.iter().filter_map(|(asset, request)| {
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
    pub fn capacities(&self) -> Option<(Vec<(&GeometryClass, u32)>, u32)> {
        self.arena.as_ref().map(|arena| {
            (
                arena
                    .classes
                    .iter()
                    .map(|(class, space)| (class, space.capacity))
                    .collect(),
                arena.indices.capacity,
            )
        })
    }

    #[doc(hidden)]
    pub fn class(&self, class: u32) -> Option<&GeometryClass> {
        self.arena
            .as_ref()?
            .classes
            .get(class as usize)
            .map(|(held, _)| held)
    }

    fn release(&mut self, asset: Asset, request: &Request<Resident>) {
        if let (Asset::Mesh(handle), Request::Ready(Resident::Mesh(mesh))) = (asset, request) {
            if let (MeshResidence::Arena(placement), Some(arena)) =
                (mesh.placement, self.arena.as_mut())
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
                let (Some(mesh), Some(arena)) = (library.mesh(handle), self.arena.as_mut()) else {
                    return Resolved::Waiting;
                };
                let resident = resident_mesh(mesh, revision, arena);
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
        let placed = if let (Asset::Mesh(handle), Some(Request::Ready(Resident::Mesh(mesh)))) =
            (asset, held)
        {
            Some((handle, mesh.placement))
        } else {
            None
        };
        if let Some((handle, placement)) = placed {
            if let (MeshResidence::Arena(placement), Some(arena)) = (placement, self.arena.as_mut())
            {
                arena.remove(placement);
            }
            self.placed.push(handle);
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
        let placed: Vec<Asset> = self
            .meshes()
            .map(|(handle, _)| Asset::Mesh(handle))
            .collect();
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
}

#[doc(hidden)]
pub fn configure_assets(world: &mut World, arena: ArenaShape) {
    let _ = world.modify_resource(|residency: &mut AssetResidency| {
        residency.unplace();
        residency.arena = Some(Arena::new(arena));
        residency.placed.replace();
        residency.seen = None;
        residency.revision = residency.revision.wrapping_add(1);
    });
}

#[doc(hidden)]
pub fn settle_assets(world: &mut World) {
    {
        let _profile = engine_profile::span("render", "validate_and_retire");
        settle_residency(world);
    }
    crate::drawn::resolve(world);
}
