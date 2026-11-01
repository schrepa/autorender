use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::change_detection::Mut;
use bevy_ecs::component::Component;
use bevy_ecs::query::QueryState;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_assets::{AssetLibrary, MaterialHandle, Mesh, MeshHandle, MeshLods, TextureHandle};
pub use engine_textures::solid_block;

mod impostors;

pub use impostors::*;

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
    impostors::configure(world, atlas);
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
    impostors::settle_impostors(world);
}
