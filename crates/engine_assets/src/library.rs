use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use bevy_ecs::prelude::Resource;

use crate::{AssetError, Material, Mesh, MeshLods, Texture, takes_levels};

static NEXT_LIBRARY: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Id {
    library: u64,
    slot: usize,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshHandle(Id);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialHandle(Id);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureHandle(Id);

impl MeshHandle {
    pub fn slot(self) -> usize {
        self.0.slot
    }
}

impl MaterialHandle {
    pub fn slot(self) -> usize {
        self.0.slot
    }
}

impl engine_settings::Setting for MeshHandle {}
impl engine_settings::Setting for MaterialHandle {}
impl engine_settings::Setting for TextureHandle {}

#[derive(Debug, Clone, Copy)]
pub struct AssetBudget {
    pub max_meshes: usize,
    pub max_materials: usize,
    pub max_textures: usize,
    pub max_cpu_bytes: usize,
}

impl Default for AssetBudget {
    fn default() -> Self {
        Self {
            max_meshes: usize::MAX,
            max_materials: usize::MAX,
            max_textures: usize::MAX,
            max_cpu_bytes: usize::MAX,
        }
    }
}

struct Slot<T> {
    generation: u64,
    value: Option<T>,
    reserved: bool,
}

struct Storage<T> {
    slots: Vec<Slot<T>>,
    free: Vec<usize>,
    count: usize,
}

impl<T> Default for Storage<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            count: 0,
        }
    }
}

impl<T> Storage<T> {
    fn has_capacity(&self, limit: usize) -> bool {
        self.count < limit && (!self.free.is_empty() || self.slots.len() < limit)
    }

    fn insert(&mut self, library: u64, value: T) -> Id {
        self.place(library, Some(value))
    }

    fn reserve(&mut self, library: u64) -> Id {
        self.place(library, None)
    }

    fn place(&mut self, library: u64, value: Option<T>) -> Id {
        self.count += 1;
        let reserved = value.is_none();
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index];
            slot.value = value;
            slot.reserved = reserved;
            Id {
                library,
                slot: index,
                generation: slot.generation,
            }
        } else {
            let slot = self.slots.len();
            self.slots.push(Slot {
                generation: 0,
                value,
                reserved,
            });
            Id {
                library,
                slot,
                generation: 0,
            }
        }
    }

    fn fill(&mut self, library: u64, id: Id, value: T) -> bool {
        match self.slot_mut(library, id) {
            Some(slot) if slot.reserved => {
                slot.value = Some(value);
                slot.reserved = false;
                true
            }
            _ => false,
        }
    }

    fn holds(&self, library: u64, id: Id) -> bool {
        id.library == library
            && self.slots.get(id.slot).is_some_and(|slot| {
                slot.generation == id.generation && (slot.value.is_some() || slot.reserved)
            })
    }

    fn slot_mut(&mut self, library: u64, id: Id) -> Option<&mut Slot<T>> {
        if id.library != library {
            return None;
        }
        self.slots
            .get_mut(id.slot)
            .filter(|slot| slot.generation == id.generation)
    }

    fn get(&self, library: u64, id: Id) -> Option<&T> {
        if id.library != library {
            return None;
        }
        let slot = self.slots.get(id.slot)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.value.as_ref()
    }

    fn values(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| entry.value.as_ref().map(|value| (slot, value)))
    }

    fn get_mut(&mut self, library: u64, id: Id) -> Option<&mut T> {
        if id.library != library {
            return None;
        }
        let slot = self.slots.get_mut(id.slot)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.value.as_mut()
    }

    fn remove(&mut self, library: u64, id: Id) -> Option<Option<T>> {
        let slot = self
            .slot_mut(library, id)
            .filter(|slot| slot.value.is_some() || slot.reserved)?;
        let value = slot.value.take();
        slot.reserved = false;
        let next = slot.generation.checked_add(1);
        if let Some(next) = next {
            slot.generation = next;
        }
        self.count -= 1;
        if next.is_some() {
            self.free.push(id.slot);
        }
        Some(value)
    }
}

struct HeldMesh {
    mesh: Arc<Mesh>,
    lods: Option<Option<Arc<MeshLods>>>,
}

impl HeldMesh {
    fn new(mesh: Mesh) -> Self {
        let lods = match mesh.lods() {
            Some(lods) => Some(Some(Arc::clone(lods))),
            None if takes_levels(&mesh) => None,
            None => Some(None),
        };
        Self {
            mesh: Arc::new(mesh),
            lods,
        }
    }
}

#[derive(Resource)]
pub struct AssetLibrary {
    id: u64,
    revision: u64,
    material_revision: u64,
    budget: AssetBudget,
    meshes: Storage<HeldMesh>,
    awaiting_lods: Vec<MeshHandle>,
    materials: Storage<(Material, u64)>,
    textures: Storage<Arc<Texture>>,
    resident_bytes: usize,
    texture_bytes: usize,
}

impl Default for AssetLibrary {
    fn default() -> Self {
        Self::new(AssetBudget::default())
    }
}

impl AssetLibrary {
    pub fn set_budget(&mut self, budget: AssetBudget) {
        self.budget = budget;
    }

    pub fn budget(&self) -> AssetBudget {
        self.budget
    }

    pub fn new(budget: AssetBudget) -> Self {
        let id = NEXT_LIBRARY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .expect("library_ids_left");
        Self {
            id,
            revision: 0,
            material_revision: 0,
            budget,
            meshes: Storage::default(),
            awaiting_lods: Vec::new(),
            materials: Storage::default(),
            textures: Storage::default(),
            resident_bytes: 0,
            texture_bytes: 0,
        }
    }

    pub fn insert_mesh(&mut self, mesh: Mesh) -> Result<MeshHandle, AssetError> {
        if !self.meshes.has_capacity(self.budget.max_meshes) {
            return Err(AssetError::MeshCapacity {
                limit: self.budget.max_meshes,
            });
        }
        self.reserve_bytes(mesh.byte_len())?;
        let held = HeldMesh::new(mesh);
        let awaits = held.lods.is_none();
        let handle = MeshHandle(self.meshes.insert(self.id, held));
        if awaits {
            self.awaiting_lods.push(handle);
        }
        Ok(handle)
    }

    pub fn reserve_mesh(&mut self) -> Result<MeshHandle, AssetError> {
        if !self.meshes.has_capacity(self.budget.max_meshes) {
            return Err(AssetError::MeshCapacity {
                limit: self.budget.max_meshes,
            });
        }
        self.revision = self.revision.saturating_add(1);
        Ok(MeshHandle(self.meshes.reserve(self.id)))
    }

    pub fn fill_mesh(&mut self, handle: MeshHandle, mesh: Mesh) -> Result<(), AssetError> {
        if !self.meshes.holds(self.id, handle.0) || self.mesh(handle).is_some() {
            return Err(AssetError::InvalidMeshHandle);
        }
        self.reserve_bytes(mesh.byte_len())?;
        let held = HeldMesh::new(mesh);
        if held.lods.is_none() {
            self.awaiting_lods.push(handle);
        }
        self.meshes.fill(self.id, handle.0, held);
        Ok(())
    }

    pub fn take_awaiting_lods(&mut self) -> Vec<MeshHandle> {
        std::mem::take(&mut self.awaiting_lods)
    }

    pub fn settle_lods(
        &mut self,
        handle: MeshHandle,
        lods: Option<MeshLods>,
    ) -> Result<(), AssetError> {
        let held = self
            .meshes
            .get_mut(self.id, handle.0)
            .filter(|held| held.lods.is_none())
            .ok_or(AssetError::InvalidMeshHandle)?;
        if let Some(lods) = &lods {
            lods.check(held.mesh.indices().len(), held.mesh.vertices().len())?;
        }
        let bytes = lods.as_ref().map_or(0, MeshLods::byte_len);
        held.lods = Some(lods.map(Arc::new));
        self.resident_bytes += bytes;
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub fn mesh_lods(&self, handle: MeshHandle) -> Option<Option<&Arc<MeshLods>>> {
        let held = self.meshes.get(self.id, handle.0)?;
        held.lods.as_ref().map(Option::as_ref)
    }

    pub fn insert_material(&mut self, material: Material) -> Result<MaterialHandle, AssetError> {
        if material
            .textures()
            .any(|texture| !self.textures.holds(self.id, texture.0))
        {
            return Err(AssetError::InvalidTextureHandle);
        }
        if !self.materials.has_capacity(self.budget.max_materials) {
            return Err(AssetError::MaterialCapacity {
                limit: self.budget.max_materials,
            });
        }
        self.reserve_bytes(size_of::<Material>())?;
        Ok(MaterialHandle(
            self.materials.insert(self.id, (material, 0)),
        ))
    }

    pub fn replace_material(
        &mut self,
        handle: MaterialHandle,
        material: Material,
    ) -> Result<(), AssetError> {
        if material
            .textures()
            .any(|texture| !self.textures.holds(self.id, texture.0))
        {
            return Err(AssetError::InvalidTextureHandle);
        }
        let slot = self
            .materials
            .get_mut(self.id, handle.0)
            .ok_or(AssetError::InvalidMaterialHandle)?;
        self.material_revision = self.material_revision.saturating_add(1);
        *slot = (material, self.material_revision);
        Ok(())
    }

    pub fn insert_texture(&mut self, texture: Texture) -> Result<TextureHandle, AssetError> {
        if !self.textures.has_capacity(self.budget.max_textures) {
            return Err(AssetError::TextureCapacity {
                limit: self.budget.max_textures,
            });
        }
        self.reserve_bytes(texture.byte_len())?;
        self.texture_bytes += texture.byte_len();
        Ok(TextureHandle(
            self.textures.insert(self.id, Arc::new(texture)),
        ))
    }

    pub fn reserve_texture(&mut self) -> Result<TextureHandle, AssetError> {
        if !self.textures.has_capacity(self.budget.max_textures) {
            return Err(AssetError::TextureCapacity {
                limit: self.budget.max_textures,
            });
        }
        self.revision = self.revision.saturating_add(1);
        Ok(TextureHandle(self.textures.reserve(self.id)))
    }

    pub fn fill_texture(
        &mut self,
        handle: TextureHandle,
        texture: Texture,
    ) -> Result<(), AssetError> {
        if !self.textures.holds(self.id, handle.0) || self.texture(handle).is_some() {
            return Err(AssetError::InvalidTextureHandle);
        }
        self.reserve_bytes(texture.byte_len())?;
        self.texture_bytes += texture.byte_len();
        self.textures.fill(self.id, handle.0, Arc::new(texture));
        Ok(())
    }

    pub fn mesh(&self, handle: MeshHandle) -> Option<&Arc<Mesh>> {
        self.meshes.get(self.id, handle.0).map(|held| &held.mesh)
    }

    pub fn revision(&self) -> (u64, u64) {
        (self.id, self.revision)
    }

    pub fn material_revision(&self) -> u64 {
        self.material_revision
    }

    pub fn material(&self, handle: MaterialHandle) -> Option<&Material> {
        self.materials
            .get(self.id, handle.0)
            .map(|(material, _)| material)
    }

    pub fn material_version(&self, handle: MaterialHandle) -> Option<u64> {
        self.materials
            .get(self.id, handle.0)
            .map(|&(_, version)| version)
    }

    pub fn texture(&self, handle: TextureHandle) -> Option<&Arc<Texture>> {
        self.textures.get(self.id, handle.0)
    }

    pub fn remove_mesh(&mut self, handle: MeshHandle) -> bool {
        let Some(mesh) = self.meshes.remove(self.id, handle.0) else {
            return false;
        };
        self.resident_bytes -= mesh.map_or(0, |held| {
            held.mesh.byte_len()
                + held
                    .lods
                    .flatten()
                    .filter(|_| held.mesh.lods().is_none())
                    .map_or(0, |lods| lods.byte_len())
        });
        self.revision = self.revision.saturating_add(1);
        true
    }

    pub fn remove_material(&mut self, handle: MaterialHandle) -> bool {
        if self.materials.remove(self.id, handle.0).is_none() {
            return false;
        }
        self.resident_bytes -= size_of::<Material>();
        self.revision = self.revision.saturating_add(1);
        true
    }

    pub fn remove_texture(&mut self, handle: TextureHandle) -> bool {
        let Some(texture) = self.textures.remove(self.id, handle.0) else {
            return false;
        };
        let bytes = texture.map_or(0, |texture| texture.byte_len());
        self.resident_bytes -= bytes;
        self.texture_bytes -= bytes;
        self.revision = self.revision.saturating_add(1);
        true
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.count
    }

    pub fn material_count(&self) -> usize {
        self.materials.count
    }

    pub fn texture_count(&self) -> usize {
        self.textures.count
    }

    pub fn meshes(&self) -> impl Iterator<Item = (usize, &Arc<Mesh>)> {
        self.meshes.values().map(|(slot, held)| (slot, &held.mesh))
    }

    pub fn materials(&self) -> impl Iterator<Item = (usize, &Material)> {
        self.materials
            .values()
            .map(|(slot, (material, _))| (slot, material))
    }

    pub fn textures(&self) -> impl Iterator<Item = (usize, &Arc<Texture>)> {
        self.textures.values()
    }

    pub fn texture_bytes(&self) -> usize {
        self.texture_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    fn reserve_bytes(&mut self, requested: usize) -> Result<(), AssetError> {
        let available = self
            .budget
            .max_cpu_bytes
            .saturating_sub(self.resident_bytes);
        if requested > available {
            return Err(AssetError::CpuBudget {
                requested,
                available,
            });
        }
        self.resident_bytes += requested;
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }
}
