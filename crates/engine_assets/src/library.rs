use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use bevy_ecs::prelude::Resource;

use crate::{AssetError, Material, Mesh, Texture};

static NEXT_LIBRARY: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Id {
    library: u64,
    slot: usize,
    generation: u64,
}

/// Weak runtime mesh reference, valid only in the originating library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshHandle(Id);

/// Weak runtime material reference, valid only in the originating library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialHandle(Id);

/// Weak runtime texture reference, valid only in the originating library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureHandle(Id);

/// Library admission limits. Zero is supported for every limit.
#[derive(Debug, Clone, Copy)]
pub struct AssetBudget {
    pub max_meshes: usize,
    pub max_materials: usize,
    pub max_textures: usize,
    /// Resident payload bytes (vertices, indices, material records and pixels),
    /// excluding allocation/slot metadata and assets retained by old snapshots.
    pub max_cpu_bytes: usize,
}

impl Default for AssetBudget {
    fn default() -> Self {
        Self {
            max_meshes: 128,
            max_materials: 256,
            max_textures: 64,
            max_cpu_bytes: 32 * 1024 * 1024,
        }
    }
}

struct Slot<T> {
    generation: u64,
    value: Option<T>,
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
        // Exhausted generations retire their slots permanently. Count those
        // slots against metadata admission as well as bounding live values.
        self.count < limit && (!self.free.is_empty() || self.slots.len() < limit)
    }

    fn insert(&mut self, library: u64, value: T) -> Id {
        self.count += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index];
            slot.value = Some(value);
            Id {
                library,
                slot: index,
                generation: slot.generation,
            }
        } else {
            let slot = self.slots.len();
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
            Id {
                library,
                slot,
                generation: 0,
            }
        }
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

    fn remove(&mut self, library: u64, id: Id) -> Option<T> {
        self.get(library, id)?;
        let slot = &mut self.slots[id.slot];
        let value = slot.value.take()?;
        self.count -= 1;
        // Retire an exhausted slot instead of wrapping and reviving old IDs.
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.free.push(id.slot);
        }
        Some(value)
    }
}

/// Single CPU asset owner. Insertions and removals are explicit synchronous
/// operations; no loading states, background work, or automatic eviction exist.
#[derive(Resource)]
pub struct AssetLibrary {
    id: u64,
    revision: u64,
    budget: AssetBudget,
    meshes: Storage<Arc<Mesh>>,
    materials: Storage<Material>,
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
    /// Creates an empty library without preallocating the configured budgets.
    /// Panics only if this process exhausts all possible library identities.
    pub fn new(budget: AssetBudget) -> Self {
        let id = NEXT_LIBRARY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .expect("asset library identities exhausted");
        Self {
            id,
            revision: 0,
            budget,
            meshes: Storage::default(),
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
        Ok(MeshHandle(self.meshes.insert(self.id, Arc::new(mesh))))
    }

    pub fn insert_material(&mut self, material: Material) -> Result<MaterialHandle, AssetError> {
        if material
            .texture()
            .is_some_and(|texture| self.texture(texture).is_none())
        {
            return Err(AssetError::InvalidTextureHandle);
        }
        if !self.materials.has_capacity(self.budget.max_materials) {
            return Err(AssetError::MaterialCapacity {
                limit: self.budget.max_materials,
            });
        }
        self.reserve_bytes(size_of::<Material>())?;
        Ok(MaterialHandle(self.materials.insert(self.id, material)))
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

    pub fn mesh(&self, handle: MeshHandle) -> Option<&Arc<Mesh>> {
        self.meshes.get(self.id, handle.0)
    }

    /// Opaque library identity and successful mutation revision. Immutable
    /// asset lookups may be retained until this token changes. An exhausted
    /// revision (`u64::MAX`) must always be treated as uncacheable.
    pub fn revision(&self) -> (u64, u64) {
        (self.id, self.revision)
    }

    pub fn material(&self, handle: MaterialHandle) -> Option<&Material> {
        self.materials.get(self.id, handle.0)
    }

    pub fn texture(&self, handle: TextureHandle) -> Option<&Arc<Texture>> {
        self.textures.get(self.id, handle.0)
    }

    pub fn remove_mesh(&mut self, handle: MeshHandle) -> bool {
        if let Some(mesh) = self.meshes.remove(self.id, handle.0) {
            self.resident_bytes -= mesh.byte_len();
            self.revision = self.revision.saturating_add(1);
            true
        } else {
            false
        }
    }

    pub fn remove_material(&mut self, handle: MaterialHandle) -> bool {
        if self.materials.remove(self.id, handle.0).is_some() {
            self.resident_bytes -= size_of::<Material>();
            self.revision = self.revision.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Invalidates this handle without changing materials that refer to it.
    /// Those weak references fail texture lookup; retained snapshots remain valid.
    pub fn remove_texture(&mut self, handle: TextureHandle) -> bool {
        if let Some(texture) = self.textures.remove(self.id, handle.0) {
            self.resident_bytes -= texture.byte_len();
            self.texture_bytes -= texture.byte_len();
            self.revision = self.revision.saturating_add(1);
            true
        } else {
            false
        }
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

    /// Resident texture pixel bytes, also included in [`Self::resident_bytes`].
    pub fn texture_bytes(&self) -> usize {
        self.texture_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    fn reserve_bytes(&mut self, requested: usize) -> Result<(), AssetError> {
        let available = self.budget.max_cpu_bytes - self.resident_bytes;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_generations_cannot_grow_metadata_beyond_admission_limits() {
        let mut library = AssetLibrary::new(AssetBudget {
            max_meshes: 1,
            max_materials: 1,
            max_textures: 1,
            ..AssetBudget::default()
        });
        let mut mesh = library.insert_mesh(Mesh::cube()).unwrap();
        let mut material = library
            .insert_material(Material::new([1.0; 4]).unwrap())
            .unwrap();
        let mut texture = library
            .insert_texture(Texture::new(1, 1, vec![255; 4]).unwrap())
            .unwrap();
        // Simulate the last valid generation without executing 2^64 removals.
        library.meshes.slots[mesh.0.slot].generation = u64::MAX;
        mesh.0.generation = u64::MAX;
        library.materials.slots[material.0.slot].generation = u64::MAX;
        material.0.generation = u64::MAX;
        library.textures.slots[texture.0.slot].generation = u64::MAX;
        texture.0.generation = u64::MAX;
        assert!(library.remove_mesh(mesh));
        assert!(library.remove_material(material));
        assert!(library.remove_texture(texture));
        assert_eq!(library.mesh_count(), 0);
        assert_eq!(library.material_count(), 0);
        assert_eq!(library.texture_count(), 0);
        assert_eq!(library.texture_bytes(), 0);
        assert_eq!(library.resident_bytes(), 0);
        assert_eq!(
            library.insert_mesh(Mesh::plane()).unwrap_err(),
            AssetError::MeshCapacity { limit: 1 }
        );
        assert_eq!(
            library
                .insert_material(Material::new([1.0; 4]).unwrap())
                .unwrap_err(),
            AssetError::MaterialCapacity { limit: 1 }
        );
        assert_eq!(library.meshes.slots.len(), 1);
        assert_eq!(
            library
                .insert_texture(Texture::new(1, 1, vec![255; 4]).unwrap())
                .unwrap_err(),
            AssetError::TextureCapacity { limit: 1 }
        );
        assert_eq!(library.materials.slots.len(), 1);
        assert_eq!(library.textures.slots.len(), 1);
        assert!(library.mesh(mesh).is_none());
        assert!(library.material(material).is_none());
        assert!(library.texture(texture).is_none());
        assert_eq!(library.resident_bytes(), 0);
    }
}
