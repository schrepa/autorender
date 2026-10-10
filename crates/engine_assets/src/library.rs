use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use bevy_ecs::prelude::Resource;

use crate::{AnimationClip, AssetError, Material, Mesh, Texture};

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

/// Handles name assets; libraries check them as they resolve.
impl engine_settings::Setting for MeshHandle {}
impl engine_settings::Setting for MaterialHandle {}
impl engine_settings::Setting for TextureHandle {}

/// Weak runtime animation clip reference, valid only in the originating
/// library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClipHandle(Id);

/// Library admission limits. Zero is supported for every limit; the default
/// admits everything, so only memory bounds a library until a game sets one.
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
    // Advances with every material replacement; each material keeps the
    // value of its last one (0 until replaced).
    material_revision: u64,
    budget: AssetBudget,
    meshes: Storage<Arc<Mesh>>,
    materials: Storage<(Material, u64)>,
    textures: Storage<Arc<Texture>>,
    clips: Storage<Arc<AnimationClip>>,
    resident_bytes: usize,
    texture_bytes: usize,
}

impl Default for AssetLibrary {
    fn default() -> Self {
        Self::new(AssetBudget::default())
    }
}

impl AssetLibrary {
    /// Limits later admissions to `budget`; assets already admitted stay.
    pub fn set_budget(&mut self, budget: AssetBudget) {
        self.budget = budget;
    }

    /// The admission limits.
    pub fn budget(&self) -> AssetBudget {
        self.budget
    }

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
            material_revision: 0,
            budget,
            meshes: Storage::default(),
            materials: Storage::default(),
            textures: Storage::default(),
            clips: Storage::default(),
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
            .textures()
            .any(|texture| self.texture(texture).is_none())
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

    /// Replaces live material `handle` in place, keeping the handle:
    /// objects using it draw with `material` from their next
    /// extraction. Its textures must be live in this library. This advances
    /// [`Self::material_revision`] and the material's
    /// [`Self::material_version`], not [`Self::revision`]: renderers
    /// re-resolve only the objects using it. Prefer per-object tints
    /// (`engine_core::Tint`) for frequent color changes.
    pub fn replace_material(
        &mut self,
        handle: MaterialHandle,
        material: Material,
    ) -> Result<(), AssetError> {
        if material
            .textures()
            .any(|texture| self.texture(texture).is_none())
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

    /// Admits an animation clip; its keys count against the byte budget.
    pub fn insert_clip(&mut self, clip: AnimationClip) -> Result<ClipHandle, AssetError> {
        self.reserve_bytes(clip.byte_len())?;
        Ok(ClipHandle(self.clips.insert(self.id, Arc::new(clip))))
    }

    pub fn clip(&self, handle: ClipHandle) -> Option<&Arc<AnimationClip>> {
        self.clips.get(self.id, handle.0)
    }

    pub fn remove_clip(&mut self, handle: ClipHandle) -> bool {
        if let Some(clip) = self.clips.remove(self.id, handle.0) {
            self.resident_bytes -= clip.byte_len();
            self.revision = self.revision.saturating_add(1);
            true
        } else {
            false
        }
    }

    pub fn mesh(&self, handle: MeshHandle) -> Option<&Arc<Mesh>> {
        self.meshes.get(self.id, handle.0)
    }

    /// Opaque library identity and successful mutation revision: insertions
    /// and removals. Asset lookups may be retained until this token changes,
    /// material contents only while [`Self::material_revision`] stands too.
    /// An exhausted revision (`u64::MAX`) must always be treated as
    /// uncacheable.
    pub fn revision(&self) -> (u64, u64) {
        (self.id, self.revision)
    }

    /// Advances with every [`Self::replace_material`]: compare each retained
    /// material's [`Self::material_version`] to find the replaced ones. An
    /// exhausted value (`u64::MAX`) must always be treated as uncacheable.
    pub fn material_revision(&self) -> u64 {
        self.material_revision
    }

    pub fn material(&self, handle: MaterialHandle) -> Option<&Material> {
        self.materials
            .get(self.id, handle.0)
            .map(|(material, _)| material)
    }

    /// The [`Self::material_revision`] of live material `handle`'s last
    /// replacement, 0 until replaced.
    pub fn material_version(&self, handle: MaterialHandle) -> Option<u64> {
        self.materials
            .get(self.id, handle.0)
            .map(|&(_, version)| version)
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
