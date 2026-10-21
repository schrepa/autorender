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

impl MeshHandle {
    /// Its slot in its library, which reports name it by.
    pub fn slot(self) -> usize {
        self.0.slot
    }
}

impl MaterialHandle {
    /// Its slot in its library, which reports name it by.
    pub fn slot(self) -> usize {
        self.0.slot
    }
}

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
    // Held for a value to come.
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
        // Exhausted generations retire their slots permanently. Count those
        // slots against metadata admission as well as bounding live values.
        self.count < limit && (!self.free.is_empty() || self.slots.len() < limit)
    }

    fn insert(&mut self, library: u64, value: T) -> Id {
        self.place(library, Some(value))
    }

    /// A slot for a value to come ([`Self::fill`]), which names none until
    /// then.
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

    /// Puts `value` in reserved slot `id`; whether `id` was reserved.
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

    /// Whether `id` names a value or a reserved slot.
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

    /// Every value held, with its slot, in slot order.
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

    /// Frees slot `id`: its value, `None` for a reserved slot; `None` where
    /// `id` names neither.
    fn remove(&mut self, library: u64, id: Id) -> Option<Option<T>> {
        let slot = self
            .slot_mut(library, id)
            .filter(|slot| slot.value.is_some() || slot.reserved)?;
        let value = slot.value.take();
        slot.reserved = false;
        // Retire an exhausted slot instead of wrapping and reviving old IDs.
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

/// Single CPU asset owner. Insertions and removals are explicit synchronous
/// operations; no background work or automatic eviction exist. A mesh or
/// texture still to come (a record its owner fetches) holds a reserved handle
/// meanwhile, which names nothing until filled: what reads it finds it
/// missing, as a removed one.
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

    /// A handle for a mesh to come, which counts against the mesh capacity
    /// and names no mesh until [`Self::fill_mesh`] gives it one.
    pub fn reserve_mesh(&mut self) -> Result<MeshHandle, AssetError> {
        if !self.meshes.has_capacity(self.budget.max_meshes) {
            return Err(AssetError::MeshCapacity {
                limit: self.budget.max_meshes,
            });
        }
        self.revision = self.revision.saturating_add(1);
        Ok(MeshHandle(self.meshes.reserve(self.id)))
    }

    /// Gives reserved `handle` its mesh, admitted as [`Self::insert_mesh`]
    /// admits one; a handle not reserved is refused.
    pub fn fill_mesh(&mut self, handle: MeshHandle, mesh: Mesh) -> Result<(), AssetError> {
        if !self.meshes.holds(self.id, handle.0) || self.mesh(handle).is_some() {
            return Err(AssetError::InvalidMeshHandle);
        }
        self.reserve_bytes(mesh.byte_len())?;
        self.meshes.fill(self.id, handle.0, Arc::new(mesh));
        Ok(())
    }

    /// Admits `material`, whose textures must be live in this library or
    /// reserved there ([`Self::reserve_texture`]).
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

    /// Replaces live material `handle` in place, keeping the handle:
    /// objects using it draw with `material` from their next
    /// extraction. Its textures must be live in this library or reserved
    /// there. This advances
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

    /// Replaces live texture `handle`'s texels with `texture`'s, keeping
    /// the handle: materials naming it show them from their next
    /// extraction. This advances [`Self::revision`], so retained lookups
    /// resolve again, and [`Self::material_version`] of every material
    /// naming it, so what renderers built from those materials builds
    /// again.
    pub fn replace_texture(
        &mut self,
        handle: TextureHandle,
        texture: Texture,
    ) -> Result<(), AssetError> {
        let old = self
            .texture(handle)
            .ok_or(AssetError::InvalidTextureHandle)?
            .byte_len();
        let new = texture.byte_len();
        if new > old {
            self.reserve_bytes(new - old)?;
        } else {
            self.resident_bytes -= old - new;
        }
        self.texture_bytes = self.texture_bytes - old + new;
        if let Some(slot) = self.textures.get_mut(self.id, handle.0) {
            *slot = Arc::new(texture);
        }
        self.revision = self.revision.saturating_add(1);
        self.material_revision = self.material_revision.saturating_add(1);
        let revision = self.material_revision;
        for slot in &mut self.materials.slots {
            if let Some((material, version)) = &mut slot.value
                && material.textures().any(|texture| texture == handle)
            {
                *version = revision;
            }
        }
        Ok(())
    }

    /// A handle for a texture to come, which materials may name and which
    /// counts against the texture capacity; it names no texture until
    /// [`Self::fill_texture`] gives it one.
    pub fn reserve_texture(&mut self) -> Result<TextureHandle, AssetError> {
        if !self.textures.has_capacity(self.budget.max_textures) {
            return Err(AssetError::TextureCapacity {
                limit: self.budget.max_textures,
            });
        }
        self.revision = self.revision.saturating_add(1);
        Ok(TextureHandle(self.textures.reserve(self.id)))
    }

    /// Gives reserved `handle` its texture, admitted as
    /// [`Self::insert_texture`] admits one; a handle not reserved is refused.
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

    /// Admits an animation clip; its keys count against the byte budget.
    pub fn insert_clip(&mut self, clip: AnimationClip) -> Result<ClipHandle, AssetError> {
        self.reserve_bytes(clip.byte_len())?;
        Ok(ClipHandle(self.clips.insert(self.id, Arc::new(clip))))
    }

    pub fn clip(&self, handle: ClipHandle) -> Option<&Arc<AnimationClip>> {
        self.clips.get(self.id, handle.0)
    }

    pub fn remove_clip(&mut self, handle: ClipHandle) -> bool {
        let Some(clip) = self.clips.remove(self.id, handle.0) else {
            return false;
        };
        self.resident_bytes -= clip.map_or(0, |clip| clip.byte_len());
        self.revision = self.revision.saturating_add(1);
        true
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

    /// Removes mesh `handle`, or releases it while reserved; whether it
    /// named either.
    pub fn remove_mesh(&mut self, handle: MeshHandle) -> bool {
        let Some(mesh) = self.meshes.remove(self.id, handle.0) else {
            return false;
        };
        self.resident_bytes -= mesh.map_or(0, |mesh| mesh.byte_len());
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

    /// Invalidates this handle, or releases it while reserved, without
    /// changing materials that refer to it. Those weak references fail
    /// texture lookup; retained snapshots remain valid.
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

    /// Every mesh held, by slot ([`MeshHandle::slot`]), in slot order.
    pub fn meshes(&self) -> impl Iterator<Item = (usize, &Arc<Mesh>)> {
        self.meshes.values()
    }

    /// Every material held, by slot ([`MaterialHandle::slot`]), in slot
    /// order.
    pub fn materials(&self) -> impl Iterator<Item = (usize, &Material)> {
        self.materials
            .values()
            .map(|(slot, (material, _))| (slot, material))
    }

    /// Every texture held, by slot, in slot order.
    pub fn textures(&self) -> impl Iterator<Item = (usize, &Arc<Texture>)> {
        self.textures.values()
    }

    /// Every animation clip held, by slot, in slot order.
    pub fn clips(&self) -> impl Iterator<Item = (usize, &Arc<AnimationClip>)> {
        self.clips.values()
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
