use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use bevy_ecs::prelude::Resource;
use engine_codec::{Error, ErrorCode, Expected};

use crate::{Material, Mesh, Texture};

fn full(path: &str, limit: usize) -> Error {
    Error::new(ErrorCode::AssetBudgetExceeded, path).at_most(limit as u64)
}

fn dead(code: ErrorCode, path: &str) -> Error {
    Error::new(code, path).expected(Expected::Live)
}

static NEXT_LIBRARY: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Id {
    library: u64,
    slot: usize,
    generation: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeometryHandle(Id);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialHandle(Id);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureHandle(Id);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Asset {
    Mesh(GeometryHandle),
    Material(MaterialHandle),
    Texture(TextureHandle),
}

impl From<GeometryHandle> for Asset {
    fn from(handle: GeometryHandle) -> Self {
        Self::Mesh(handle)
    }
}

impl From<MaterialHandle> for Asset {
    fn from(handle: MaterialHandle) -> Self {
        Self::Material(handle)
    }
}

impl From<TextureHandle> for Asset {
    fn from(handle: TextureHandle) -> Self {
        Self::Texture(handle)
    }
}

impl GeometryHandle {
    pub fn slot(self) -> usize {
        self.0.slot
    }
}

impl MaterialHandle {
    pub fn slot(self) -> usize {
        self.0.slot
    }
}

impl TextureHandle {
    pub fn slot(self) -> usize {
        self.0.slot
    }
}

impl engine_settings::Setting for GeometryHandle {}
impl engine_settings::Setting for MaterialHandle {}
impl engine_settings::Setting for TextureHandle {}

thread_local! {
    static SCOPE: RefCell<Option<(Arc<SceneAssets>, bool)>> = const { RefCell::new(None) };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneTable<H> {
    names: BTreeMap<String, H>,
    order: Vec<String>,
}

impl<H> Default for SceneTable<H> {
    fn default() -> Self {
        Self {
            names: BTreeMap::new(),
            order: Vec::new(),
        }
    }
}

impl<H> std::ops::Deref for SceneTable<H> {
    type Target = BTreeMap<String, H>;

    fn deref(&self) -> &Self::Target {
        &self.names
    }
}

impl<H> FromIterator<(String, H)> for SceneTable<H> {
    fn from_iter<I: IntoIterator<Item = (String, H)>>(entries: I) -> Self {
        let mut table = Self::default();
        for (name, handle) in entries {
            if table.names.insert(name.clone(), handle).is_none() {
                table.order.push(name);
            }
        }
        table
    }
}

impl<H> SceneTable<H> {
    pub fn iter(&self) -> impl Iterator<Item = (&String, &H)> + '_ {
        self.order.iter().map(|name| (name, &self.names[name]))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneAssets {
    pub meshes: SceneTable<GeometryHandle>,
    pub materials: SceneTable<MaterialHandle>,
    pub textures: SceneTable<TextureHandle>,
}

struct Restore(Option<(Arc<SceneAssets>, bool)>);

impl Drop for Restore {
    fn drop(&mut self) {
        let previous = self.0.take();
        SCOPE.with(|scope| *scope.borrow_mut() = previous);
    }
}

impl SceneAssets {
    pub fn scope<R>(self: &Arc<Self>, run: impl FnOnce() -> R) -> R {
        self.within(false, run)
    }

    pub fn lenient<R>(self: &Arc<Self>, run: impl FnOnce() -> R) -> R {
        self.within(true, run)
    }

    fn within<R>(self: &Arc<Self>, lenient: bool, run: impl FnOnce() -> R) -> R {
        let previous = SCOPE.with(|scope| scope.borrow_mut().replace((Arc::clone(self), lenient)));
        let _restore = Restore(previous);
        run()
    }
}

fn named<H: Copy + PartialEq>(
    handle: H,
    table: fn(&SceneAssets) -> &SceneTable<H>,
    kind: &str,
) -> Result<String, String> {
    SCOPE.with(|scope| {
        let scope = scope.borrow();
        let (assets, _) = scope
            .as_ref()
            .ok_or_else(|| format!("a {kind} handle is written only within its scene's assets"))?;
        table(assets)
            .iter()
            .find(|(_, known)| **known == handle)
            .map(|(name, _)| name.clone())
            .ok_or_else(|| format!("the scene names no such {kind}"))
    })
}

fn handle<H: Copy + Default>(
    name: &str,
    table: fn(&SceneAssets) -> &SceneTable<H>,
    kind: &str,
) -> Result<H, String> {
    SCOPE.with(|scope| {
        let scope = scope.borrow();
        let (assets, lenient) = scope
            .as_ref()
            .ok_or_else(|| format!("a {kind} handle is read only within its scene's assets"))?;
        match reference(name, table(assets)) {
            Ok(handle) => Ok(handle),
            Err(_) if *lenient => Ok(H::default()),
            Err(error) => Err(error.to_string()),
        }
    })
}

fn reference<H: Copy>(name: &str, names: &SceneTable<H>) -> Result<H, Error> {
    if !engine_settings::is_id(name) {
        return Err(Error::new(ErrorCode::InvalidId, "$")
            .expected(Expected::Id)
            .at_most(engine_settings::MAX_ID_BYTES as u64)
            .got(name));
    }
    names.get(name).copied().ok_or_else(|| {
        Error::new(ErrorCode::UnknownReference, "$")
            .got(name)
            .variants(names.keys().cloned())
    })
}

fn handle_schema() -> schemars::Schema {
    schemars::json_schema!({
        "type": "string",
        "maxLength": engine_settings::MAX_ID_BYTES,
        "minLength": 1,
        "pattern": engine_settings::ID_PATTERN,
    })
}

macro_rules! scene_named {
    ($handle:ident, $table:ident, $kind:literal) => {
        impl serde::Serialize for $handle {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let name = named(*self, |assets| &assets.$table, $kind)
                    .map_err(<S::Error as serde::ser::Error>::custom)?;
                serializer.serialize_str(&name)
            }
        }

        impl<'de> serde::Deserialize<'de> for $handle {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let name = <String as serde::Deserialize>::deserialize(deserializer)?;
                handle(&name, |assets| &assets.$table, $kind)
                    .map_err(<D::Error as serde::de::Error>::custom)
            }
        }

        impl schemars::JsonSchema for $handle {
            fn inline_schema() -> bool {
                true
            }

            fn schema_name() -> std::borrow::Cow<'static, str> {
                stringify!($handle).into()
            }

            fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
                handle_schema()
            }
        }
    };
}

scene_named!(GeometryHandle, meshes, "mesh");
scene_named!(MaterialHandle, materials, "material");
scene_named!(TextureHandle, textures, "texture");

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

enum Held<T> {
    Free,
    Reserved,
    Filled(T),
}

struct Slot<T> {
    generation: u64,
    revision: u64,
    held: Held<T>,
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

    fn insert(&mut self, library: u64, value: T, revision: u64) -> Id {
        self.place(library, Held::Filled(value), revision)
    }

    fn reserve(&mut self, library: u64, revision: u64) -> Id {
        self.place(library, Held::Reserved, revision)
    }

    fn place(&mut self, library: u64, held: Held<T>, revision: u64) -> Id {
        self.count += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index];
            slot.held = held;
            slot.revision = revision;
            Id {
                library,
                slot: index,
                generation: slot.generation,
            }
        } else {
            let slot = self.slots.len();
            self.slots.push(Slot {
                generation: 0,
                revision,
                held,
            });
            Id {
                library,
                slot,
                generation: 0,
            }
        }
    }

    fn fill(&mut self, library: u64, id: Id, value: T, revision: u64) -> bool {
        let Some(slot) = self.slot_mut(library, id) else {
            return false;
        };
        match slot.held {
            Held::Reserved => {
                slot.held = Held::Filled(value);
                slot.revision = revision;
                true
            }
            Held::Free | Held::Filled(_) => false,
        }
    }

    fn holds(&self, library: u64, id: Id) -> bool {
        self.revision(library, id).is_some()
    }

    fn revision(&self, library: u64, id: Id) -> Option<u64> {
        if id.library != library {
            return None;
        }
        let slot = self.slots.get(id.slot)?;
        match slot.held {
            Held::Reserved | Held::Filled(_) if slot.generation == id.generation => {
                Some(slot.revision)
            }
            Held::Free | Held::Reserved | Held::Filled(_) => None,
        }
    }

    fn touch(&mut self, library: u64, id: Id, revision: u64) {
        if let Some(slot) = self.slot_mut(library, id) {
            slot.revision = revision;
        }
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
        match &slot.held {
            Held::Filled(value) if slot.generation == id.generation => Some(value),
            Held::Free | Held::Reserved | Held::Filled(_) => None,
        }
    }

    fn values(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| match &entry.held {
                Held::Filled(value) => Some((slot, value)),
                Held::Free | Held::Reserved => None,
            })
    }

    fn get_mut(&mut self, library: u64, id: Id) -> Option<&mut T> {
        let slot = self.slot_mut(library, id)?;
        match &mut slot.held {
            Held::Filled(value) => Some(value),
            Held::Free | Held::Reserved => None,
        }
    }

    fn remove(&mut self, library: u64, id: Id) -> Option<Option<T>> {
        let slot = self.slot_mut(library, id)?;
        let value = match std::mem::replace(&mut slot.held, Held::Free) {
            Held::Free => return None,
            Held::Reserved => None,
            Held::Filled(value) => Some(value),
        };
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
            budget,
            meshes: Storage::default(),
            materials: Storage::default(),
            textures: Storage::default(),
            resident_bytes: 0,
            texture_bytes: 0,
        }
    }

    pub fn insert_mesh(&mut self, mesh: Mesh) -> Result<GeometryHandle, Error> {
        if !self.meshes.has_capacity(self.budget.max_meshes) {
            return Err(full("$.meshes", self.budget.max_meshes));
        }
        self.reserve_bytes(mesh.byte_len())?;
        let revision = self.bump();
        Ok(GeometryHandle(self.meshes.insert(
            self.id,
            Arc::new(mesh),
            revision,
        )))
    }

    pub fn reserve_mesh(&mut self) -> Result<GeometryHandle, Error> {
        if !self.meshes.has_capacity(self.budget.max_meshes) {
            return Err(full("$.meshes", self.budget.max_meshes));
        }
        let revision = self.bump();
        Ok(GeometryHandle(self.meshes.reserve(self.id, revision)))
    }

    pub fn fill_mesh(&mut self, handle: GeometryHandle, mesh: Mesh) -> Result<(), Error> {
        if !self.meshes.holds(self.id, handle.0) || self.mesh(handle).is_some() {
            return Err(dead(ErrorCode::InvalidMesh, "$.mesh"));
        }
        self.reserve_bytes(mesh.byte_len())?;
        let revision = self.bump();
        self.meshes
            .fill(self.id, handle.0, Arc::new(mesh), revision);
        Ok(())
    }

    pub fn insert_material(&mut self, material: Material) -> Result<MaterialHandle, Error> {
        if material
            .textures()
            .any(|texture| !self.textures.holds(self.id, texture.0))
        {
            return Err(dead(ErrorCode::InvalidTexture, "$.textures"));
        }
        if !self.materials.has_capacity(self.budget.max_materials) {
            return Err(full("$.materials", self.budget.max_materials));
        }
        self.reserve_bytes(size_of::<Material>())?;
        let revision = self.bump();
        Ok(MaterialHandle(
            self.materials.insert(self.id, material, revision),
        ))
    }

    pub fn replace_material(
        &mut self,
        handle: MaterialHandle,
        material: Material,
    ) -> Result<(), Error> {
        if material
            .textures()
            .any(|texture| !self.textures.holds(self.id, texture.0))
        {
            return Err(dead(ErrorCode::InvalidTexture, "$.textures"));
        }
        let revision = self.revision.saturating_add(1);
        let slot = self
            .materials
            .get_mut(self.id, handle.0)
            .ok_or_else(|| dead(ErrorCode::InvalidMaterial, "$.material"))?;
        *slot = material;
        self.revision = revision;
        self.materials.touch(self.id, handle.0, revision);
        Ok(())
    }

    pub fn insert_texture(&mut self, texture: Texture) -> Result<TextureHandle, Error> {
        if !self.textures.has_capacity(self.budget.max_textures) {
            return Err(full("$.textures", self.budget.max_textures));
        }
        self.reserve_bytes(texture.byte_len())?;
        self.texture_bytes += texture.byte_len();
        let revision = self.bump();
        Ok(TextureHandle(self.textures.insert(
            self.id,
            Arc::new(texture),
            revision,
        )))
    }

    pub fn reserve_texture(&mut self) -> Result<TextureHandle, Error> {
        if !self.textures.has_capacity(self.budget.max_textures) {
            return Err(full("$.textures", self.budget.max_textures));
        }
        let revision = self.bump();
        Ok(TextureHandle(self.textures.reserve(self.id, revision)))
    }

    pub fn fill_texture(&mut self, handle: TextureHandle, texture: Texture) -> Result<(), Error> {
        if !self.textures.holds(self.id, handle.0) || self.texture(handle).is_some() {
            return Err(dead(ErrorCode::InvalidTexture, "$.texture"));
        }
        self.reserve_bytes(texture.byte_len())?;
        self.texture_bytes += texture.byte_len();
        let revision = self.bump();
        self.textures
            .fill(self.id, handle.0, Arc::new(texture), revision);
        Ok(())
    }

    pub fn mesh(&self, handle: GeometryHandle) -> Option<&Arc<Mesh>> {
        self.meshes.get(self.id, handle.0)
    }

    pub fn revision(&self) -> (u64, u64) {
        (self.id, self.revision)
    }

    pub fn revision_of(&self, asset: impl Into<Asset>) -> Option<u64> {
        match asset.into() {
            Asset::Mesh(handle) => self.meshes.revision(self.id, handle.0),
            Asset::Material(handle) => self.materials.revision(self.id, handle.0),
            Asset::Texture(handle) => self.textures.revision(self.id, handle.0),
        }
    }

    pub fn material(&self, handle: MaterialHandle) -> Option<&Material> {
        self.materials.get(self.id, handle.0)
    }

    pub fn texture(&self, handle: TextureHandle) -> Option<&Arc<Texture>> {
        self.textures.get(self.id, handle.0)
    }

    pub fn remove_mesh(&mut self, handle: GeometryHandle) -> bool {
        let Some(mesh) = self.meshes.remove(self.id, handle.0) else {
            return false;
        };
        self.resident_bytes -= mesh.map_or(0, |mesh| mesh.byte_len());
        self.bump();
        true
    }

    pub fn remove_material(&mut self, handle: MaterialHandle) -> bool {
        if self.materials.remove(self.id, handle.0).is_none() {
            return false;
        }
        self.resident_bytes -= size_of::<Material>();
        self.bump();
        true
    }

    pub fn remove_texture(&mut self, handle: TextureHandle) -> bool {
        let Some(texture) = self.textures.remove(self.id, handle.0) else {
            return false;
        };
        let bytes = texture.map_or(0, |texture| texture.byte_len());
        self.resident_bytes -= bytes;
        self.texture_bytes -= bytes;
        self.bump();
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
        self.meshes.values()
    }

    pub fn materials(&self) -> impl Iterator<Item = (usize, &Material)> {
        self.materials.values()
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

    fn reserve_bytes(&mut self, requested: usize) -> Result<(), Error> {
        let available = self
            .budget
            .max_cpu_bytes
            .saturating_sub(self.resident_bytes);
        if requested > available {
            return Err(full("$.cpu_bytes", available).got(requested));
        }
        self.resident_bytes += requested;
        Ok(())
    }

    fn bump(&mut self) -> u64 {
        self.revision = self.revision.saturating_add(1);
        self.revision
    }
}
