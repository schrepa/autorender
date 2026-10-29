use std::collections::BTreeMap;

use bevy_ecs::prelude::Resource;

use crate::{MaterialHandle, MeshHandle, TextureHandle};

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct ProvidedAssets {
    meshes: BTreeMap<String, MeshHandle>,
    textures: BTreeMap<String, TextureHandle>,
    materials: BTreeMap<String, MaterialHandle>,
}

impl ProvidedAssets {
    pub fn provide_mesh(&mut self, name: impl Into<String>, mesh: MeshHandle) -> &mut Self {
        self.meshes.insert(name.into(), mesh);
        self
    }

    pub fn provide_texture(
        &mut self,
        name: impl Into<String>,
        texture: TextureHandle,
    ) -> &mut Self {
        self.textures.insert(name.into(), texture);
        self
    }

    pub fn provide_material(
        &mut self,
        name: impl Into<String>,
        material: MaterialHandle,
    ) -> &mut Self {
        self.materials.insert(name.into(), material);
        self
    }

    pub fn mesh(&self, name: &str) -> Option<MeshHandle> {
        self.meshes.get(name).copied()
    }

    pub fn texture(&self, name: &str) -> Option<TextureHandle> {
        self.textures.get(name).copied()
    }

    pub fn material(&self, name: &str) -> Option<MaterialHandle> {
        self.materials.get(name).copied()
    }

    pub fn mesh_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.meshes.keys().map(String::as_str)
    }

    pub fn texture_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.textures.keys().map(String::as_str)
    }

    pub fn material_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.materials.keys().map(String::as_str)
    }
}
