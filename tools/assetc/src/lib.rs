mod geometry;
mod materials;
mod validation;

use engine_core::{Error, ErrorCode, Expected};
use engine_scene::{SceneCamera, SceneDocument, ScenePrefab};
use sha2::{Digest, Sha256};

pub const MAX_INPUT_BYTES: usize = u32::MAX as usize;
pub const IMPORTER_VERSION: &str = "gltf-pbr-10";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedFile {
    pub uri: String,
    pub bytes: Vec<u8>,
}

impl ImportedFile {
    pub(crate) fn named(id: &str, name: &str, extension: &str, bytes: Vec<u8>) -> Self {
        let hash = format!("{:x}", Sha256::digest(&bytes));
        Self {
            uri: format!("{id}/{name}.{}.{extension}", &hash[..12]),
            bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Imported {
    pub document: SceneDocument,
    pub files: Vec<ImportedFile>,
    pub ignored_extensions: Vec<String>,
}

pub fn source_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub const MAX_ID_BYTES: usize = 40;

pub fn import_glb(bytes: &[u8], id: &str) -> Result<Imported, Error> {
    if id.is_empty()
        || id.len() > MAX_ID_BYTES
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(Error::new(ErrorCode::InvalidId, "$.id")
            .expected(Expected::Id)
            .at_most(MAX_ID_BYTES as u64)
            .got(id));
    }
    let json = validation::container(bytes)?;
    let ignored_extensions = validation::structure(&json)?;
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|_| gltf_error("$"))?;
    let binary = gltf
        .blob
        .as_deref()
        .ok_or_else(|| gltf_error("$.buffers[0]").expected(Expected::Present))?;
    validation::buffers(&gltf.document, binary)?;
    let mut scene = SceneDocument {
        id: id.to_owned(),
        revision: 1,
        camera: SceneCamera::default(),
        environment: None,
        textures: Vec::new(),
        materials: Vec::new(),
        meshes: Vec::new(),
        prefabs: Vec::new(),
        entities: Vec::new(),
        streamed: Vec::new(),
    };
    let mut files = Vec::new();
    materials::import(&gltf.document, &json, binary, id, &mut scene, &mut files)?;
    geometry::import(&gltf.document, binary, id, &mut scene, &mut files)?;
    scene.to_json()?;
    Ok(Imported {
        document: scene,
        files,
        ignored_extensions,
    })
}

pub fn import_glb_prefab(bytes: &[u8], id: &str) -> Result<Imported, Error> {
    let mut imported = import_glb(bytes, id)?;
    let scene = &mut imported.document;
    let prefix = format!("{id}.");
    let local = |value: &mut String| {
        *value = value
            .strip_prefix(&prefix)
            .expect("entity_id_in_namespace")
            .to_owned();
    };
    let mut entities = std::mem::take(&mut scene.entities);
    for entity in &mut entities {
        local(&mut entity.id);
        if let Some(parent) = &mut entity.parent {
            local(parent);
        }
    }
    scene.prefabs.push(ScenePrefab {
        id: id.to_owned(),
        entities,
    });
    scene.to_json()?;
    Ok(imported)
}

pub(crate) fn unsupported(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedFeature, path)
}

pub(crate) fn gltf_error(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidGltf, path)
}
