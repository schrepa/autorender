//! Native, bounded conversion of a documented static GLB subset into scenes.
//! No filesystem access, runtime imports, or external resource resolution.

mod geometry;
mod textures;
mod validation;

use std::{error::Error, fmt};

use engine_scene::{SCENE_VERSION, SceneCamera, SceneDocument, SceneError, ScenePrefab};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Maximum source container size before parsing or copying.
pub const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;
/// Changes whenever import behavior or supported settings change.
pub const IMPORTER_VERSION: &str = "static-glb-3";
pub(crate) const MAX_CPU_BYTES: usize = 32 * 1024 * 1024;

/// Machine-readable failure; paths address the source glTF JSON where possible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImportError {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

impl ImportError {
    pub fn new(code: &'static str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at {}: {}",
            self.code, self.path, self.message
        )
    }
}

impl Error for ImportError {}

impl From<SceneError> for ImportError {
    fn from(error: SceneError) -> Self {
        Self::new(error.code, error.path, error.message)
    }
}

/// SHA-256 of the exact source bytes, for build provenance (not authentication).
pub fn source_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Imports one static GLB scene, baking node transforms and preserving primitive
/// material mapping. IDs derive from source indices and the caller's namespace.
/// The resulting document is validated with the runtime's scene validation and
/// canonical 1 MiB text limit before being returned. Nothing is written.
pub fn import_glb(bytes: &[u8], id: &str) -> Result<SceneDocument, ImportError> {
    if id.is_empty()
        || id.len() > 40
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(ImportError::new(
            "invalid_id",
            "--id",
            "import namespace must be 1 to 40 ASCII letters, digits, underscores, hyphens or periods",
        ));
    }
    let json = validation::container(bytes)?;
    validation::structure(&json)?;
    let gltf = gltf::Gltf::from_slice(bytes)
        .map_err(|error| ImportError::new("invalid_gltf", "$", error.to_string()))?;
    let binary = gltf
        .blob
        .as_deref()
        .ok_or_else(|| ImportError::new("missing_binary", "$", "a GLB BIN chunk is required"))?;
    validation::buffers(&gltf.document, binary)?;
    let mut scene = SceneDocument {
        version: SCENE_VERSION,
        id: id.to_owned(),
        revision: 1,
        camera: SceneCamera::default(),
        textures: Vec::new(),
        materials: Vec::new(),
        meshes: Vec::new(),
        prefabs: Vec::new(),
        entities: Vec::new(),
    };
    let mut resident_bytes = 0;
    textures::import(&gltf.document, binary, id, &mut scene, &mut resident_bytes)?;
    geometry::import(&gltf.document, binary, id, &mut scene, &mut resident_bytes)?;
    // The serialization limit matters as much as the decoded CPU budget: these
    // cooked artifacts use the exact authoring representation loaded by games.
    scene.to_json()?;
    Ok(scene)
}

/// Imports a reusable prefab definition and its shared global assets, without
/// placing an instance in the scene. Static source transforms remain baked just
/// as in [`import_glb`]. Template IDs omit the namespace, so placement roots can
/// qualify them independently without duplicating mesh or texture data.
pub fn import_glb_prefab(bytes: &[u8], id: &str) -> Result<SceneDocument, ImportError> {
    let mut scene = import_glb(bytes, id)?;
    let prefix = format!("{id}.");
    let mut entities = std::mem::take(&mut scene.entities);
    for entity in &mut entities {
        entity.id = entity
            .id
            .strip_prefix(&prefix)
            .expect("imported entity IDs use the requested namespace")
            .to_owned();
    }
    scene.prefabs.push(ScenePrefab {
        id: id.to_owned(),
        entities,
    });
    scene.to_json()?;
    Ok(scene)
}

pub(crate) fn reserve_bytes(
    used: &mut usize,
    additional: usize,
    path: &str,
) -> Result<(), ImportError> {
    *used = used
        .checked_add(additional)
        .filter(|next| *next <= MAX_CPU_BYTES)
        .ok_or_else(|| {
            ImportError::new(
                "asset_budget",
                path,
                "decoded texture and mesh data exceed the 32 MiB import budget",
            )
        })?;
    Ok(())
}

pub(crate) fn unsupported(path: impl Into<String>, message: impl Into<String>) -> ImportError {
    ImportError::new("unsupported_feature", path, message)
}

#[cfg(test)]
mod tests;
