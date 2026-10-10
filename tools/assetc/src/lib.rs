//! Native conversion of GLB models with glTF 2.0 metallic-roughness
//! materials, animations and skins into scenes and the files beside them.
//! No filesystem access, runtime imports, or
//! external resource resolution.

mod animation;
mod geometry;
mod materials;
mod validation;

use std::{error::Error, fmt};

use engine_scene::{SCENE_VERSION, SceneCamera, SceneDocument, SceneError, ScenePrefab};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Largest source container: the GLB length field's range.
pub const MAX_INPUT_BYTES: usize = u32::MAX as usize;
/// Changes whenever import behavior or supported settings change.
pub const IMPORTER_VERSION: &str = "gltf-pbr-9";
/// Triangles from which an opaque primitive imports as virtual geometry
/// unless [`ImportOptions`] say otherwise.
pub const DEFAULT_VIRTUAL_TRIANGLES: u32 = 512;

/// How a model imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportOptions {
    /// Primitives of at least this many triangles, opaque and without vertex
    /// colors, second texture coordinates or joints, import as virtual
    /// meshes (`EVG1` files); `None` imports none.
    pub virtual_triangles: Option<u32>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            virtual_triangles: Some(DEFAULT_VIRTUAL_TRIANGLES),
        }
    }
}

/// A file the imported scene references, at `uri` relative to the scene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedFile {
    pub uri: String,
    pub bytes: Vec<u8>,
}

impl ImportedFile {
    /// `ID/NAME.HASH.EXTENSION`: named by content, so a new import never
    /// overwrites a file an existing scene reads.
    pub(crate) fn named(id: &str, name: &str, extension: &str, bytes: Vec<u8>) -> Self {
        let hash = format!("{:x}", Sha256::digest(&bytes));
        Self {
            uri: format!("{id}/{name}.{}.{extension}", &hash[..12]),
            bytes,
        }
    }
}

/// An imported scene document and the image and mesh-buffer files it
/// references, which the caller writes beside the scene.
#[derive(Debug, Clone)]
pub struct Imported {
    pub document: SceneDocument,
    pub files: Vec<ImportedFile>,
    /// Optional glTF extensions the source uses that the import ignores.
    pub ignored_extensions: Vec<String>,
}

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
        Self::new(error.code.as_str(), error.path, error.message)
    }
}

/// SHA-256 of the exact source bytes, for build provenance (not authentication).
pub fn source_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Imports one GLB scene, preserving primitive material mapping. Static
/// models bake node transforms into their vertices; models with animations
/// or skins keep their nodes as entities (`ID.node<N>`) below a root
/// (`ID.root`) that plays the first animation (`ID.anim<N>`), with skins
/// `ID.skin<N>`. IDs derive from source indices and the caller's namespace;
/// files are named `ID/image<N>.<HASH>.png|jpg`, `ID/node<N>.p<M>.<HASH>.bin`
/// (`.evg` for virtual meshes) and `ID/anim<N>.<HASH>.bin`. The resulting
/// document is validated with the runtime's scene validation before being
/// returned. Nothing is written. Imports with the default [`ImportOptions`].
pub fn import_glb(bytes: &[u8], id: &str) -> Result<Imported, ImportError> {
    import_glb_with(bytes, id, &ImportOptions::default())
}

/// [`import_glb`] with `options`.
pub fn import_glb_with(
    bytes: &[u8],
    id: &str,
    options: &ImportOptions,
) -> Result<Imported, ImportError> {
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
    let ignored_extensions = validation::structure(&json)?;
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
        environment: None,
        textures: Vec::new(),
        materials: Vec::new(),
        meshes: Vec::new(),
        prefabs: Vec::new(),
        animations: Vec::new(),
        skins: Vec::new(),
        entities: Vec::new(),
        streamed: Vec::new(),
    };
    let mut files = Vec::new();
    materials::import(&gltf.document, &json, binary, id, &mut scene, &mut files)?;
    geometry::import(&gltf.document, binary, id, options, &mut scene, &mut files)?;
    // These cooked artifacts use the exact authoring representation games load.
    scene.to_json()?;
    Ok(Imported {
        document: scene,
        files,
        ignored_extensions,
    })
}

/// Imports a reusable prefab definition and its shared global assets, without
/// placing an instance in the scene. Source transforms are baked or kept just
/// as in [`import_glb`]. Template IDs, and the animation targets and skin
/// joints naming them, omit the namespace, so placement roots can qualify
/// them independently without duplicating mesh, texture or clip data.
pub fn import_glb_prefab(bytes: &[u8], id: &str) -> Result<Imported, ImportError> {
    import_glb_prefab_with(bytes, id, &ImportOptions::default())
}

/// [`import_glb_prefab`] with `options`.
pub fn import_glb_prefab_with(
    bytes: &[u8],
    id: &str,
    options: &ImportOptions,
) -> Result<Imported, ImportError> {
    let mut imported = import_glb_with(bytes, id, options)?;
    let scene = &mut imported.document;
    let prefix = format!("{id}.");
    let local = |value: &mut String| {
        *value = value
            .strip_prefix(&prefix)
            .expect("imported entity IDs use the requested namespace")
            .to_owned();
    };
    let mut entities = std::mem::take(&mut scene.entities);
    for entity in &mut entities {
        local(&mut entity.id);
        if let Some(parent) = &mut entity.parent {
            local(parent);
        }
    }
    for target in scene
        .animations
        .iter_mut()
        .flat_map(|animation| &mut animation.targets)
        .chain(scene.skins.iter_mut().flat_map(|skin| &mut skin.joints))
    {
        local(target);
    }
    scene.prefabs.push(ScenePrefab {
        id: id.to_owned(),
        entities,
    });
    scene.to_json()?;
    Ok(imported)
}

pub(crate) fn unsupported(path: impl Into<String>, message: impl Into<String>) -> ImportError {
    ImportError::new("unsupported_feature", path, message)
}
