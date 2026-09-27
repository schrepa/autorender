use engine_core::{Camera, Mesh, MeshVertex, Surface, Transform, glam};
use serde::{Deserialize, Serialize};

use crate::{
    SceneCollider, SceneEmitter, SceneEnvironment, SceneError, SceneFiles, SceneLight,
    SceneMaterial, SceneTexture, files::valid_uri,
};

/// Accepted authoring format version. Changes require an explicit migration.
pub const SCENE_VERSION: u32 = 8;
/// Largest scene source parsed: a sanity bound on JSON, not on content,
/// which references image and buffer files for bulk data (decision 030).
pub(crate) const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;

/// A persistent scene source. IDs are authored ASCII names, never runtime IDs.
///
/// Loading sets no count budget (decision 029); bulk pixels and geometry
/// belong in files beside the scene (version 7). Prefabs share scene-wide
/// assets across their placements.
/// Unknown fields are rejected to expose authoring mistakes. Arrays retain
/// authored order when serialized, yielding stable, readable source diffs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneDocument {
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 8)))]
    pub version: u32,
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub id: String,
    pub revision: u64,
    #[serde(default)]
    pub camera: SceneCamera,
    /// The scene's atmosphere and rendering effects (decision 022). Requires
    /// version 5. With it the scene owns the whole look: instantiation sets
    /// every section, omitted ones to the engine defaults. Without it the
    /// game's own settings stay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<SceneEnvironment>,
    /// Opaque sRGB pixels: inline base64, or image files (version 7).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<SceneTexture>,
    pub materials: Vec<SceneMaterial>,
    pub meshes: Vec<SceneMesh>,
    /// Reusable local entity trees. Definitions share this document's assets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefabs: Vec<ScenePrefab>,
    pub entities: Vec<SceneObject>,
}

impl SceneDocument {
    /// Parses and validates bounded authoring input without changing any world.
    pub fn from_json(source: &str) -> Result<Self, SceneError> {
        if source.len() > MAX_SOURCE_BYTES {
            return Err(SceneError::new(
                "source_too_large",
                "$",
                format!("scene source must be at most {MAX_SOURCE_BYTES} bytes"),
            ));
        }
        let document: Self = serde_json::from_str(source)
            .map_err(|error| SceneError::new("invalid_json", "$", error.to_string()))?;
        document.validate()?;
        Ok(document)
    }

    /// Moves this document to [`SCENE_VERSION`] without changing what it
    /// describes: from before version 8, plain texture IDs become maps that
    /// clamp at the texture's edges, as those versions sample them.
    pub fn upgrade(&mut self) {
        if self.version < 8 {
            for material in &mut self.materials {
                if let Some(crate::SceneMap::Id(id)) = &material.texture {
                    material.texture = Some(crate::SceneMap::Map(crate::SceneTextureMap {
                        id: id.clone(),
                        wrap_u: Some(crate::SceneWrap::ClampToEdge),
                        wrap_v: Some(crate::SceneWrap::ClampToEdge),
                        mag_filter: None,
                        min_filter: None,
                        mipmap_filter: None,
                        offset: None,
                        rotation: None,
                        scale: None,
                    }));
                }
            }
        }
        self.version = SCENE_VERSION;
    }

    /// Writes validated, pretty JSON with stable field order and a final newline.
    ///
    /// This only returns source text; the caller owns persistence and revisions.
    pub fn to_json(&self) -> Result<String, SceneError> {
        self.validate()?;
        let mut output = BoundedSource {
            bytes: Vec::new(),
            exceeded: false,
        };
        let result = serde_json::to_writer_pretty(&mut output, self);
        if output.exceeded {
            return Err(source_too_large());
        }
        result.map_err(|error| SceneError::new("invalid_json", "$", error.to_string()))?;
        if output.bytes.len() == MAX_SOURCE_BYTES {
            return Err(source_too_large());
        }
        output.bytes.push(b'\n');
        Ok(String::from_utf8(output.bytes).expect("JSON serializer writes UTF-8"))
    }

    /// Generates structural metadata for authoring tools in host builds.
    ///
    /// Runtime [`Self::validate`] also enforces referential integrity, finite
    /// data, budgets and unique IDs. The optional `schema` feature keeps schema
    /// generation dependencies out of ordinary browser builds.
    #[cfg(feature = "schema")]
    pub fn schema_json() -> String {
        let schema = schemars::schema_for!(Self);
        // Schema construction emits only JSON values, so serialization cannot
        // encounter unsupported map keys or nonfinite floating point values.
        let mut source =
            serde_json::to_string_pretty(&schema).expect("JSON schema is serializable");
        source.push('\n');
        source
    }
}

fn source_too_large() -> SceneError {
    SceneError::new(
        "source_too_large",
        "$",
        format!("canonical scene source exceeds {MAX_SOURCE_BYTES} bytes"),
    )
}

// Stop serialization at the transport budget instead of first allocating a
// potentially much larger string for a Rust-assembled or imported scene.
struct BoundedSource {
    bytes: Vec<u8>,
    exceeded: bool,
}

impl std::io::Write for BoundedSource {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_SOURCE_BYTES - self.bytes.len() {
            self.exceeded = true;
            return Err(std::io::Error::other("scene source budget exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Right-handed, Y-up perspective camera. Lens values are in world units.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneCamera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    #[cfg_attr(feature = "schema", schemars(range(min = 1.0, max = 179.0)))]
    pub fov_degrees: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001, max = 1_000_000.0)))]
    pub near: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001, max = 1_000_000.0)))]
    pub far: f32,
}

impl Default for SceneCamera {
    fn default() -> Self {
        let camera = Camera::default();
        Self {
            position: camera.position.to_array(),
            target: camera.target.to_array(),
            fov_degrees: camera.vertical_fov_radians.to_degrees(),
            near: camera.near,
            far: camera.far,
        }
    }
}

impl SceneCamera {
    pub(crate) fn camera(self) -> Camera {
        Camera {
            position: self.position.into(),
            target: self.target.into(),
            vertical_fov_radians: self.fov_degrees.to_radians(),
            near: self.near,
            far: self.far,
        }
    }
}

/// A material's response to light (decision 019). Omitted fields keep the
/// matte defaults: roughness 0.8 and no metalness, emission or ripples.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneSurface {
    /// 0 is a mirror, 1 spreads highlights and sky reflections widest.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub roughness: f32,
    /// Metals reflect in their own color and diffuse no light.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub metalness: f32,
    /// Emitted light as a multiple of the material color.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 16.0)))]
    pub emission: f32,
    /// Animated ripples of liquid surfaces.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub ripples: f32,
}

impl Default for SceneSurface {
    fn default() -> Self {
        let matte = Surface::MATTE;
        Self {
            roughness: matte.roughness,
            metalness: matte.metalness,
            emission: matte.emission,
            ripples: matte.ripples,
        }
    }
}

impl SceneSurface {
    pub(crate) fn surface(self) -> Surface {
        Surface {
            roughness: self.roughness,
            metalness: self.metalness,
            emission: self.emission,
            ripples: self.ripples,
        }
    }
}

/// A named CPU mesh. Sources contain no filesystem paths or executable data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneMesh {
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub id: String,
    pub source: MeshSource,
}

/// Supported geometry recipes, including arbitrary validated triangle meshes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeshSource {
    Cube {},
    /// Centered unit square in the XZ plane, facing +Y.
    Plane {},
    Octahedron {},
    Inline {
        #[cfg_attr(feature = "schema", schemars(length(min = 1)))]
        vertices: Vec<SceneVertex>,
        #[cfg_attr(feature = "schema", schemars(length(min = 3)))]
        indices: Vec<u32>,
    },
    /// An engine mesh buffer file (`EMB1`) beside the scene. Version 7.
    Buffer {
        #[cfg_attr(
            feature = "schema",
            schemars(
                length(min = 1, max = 256),
                regex(pattern = r"^[A-Za-z0-9_-][A-Za-z0-9_./-]*$")
            )
        )]
        uri: String,
    },
}

impl MeshSource {
    /// The mesh at authored path `path`. Without `files` (structural
    /// validation) a buffer mesh reads as the cube.
    pub(crate) fn mesh(&self, path: &str, files: Option<&SceneFiles>) -> Result<Mesh, SceneError> {
        match self {
            Self::Cube {} => Ok(Mesh::cube()),
            Self::Plane {} => Ok(Mesh::plane()),
            Self::Octahedron {} => Ok(Mesh::octahedron()),
            Self::Buffer { uri } => {
                let uri_path = format!("{path}.uri");
                valid_uri(uri, &uri_path)?;
                let Some(files) = files else {
                    return Ok(Mesh::cube());
                };
                let (vertices, indices) = files.mesh(uri, &uri_path)?;
                Mesh::new(vertices, indices)
                    .map_err(|error| SceneError::new("invalid_mesh", path, error.to_string()))
            }
            Self::Inline { vertices, indices } => Mesh::new(
                vertices
                    .iter()
                    .map(|vertex| MeshVertex {
                        position: vertex.position,
                        normal: vertex.normal,
                        uv: vertex.uv,
                    })
                    .collect(),
                indices.clone(),
            )
            .map_err(|error| SceneError::new("invalid_mesh", path, error.to_string())),
        }
    }
}

/// CPU vertex authoring data; the renderer owns its separate upload layout.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Top-left origin texture coordinates, sampled with clamp-to-edge.
    #[serde(default, skip_serializing_if = "zero_uv")]
    pub uv: [f32; 2],
}

impl From<SceneVertex> for MeshVertex {
    fn from(vertex: SceneVertex) -> Self {
        Self {
            position: vertex.position,
            normal: vertex.normal,
            uv: vertex.uv,
        }
    }
}

fn zero_uv(uv: &[f32; 2]) -> bool {
    *uv == [0.0; 2]
}

/// A reusable entity tree whose parents reference IDs within the definition.
///
/// Instance roots use [`SceneObject::prefab`]. Template roots become children of
/// that instance root. Nested prefab references are deliberately unsupported.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ScenePrefab {
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub id: String,
    #[cfg_attr(feature = "schema", schemars(length(max = 10_000)))]
    pub entities: Vec<SceneObject>,
}

/// One local transform, optional renderable, or prefab instance root.
/// Tags are game-defined startup hook names.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneObject {
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub id: String,
    /// Optional display label, at most 128 UTF-8 bytes without control characters.
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 128)))]
    pub name: String,
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<String>,
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    /// Parent ID in the scene, or in this prefab's local definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub parent: Option<String>,
    /// Named prefab to expand below this transform-only instance root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub prefab: Option<String>,
    /// Local collision shape, independent of rendering. Requires version 4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider: Option<SceneCollider>,
    /// A point or spot light (decision 023). Requires version 6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light: Option<SceneLight>,
    /// A particle emitter (decision 026). Requires version 6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub particles: Option<SceneEmitter>,
    /// Multiplies this object's color and opacity channel by channel
    /// (decision 033), each factor finite and in `[0, 1]`. Requires version 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tint: Option<[f32; 4]>,
    #[serde(default)]
    pub transform: SceneTransform,
    /// At most 16 distinct authored IDs, each at most 64 ASCII bytes.
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 16)))]
    pub tags: Vec<String>,
}

/// Local transform with Euler XYZ rotation expressed in degrees.
///
/// Translation is bounded to ±1,000,000 world units and each scale component
/// to `[0.0001, 10000]`, avoiding singular or overflowing normal transforms.
/// Parent composition preserves full matrices, including shear. Negative scale
/// reflection is not authored by this format.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneTransform {
    pub translation: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub scale: [f32; 3],
}

impl Default for SceneTransform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            scale: [1.0; 3],
        }
    }
}

impl SceneTransform {
    pub(crate) fn transform(self) -> Transform {
        let [x, y, z] = self.rotation_degrees.map(f32::to_radians);
        Transform {
            translation: self.translation.into(),
            rotation: glam::Quat::from_euler(glam::EulerRot::XYZ, x, y, z),
            scale: self.scale.into(),
        }
    }
}
