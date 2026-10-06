use engine_core::{Camera, Mesh, MeshVertex, Surface, Transform, glam};
use serde::{Deserialize, Serialize};

use crate::{
    SceneAnimation, SceneCollider, SceneEmitter, SceneEnvironment, SceneError, SceneFiles,
    SceneLight, SceneMaterial, ScenePlayer, SceneSkin, SceneTexture, files::valid_uri,
};

/// Accepted authoring format version. Changes require an explicit migration.
pub const SCENE_VERSION: u32 = 9;

/// The repository path of the generated schema of [`SCENE_VERSION`].
pub fn schema_path() -> String {
    format!("docs/schema/scene-v{SCENE_VERSION}.schema.json")
}
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
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 9)))]
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
    /// Keyframed clips that entity `animation` players play (decision 035).
    /// Version 8.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub animations: Vec<SceneAnimation>,
    /// Joint lists that deform skinned meshes (decision 036). Version 8.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skins: Vec<SceneSkin>,
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
                        tex_coord: None,
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
    /// A virtual mesh file (`EVG1`, decision 044) beside the scene: a
    /// cluster hierarchy renderers draw as the clusters each view needs, and
    /// elsewhere as its most detailed cut of at most `fallback_triangles`
    /// triangles (4096 when omitted). Built by `cargo xtask asset import`
    /// for dense meshes. Version 8.
    Virtual {
        #[cfg_attr(
            feature = "schema",
            schemars(
                length(min = 1, max = 256),
                regex(pattern = r"^[A-Za-z0-9_-][A-Za-z0-9_./-]*$")
            )
        )]
        uri: String,
        #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fallback_triangles: Option<u32>,
    },
}

impl MeshSource {
    /// The mesh at authored path `path`. Without `files` (structural
    /// validation) a buffer mesh reads as the cube.
    /// The mesh at authored path `path` of a document of `version`.
    pub(crate) fn mesh(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        version: u32,
    ) -> Result<Mesh, SceneError> {
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
                let buffer = files.mesh(uri, &uri_path)?;
                if version < 8
                    && (buffer.colors.is_some() || buffer.uv1.is_some() || buffer.skin.is_some())
                {
                    return Err(crate::versions::version_required(
                        uri_path,
                        "vertex colors, second texture coordinates, joints and weights",
                        8,
                    ));
                }
                with_extras(
                    Mesh::new(buffer.vertices, buffer.indices),
                    buffer.colors,
                    buffer.uv1,
                    buffer.skin,
                    path,
                )
            }
            Self::Virtual {
                uri,
                fallback_triangles,
            } => {
                let uri_path = format!("{path}.uri");
                valid_uri(uri, &uri_path)?;
                if *fallback_triangles == Some(0) {
                    return Err(SceneError::new(
                        "invalid_mesh",
                        format!("{path}.fallback_triangles"),
                        "must be at least 1",
                    ));
                }
                let Some(files) = files else {
                    return Ok(Mesh::cube());
                };
                let geometry = files.virtual_mesh(uri, &uri_path)?;
                Mesh::from_virtual(
                    std::sync::Arc::new(geometry),
                    fallback_triangles.unwrap_or(engine_core::DEFAULT_FALLBACK_TRIANGLES),
                )
                .map_err(|error| SceneError::new("invalid_mesh", uri_path, error.to_string()))
            }
            Self::Inline { vertices, indices } => {
                // Vertices carry colors and second coordinates all or none.
                let extras = |present: &dyn Fn(&SceneVertex) -> bool, what: &str| {
                    let count = vertices.iter().filter(|vertex| present(vertex)).count();
                    if count != 0 && count != vertices.len() {
                        Err(SceneError::new(
                            "invalid_mesh",
                            format!("{path}.vertices"),
                            format!("either every vertex has {what} or none does"),
                        ))
                    } else {
                        Ok(count != 0)
                    }
                };
                let colored = extras(&|vertex| vertex.color.is_some(), "a color")?;
                let second = extras(&|vertex| vertex.uv1.is_some(), "uv1")?;
                let skinned = extras(
                    &|vertex| vertex.joints.is_some() && vertex.weights.is_some(),
                    "joints and weights",
                )? || vertices
                    .iter()
                    .any(|vertex| vertex.joints.is_some() != vertex.weights.is_some());
                if skinned
                    && vertices
                        .iter()
                        .any(|vertex| vertex.joints.is_none() || vertex.weights.is_none())
                {
                    return Err(SceneError::new(
                        "invalid_mesh",
                        format!("{path}.vertices"),
                        "either every vertex has joints and weights or none does",
                    ));
                }
                with_extras(
                    Mesh::new(
                        vertices.iter().copied().map(MeshVertex::from).collect(),
                        indices.clone(),
                    ),
                    colored.then(|| {
                        vertices
                            .iter()
                            .map(|vertex| vertex.color.unwrap_or([1.0; 4]))
                            .collect()
                    }),
                    second.then(|| {
                        vertices
                            .iter()
                            .map(|vertex| vertex.uv1.unwrap_or([0.0; 2]))
                            .collect()
                    }),
                    skinned.then(|| {
                        vertices
                            .iter()
                            .map(|vertex| {
                                (
                                    vertex.joints.unwrap_or_default(),
                                    vertex.weights.unwrap_or_default(),
                                )
                            })
                            .unzip()
                    }),
                    path,
                )
            }
        }
    }
}

/// `mesh` with optional vertex colors, second texture coordinates, and
/// joints and weights.
fn with_extras(
    mesh: Result<Mesh, engine_core::AssetError>,
    colors: Option<Vec<[f32; 4]>>,
    uv1: Option<Vec<[f32; 2]>>,
    skin: Option<crate::files::SkinStreams>,
    path: &str,
) -> Result<Mesh, SceneError> {
    let invalid =
        |error: engine_core::AssetError| SceneError::new("invalid_mesh", path, error.to_string());
    let mut mesh = mesh.map_err(invalid)?;
    if let Some(colors) = colors {
        mesh = mesh.with_colors(colors).map_err(invalid)?;
    }
    if let Some(uv1) = uv1 {
        mesh = mesh.with_uv1(uv1).map_err(invalid)?;
    }
    if let Some((joints, weights)) = skin {
        mesh = mesh.with_skin(joints, weights).map_err(invalid)?;
    }
    Ok(mesh)
}

impl MeshSource {
    /// Whether any inline vertex has a color, second coordinates, joints or
    /// weights.
    pub(crate) fn has_vertex_extras(&self) -> bool {
        matches!(self, Self::Inline { vertices, .. } if vertices.iter().any(|vertex| {
            vertex.color.is_some()
                || vertex.uv1.is_some()
                || vertex.joints.is_some()
                || vertex.weights.is_some()
        }))
    }
}

/// CPU vertex authoring data; the renderer owns its separate upload layout.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Top-left origin texture coordinates.
    #[serde(default, skip_serializing_if = "zero_uv")]
    pub uv: [f32; 2],
    /// Version 8: linear RGBA in `[0, 1]` multiplying the material's base
    /// color (decision 034). Every vertex of a mesh has one or none does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 4]>,
    /// Version 8: second texture coordinates, which texture maps may read.
    /// Every vertex of a mesh has them or none does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv1: Option<[f32; 2]>,
    /// Version 8: up to four joints (indices into the skin of the entities
    /// drawing the mesh) moving this vertex (decision 036). Every vertex of
    /// a mesh has joints and weights or none does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joints: Option<[u16; 4]>,
    /// Version 8: the joints' weights, not negative and not all zero; they
    /// are normalized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<[f32; 4]>,
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
    /// Plays an animation on its targets (decision 035). Requires version 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation: Option<ScenePlayer>,
    /// The skin deforming this object's mesh (decision 036). Requires
    /// version 8 and a mesh with joints and weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub skin: Option<String>,
    /// Projects a material onto the opaque surfaces inside this entity's
    /// box (decision 039). Requires version 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decal: Option<crate::SceneDecal>,
    /// Captures this entity's surroundings into an environment map that
    /// surfaces inside its world box reflect (decision 043). Requires
    /// version 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflection_probe: Option<crate::SceneReflectionProbe>,
    /// A heightfield terrain whose tiles stream around the camera (decision
    /// 051). Requires version 9.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terrain: Option<crate::SceneTerrain>,
    /// One of this document's meshes bent along a curve (decision 071).
    /// Requires version 9.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spline: Option<crate::SceneSpline>,
    /// Rides the sea (decision 077). Requires version 9.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floating: Option<engine_core::Floating>,
    /// Presses tracks into deformable ground (decision 083). Requires
    /// version 9.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deformer: Option<engine_core::Deformer>,
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
