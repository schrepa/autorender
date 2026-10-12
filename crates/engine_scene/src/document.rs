use engine_core::{Camera, Mesh, MeshVertex, Surface, Transform, glam};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

use crate::{
    SceneAnimation, SceneCollider, SceneEmitter, SceneEnvironment, SceneError, SceneFiles,
    SceneLight, SceneMaterial, ScenePlayer, SceneSkin, SceneTexture, files::valid_uri,
};

/// Largest scene source parsed: a sanity bound on JSON, not on content,
/// which references image and buffer files for bulk data.
pub const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;

/// A persistent scene source. IDs are authored ASCII names, never runtime IDs.
///
/// Loading sets no count budget; bulk pixels and geometry
/// belong in files beside the scene. Prefabs share scene-wide
/// assets across their placements.
/// Unknown fields are rejected to expose authoring mistakes. Arrays retain
/// authored order when serialized, yielding stable, readable source diffs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneDocument {
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub id: String,
    pub revision: u64,
    #[serde(default)]
    pub camera: SceneCamera,
    /// The scene's atmosphere and rendering effects. With it the scene
    /// owns the whole look: instantiation sets every section, omitted ones
    /// to the engine defaults. Without it the game's own settings stay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<SceneEnvironment>,
    /// Opaque sRGB pixels: inline base64, or image files.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<SceneTexture>,
    pub materials: Vec<SceneMaterial>,
    pub meshes: Vec<SceneMesh>,
    /// Reusable local entity trees. Definitions share this document's assets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefabs: Vec<ScenePrefab>,
    /// Keyframed clips that entity `animation` players play.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub animations: Vec<SceneAnimation>,
    /// Joint lists that deform skinned meshes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skins: Vec<SceneSkin>,
    pub entities: Vec<SceneObject>,
    /// Entities that exist only near the camera: each root
    /// with its descendants is one item, which world streaming spawns when
    /// its cell comes within range and despawns when it leaves. Parents,
    /// animation targets and skin joints stay within an item; IDs differ
    /// from `entities`'. No terrains.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub streamed: Vec<SceneObject>,
}

impl SceneDocument {
    /// Parses and validates bounded authoring input without changing any world.
    pub fn from_json(source: &str) -> Result<Self, SceneError> {
        if source.len() > MAX_SOURCE_BYTES {
            return Err(SceneError::new(
                ErrorCode::SourceTooLarge,
                "$",
                format!("scene source must be at most {MAX_SOURCE_BYTES} bytes"),
            ));
        }
        let document: Self = crate::parse_json(source)?;
        document.validate()?;
        Ok(document)
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
        result.map_err(|error| SceneError::new(ErrorCode::InvalidJson, "$", error.to_string()))?;
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
        ErrorCode::SourceTooLarge,
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

/// The engine's default camera, as authored.
fn default_camera() -> SceneCamera {
    let camera = Camera::default();
    SceneCamera {
        position: camera.position.to_array(),
        target: camera.target.to_array(),
        fov_degrees: camera.vertical_fov_radians.to_degrees(),
        near: camera.near,
        far: camera.far,
    }
}

engine_settings::settings! {
    extra_rules;
    /// Right-handed, Y-up perspective camera. Lens values are in world units.
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneCamera {
        /// Where the camera stands.
        pub position: [f32; 3] = default_camera().position => range(-1.0e6, 1.0e6),
        /// What it looks at, at least 0.001 from its position.
        pub target: [f32; 3] = default_camera().target => range(-1.0e6, 1.0e6),
        /// Vertical field of view in degrees.
        pub fov_degrees: f32 = default_camera().fov_degrees => range(1.0, 179.0),
        /// The near plane's distance.
        pub near: f32 = default_camera().near => range(0.001, 1.0e6),
        /// The far plane's distance, at least 0.001 beyond the near plane's.
        pub far: f32 = default_camera().far => range(0.001, 1.0e6),
    }
}

impl engine_settings::ExtraRules for SceneCamera {
    /// It looks somewhere, through some depth.
    fn check_extra(&self) -> Result<(), engine_settings::InvalidSetting> {
        let camera = self.camera();
        if (camera.target - camera.position).length_squared() < 0.000_001 {
            return Err(engine_settings::InvalidSetting::new(
                "target",
                "must be at least 0.001 units from the position",
            ));
        }
        if self.far - self.near < 0.001 {
            return Err(engine_settings::InvalidSetting::new(
                "far",
                "must be at least 0.001 beyond near",
            ));
        }
        Ok(())
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

/// A material's response to light. Omitted fields keep the
/// matte defaults: roughness 0.8 and no metalness, emission or ripples.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = Surface::state_rules)
)]
#[serde(default, deny_unknown_fields)]
pub struct SceneSurface {
    /// 0 is a mirror, 1 spreads highlights and sky reflections widest.
    pub roughness: f32,
    /// Metals reflect in their own color and diffuse no light.
    pub metalness: f32,
    /// Emitted light as a multiple of the material color.
    pub emission: f32,
    /// Animated ripples of liquid surfaces.
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
    /// An engine mesh buffer file (`EMB1`) beside the scene.
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
    /// A virtual mesh file (`EVG1`) beside the scene: a
    /// cluster hierarchy renderers draw as the clusters each view needs, and
    /// elsewhere as its most detailed cut of at most `fallback_triangles`
    /// triangles (4096 when omitted). Built by `cargo xtask asset import`
    /// for dense meshes.
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
                let buffer = files.mesh(uri, &uri_path)?;
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
                        ErrorCode::InvalidMesh,
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
                .map_err(|error| {
                    SceneError::new(ErrorCode::InvalidMesh, uri_path, error.to_string())
                })
            }
            Self::Inline { vertices, indices } => {
                // Vertices carry colors and second coordinates all or none.
                let extras = |present: &dyn Fn(&SceneVertex) -> bool, what: &str| {
                    let count = vertices.iter().filter(|vertex| present(vertex)).count();
                    if count != 0 && count != vertices.len() {
                        Err(SceneError::new(
                            ErrorCode::InvalidMesh,
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
                        ErrorCode::InvalidMesh,
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
    let invalid = |error: engine_core::AssetError| {
        SceneError::new(ErrorCode::InvalidMesh, path, error.to_string())
    };
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
    /// Linear RGBA in `[0, 1]` multiplying the material's base
    /// color. Every vertex of a mesh has one or none does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 4]>,
    /// Second texture coordinates, which texture maps may read.
    /// Every vertex of a mesh has them or none does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv1: Option<[f32; 2]>,
    /// Up to four joints (indices into the skin of the entities
    /// drawing the mesh) moving this vertex. Every vertex of
    /// a mesh has joints and weights or none does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joints: Option<[u16; 4]>,
    /// The joints' weights, not negative and not all zero; they
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

engine_settings::settings! {
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
        pub id: String = String::new() => any,
        /// Optional display label, at most 128 UTF-8 bytes without control characters.
        #[serde(default)]
        #[cfg_attr(feature = "schema", schemars(length(max = 128)))]
        pub name: String = String::new() => any,
        #[cfg_attr(
            feature = "schema",
            schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
        )]
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub mesh: Option<String> = None => any,
        #[cfg_attr(
            feature = "schema",
            schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
        )]
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub material: Option<String> = None => any,
        /// Parent ID in the scene, or in this prefab's local definition.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(
            feature = "schema",
            schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
        )]
        pub parent: Option<String> = None => any,
        /// Named prefab to expand below this transform-only instance root.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(
            feature = "schema",
            schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
        )]
        pub prefab: Option<String> = None => any,
        /// Local collision shape, independent of rendering.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub collider: Option<SceneCollider> = None => any,
        /// A point or spot light.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub light: Option<SceneLight> = None => any,
        /// A particle emitter.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub particles: Option<SceneEmitter> = None => any,
        /// Multiplies this object's color and opacity channel by channel,
        /// each factor finite and in `[0, 1]`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub tint: Option<[f32; 4]> = None => range(0.0, 1.0),
        /// Plays an animation on its targets.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub animation: Option<ScenePlayer> = None => any,
        /// The skin deforming this object's mesh. Requires a mesh with
        /// joints and weights.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(
            feature = "schema",
            schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
        )]
        pub skin: Option<String> = None => any,
        /// Projects a material onto the opaque surfaces inside this entity's
        /// box.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub decal: Option<crate::SceneDecal> = None => any,
        /// Captures this entity's surroundings into an environment map that
        /// surfaces inside its world box reflect.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub reflection_probe: Option<crate::SceneReflectionProbe> = None => any,
        /// A heightfield terrain whose tiles stream around the camera.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub terrain: Option<crate::SceneTerrain> = None => any,
        /// One of this document's meshes bent along a curve.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub spline: Option<crate::SceneSpline> = None => any,
        /// Rides the sea.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub floating: Option<engine_core::Floating> = None => any,
        /// Presses tracks into deformable ground.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub deformer: Option<engine_core::Deformer> = None => any,
        /// Walks over the terrain to a destination. Roots only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub nav_agent: Option<engine_core::NavAgent> = None => any,
        /// Stands on the terrain: its translation's y becomes
        /// `offset` meters above the ground under it. Roots only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub grounded: Option<engine_core::Grounded> = None => any,
        /// Names it a place the camera may be sent to, whose
        /// terrain workers generate ahead of need.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub prefetch: Option<engine_core::TerrainPrefetch> = None => any,
        #[serde(default)]
        pub transform: SceneTransform = SceneTransform::default() => any,
        /// At most 16 distinct authored IDs, each at most 64 ASCII bytes.
        #[serde(default)]
        #[cfg_attr(feature = "schema", schemars(length(max = 16)))]
        pub tags: Vec<String> = Vec::new() => any,
    }
}

engine_settings::settings! {
    /// Local transform with Euler XYZ rotation expressed in degrees.
    ///
    /// Bounded translations and scales avoid singular or overflowing normal
    /// transforms. Parent composition preserves full matrices, including
    /// shear. Negative scale reflection is not authored by this format.
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneTransform {
        /// Local translation, world units.
        pub translation: [f32; 3] = [0.0; 3] => range(-1.0e6, 1.0e6),
        /// Euler XYZ rotation, degrees.
        pub rotation_degrees: [f32; 3] = [0.0; 3] => range(-360_000.0, 360_000.0),
        /// Scale on each local axis.
        pub scale: [f32; 3] = [1.0; 3] => range(0.0001, 10_000.0),
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
