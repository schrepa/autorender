use std::collections::BTreeMap;

use engine_core::{Camera, Error, Mesh, MeshVertex, Projection, Surface, Transform, glam};
use engine_settings::{ErrorCode, Expected, InvalidSetting, Rule, Setting, fields};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{SceneEnvironment, SceneFiles, SceneMaterial, SceneTexture, files::valid_uri};

pub const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneDocument {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub id: String,
    pub revision: u64,
    #[serde(default)]
    pub camera: SceneCamera,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<SceneEnvironment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<SceneTexture>,
    pub materials: Vec<SceneMaterial>,
    pub meshes: Vec<SceneMesh>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefabs: Vec<ScenePrefab>,
    pub entities: Vec<SceneObject>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<crate::SceneView>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub looks: Vec<crate::SceneLook>,
}

impl SceneDocument {
    pub fn from_json(source: &str) -> Result<Self, Error> {
        if source.len() > MAX_SOURCE_BYTES {
            return Err(Error::new(ErrorCode::SourceTooLarge, "$")
                .at_most(MAX_SOURCE_BYTES as u64)
                .got(source.len()));
        }
        let document: Self = crate::parse_json(source)?;
        document.structure()?;
        Ok(document)
    }

    pub fn to_json(&self) -> Result<String, Error> {
        self.structure()?;
        let mut output = BoundedSource {
            bytes: Vec::new(),
            exceeded: false,
        };
        let result = serde_json::to_writer_pretty(&mut output, self);
        if output.exceeded {
            return Err(source_too_large());
        }
        result.map_err(|_| Error::new(ErrorCode::InvalidJson, "$"))?;
        if output.bytes.len() == MAX_SOURCE_BYTES {
            return Err(source_too_large());
        }
        output.bytes.push(b'\n');
        Ok(String::from_utf8(output.bytes).expect("json_utf8"))
    }

    #[cfg(feature = "schema")]
    pub fn schema_json(registry: &engine_registry::Registry, kinds: &crate::SceneKinds) -> String {
        let mut schema =
            serde_json::to_value(schemars::schema_for!(Self)).expect("schema_serializes");
        if let Some(definitions) = schema
            .get_mut("$defs")
            .and_then(serde_json::Value::as_object_mut)
        {
            let sections = crate::section_schemas(registry, definitions);
            if let Some(properties) = definitions
                .get_mut("SceneEnvironment")
                .and_then(|environment| environment.get_mut("properties"))
                .and_then(serde_json::Value::as_object_mut)
            {
                properties.extend(sections);
            }
            let components = crate::components::component_schemas(kinds, definitions);
            if let Some(object) = definitions
                .get_mut("SceneObject")
                .and_then(serde_json::Value::as_object_mut)
            {
                if let Some(properties) = object
                    .get_mut("properties")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    properties.extend(components);
                }
                object.insert("additionalProperties".to_owned(), Value::Bool(false));
            }
        }
        let mut source = serde_json::to_string_pretty(&schema).expect("schema_serializes");
        source.push('\n');
        source
    }
}

fn source_too_large() -> Error {
    Error::new(ErrorCode::SourceTooLarge, "$").at_most(MAX_SOURCE_BYTES as u64)
}

struct BoundedSource {
    bytes: Vec<u8>,
    exceeded: bool,
}

impl std::io::Write for BoundedSource {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_SOURCE_BYTES - self.bytes.len() {
            self.exceeded = true;
            return Err(std::io::ErrorKind::FileTooLarge.into());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn default_camera() -> SceneCamera {
    let camera = Camera::default();
    let Projection::Perspective {
        vertical_fov_radians,
    } = camera.projection
    else {
        unreachable!("default_camera_perspective");
    };
    SceneCamera {
        position: camera.position.to_array(),
        target: camera.target.to_array(),
        fov_degrees: vertical_fov_radians.to_degrees(),
        near: camera.near,
        far: camera.far,
    }
}

engine_settings::fields! {
    extra_rules;
    #[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneCamera {
        pub position: [f32; 3] = default_camera().position => range(-Camera::REACH, Camera::REACH),
        pub target: [f32; 3] = default_camera().target => range(-Camera::REACH, Camera::REACH),
        pub fov_degrees: f32 = default_camera().fov_degrees
            => range(Camera::MIN_FOV_DEGREES, Camera::MAX_FOV_DEGREES),
        pub near: f32 = default_camera().near => range(Camera::MIN_DEPTH, Camera::REACH),
        pub far: f32 = default_camera().far => range(Camera::MIN_DEPTH, Camera::REACH),
    }
}

impl engine_settings::ExtraRules for SceneCamera {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        let camera = self.camera();
        if (camera.target - camera.position).length_squared()
            < Camera::MIN_DEPTH * Camera::MIN_DEPTH
        {
            return Err(
                InvalidSetting::broken("target", fields::at_least(Camera::MIN_DEPTH))
                    .got((camera.target - camera.position).length()),
            );
        }
        if camera.far - camera.near < Camera::MIN_DEPTH {
            return Err(InvalidSetting::broken(
                "far",
                fields::range(camera.near + Camera::MIN_DEPTH, Camera::REACH),
            )
            .got(camera.far));
        }
        Ok(())
    }
}

impl SceneCamera {
    pub(crate) fn camera(self) -> Camera {
        Camera {
            position: self.position.into(),
            target: self.target.into(),
            near: self.near,
            far: self.far,
            projection: Projection::Perspective {
                vertical_fov_radians: self.fov_degrees.to_radians(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = Surface::state_rules)
)]
#[serde(default, deny_unknown_fields)]
pub struct SceneSurface {
    pub roughness: f32,
    pub metalness: f32,
    pub emission: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneMesh {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub id: String,
    pub source: MeshSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeshSource {
    Cube {},
    Plane {},
    Octahedron {},
    Inline {
        #[cfg_attr(feature = "schema", schemars(length(min = 1)))]
        vertices: Vec<SceneVertex>,
        #[cfg_attr(feature = "schema", schemars(length(min = 3)))]
        indices: Vec<u32>,
    },
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
    Provided {
        #[cfg_attr(
            feature = "schema",
            schemars(transform = engine_settings::stated(engine_settings::fields::id()))
        )]
        name: String,
    },
}

impl MeshSource {
    pub(crate) fn provided(&self) -> Option<&str> {
        match self {
            Self::Provided { name } => Some(name),
            Self::Cube {}
            | Self::Plane {}
            | Self::Octahedron {}
            | Self::Inline { .. }
            | Self::Buffer { .. } => None,
        }
    }

    pub(crate) fn mesh(&self, path: &str, files: Option<&SceneFiles>) -> Result<Mesh, Error> {
        match self {
            Self::Cube {} => Ok(Mesh::cube()),
            Self::Plane {} => Ok(Mesh::plane()),
            Self::Octahedron {} => Ok(Mesh::octahedron()),
            Self::Provided { name } => {
                crate::validation::identifier(name, &format!("{path}.name"))?;
                Ok(Mesh::cube())
            }
            Self::Buffer { uri } => {
                let uri_path = format!("{path}.uri");
                valid_uri(uri, &uri_path)?;
                let Some(files) = files else {
                    return Ok(Mesh::cube());
                };
                files.mesh(uri, &uri_path)?.mesh(path)
            }
            Self::Inline { vertices, indices } => {
                let extras = |present: &dyn Fn(&SceneVertex) -> bool, field: &str| {
                    let count = vertices.iter().filter(|vertex| present(vertex)).count();
                    match vertices.iter().position(|vertex| !present(vertex)) {
                        Some(lacking) if count != 0 => Err(Error::new(
                            ErrorCode::InvalidMesh,
                            format!("{path}.vertices[{lacking}].{field}"),
                        )
                        .expected(Expected::Present)),
                        Some(_) | None => Ok(count != 0),
                    }
                };
                let colored = extras(&|vertex| vertex.color.is_some(), "color")?;
                let second = extras(&|vertex| vertex.uv1.is_some(), "uv1")?;
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
                    path,
                )
            }
        }
    }
}

pub(crate) fn with_extras(
    mesh: Result<Mesh, Error>,
    colors: Option<Vec<[f32; 4]>>,
    uv1: Option<Vec<[f32; 2]>>,
    path: &str,
) -> Result<Mesh, Error> {
    let invalid = |error: Error| error.coded(ErrorCode::InvalidMesh).within(path);
    let mut mesh = mesh.map_err(invalid)?;
    if let Some(colors) = colors {
        mesh = mesh.with_colors(colors).map_err(invalid)?;
    }
    if let Some(uv1) = uv1 {
        mesh = mesh.with_uv1(uv1).map_err(invalid)?;
    }
    Ok(mesh)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    #[serde(default, skip_serializing_if = "zero_uv")]
    pub uv: [f32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv1: Option<[f32; 2]>,
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

engine_settings::fields! {
    #[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct ScenePrefab {
        pub id: String = String::new() => id,
        pub entities: Vec<SceneObject> = Vec::new() => count(0, 10_000),
    }
}

const TAGS: Rule = Rule::count(0.0, 16.0);

#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(transform = state_object_rules)]
pub struct SceneObject {
    #[schemars(transform = engine_settings::stated(engine_settings::fields::id()))]
    pub id: String,
    #[serde(default)]
    #[schemars(length(max = crate::validation::MAX_NAME_BYTES))]
    pub name: String,
    #[schemars(transform = engine_settings::stated(engine_settings::fields::id()))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<String>,
    #[schemars(transform = engine_settings::stated(engine_settings::fields::id()))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(transform = engine_settings::stated(engine_settings::fields::id()))]
    pub parent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(transform = engine_settings::stated(engine_settings::fields::id()))]
    pub prefab: Option<String>,
    #[serde(flatten)]
    #[schemars(skip)]
    pub components: BTreeMap<String, Value>,
    #[serde(default)]
    pub transform: SceneTransform,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn state_object_rules(schema: &mut schemars::Schema) {
    engine_settings::state(schema, "tags", TAGS);
}

impl SceneObject {
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        Setting::check(&self.transform, "transform", Rule::Any)?;
        Setting::check(&self.tags, "tags", TAGS)
    }
}

impl Setting for SceneObject {
    fn check(&self, field: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        self.validate().map_err(|invalid| invalid.within(field))
    }
}

engine_settings::fields! {
    #[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneTransform {
        pub translation: [f32; 3] = [0.0; 3] => range(-1.0e6, 1.0e6),
        pub rotation_degrees: [f32; 3] = [0.0; 3] => range(-360_000.0, 360_000.0),
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
