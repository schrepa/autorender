use std::collections::BTreeMap;

use engine_assets::MeshBuffer;
use engine_core::{Error, Mesh, MeshVertex, Transform, glam};
use engine_settings::{ErrorCode, Expected, InvalidSetting, Rule, Setting, valid_uri};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{SceneEnvironment, SceneFiles, SceneMaterial, SceneTexture};

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
            Self::Cube {} | Self::Plane {} | Self::Inline { .. } | Self::Buffer { .. } => None,
        }
    }

    pub(crate) fn mesh(&self, path: &str, files: Option<&SceneFiles>) -> Result<Mesh, Error> {
        match self {
            Self::Cube {} => Ok(Mesh::cube()),
            Self::Plane {} => Ok(Mesh::plane()),
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
                MeshBuffer {
                    vertices: vertices.iter().copied().map(MeshVertex::from).collect(),
                    indices: indices.clone(),
                    colors: colored.then(|| {
                        vertices
                            .iter()
                            .map(|vertex| vertex.color.unwrap_or([1.0; 4]))
                            .collect()
                    }),
                    uv1: second.then(|| {
                        vertices
                            .iter()
                            .map(|vertex| vertex.uv1.unwrap_or([0.0; 2]))
                            .collect()
                    }),
                    named: Vec::new(),
                    lods: None,
                }
                .mesh(path)
            }
        }
    }
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
        pub entities: Vec<SceneObject> = Vec::new() => any,
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
