use std::collections::{BTreeMap, BTreeSet};

use engine_core::{Error, Transform, glam};
use engine_settings::{ErrorCode, Expected, InvalidSetting, Rule, Setting, valid_uri};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::validation::{identifier, unique_ids};
use crate::{SceneEnvironment, SceneMaterial, SceneTexture};

pub const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "dev", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneDocument {
    #[cfg_attr(
        feature = "dev",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub id: String,
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "dev",
        schemars(
            length(min = 1, max = 256),
            regex(pattern = r"^[A-Za-z0-9_-][A-Za-z0-9_./-]*$")
        )
    )]
    cooked: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<SceneEnvironment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<SceneTexture>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<SceneMaterial>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub meshes: Vec<SceneMesh>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefabs: Vec<ScenePrefab>,
    pub entities: Vec<SceneObject>,
    #[serde(skip)]
    pub(crate) folder: String,
}

#[derive(Deserialize)]
struct Cooked {
    items: CookedItems,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CookedItems {
    textures: Vec<SceneTexture>,
    materials: Vec<SceneMaterial>,
    meshes: Vec<SceneMesh>,
    prefabs: Vec<ScenePrefab>,
    entities: Vec<SceneObject>,
}

impl SceneDocument {
    pub fn from_json(source: &str) -> Result<Self, Error> {
        sized(source, "$")?;
        let document: Self = engine_settings::parse_json(source)?;
        document.ids()?;
        Ok(document)
    }

    pub fn cooked(&self) -> Option<&str> {
        self.cooked.as_deref()
    }

    pub fn append_cooked(&mut self, source: &str) -> Result<(), Error> {
        let uri = self.cooked.as_deref().ok_or_else(|| {
            Error::new(ErrorCode::MissingField, "$.cooked").expected(Expected::Present)
        })?;
        valid_uri(uri, "$.cooked")?;
        sized(source, "$.cooked")?;
        let folder = uri.rfind('/').map_or("", |at| &uri[..=at]).to_owned();
        let Cooked { items } =
            engine_settings::parse_json(source).map_err(|error| error.within("$.cooked"))?;
        let ids = |section: &str| format!("$.cooked.items.{section}");
        fresh(
            self.textures.iter().map(|texture| texture.id.as_str()),
            items.textures.iter().map(|texture| texture.id.as_str()),
            &ids("textures"),
        )?;
        fresh(
            self.materials.iter().map(|material| material.id.as_str()),
            items.materials.iter().map(|material| material.id.as_str()),
            &ids("materials"),
        )?;
        fresh(
            self.meshes.iter().map(|mesh| mesh.id.as_str()),
            items.meshes.iter().map(|mesh| mesh.id.as_str()),
            &ids("meshes"),
        )?;
        fresh(
            self.prefabs.iter().map(|prefab| prefab.id.as_str()),
            items.prefabs.iter().map(|prefab| prefab.id.as_str()),
            &ids("prefabs"),
        )?;
        fresh(
            self.entities.iter().map(|entity| entity.id.as_str()),
            items.entities.iter().map(|entity| entity.id.as_str()),
            &ids("entities"),
        )?;
        self.textures
            .extend(items.textures.into_iter().map(|mut texture| {
                texture.within(&folder);
                texture
            }));
        self.meshes.extend(items.meshes.into_iter().map(|mesh| SceneMesh {
            uri: format!("{folder}{}", mesh.uri),
            ..mesh
        }));
        self.materials.extend(items.materials);
        self.prefabs.extend(items.prefabs.into_iter().map(|mut prefab| {
            prefab.entities.iter_mut().for_each(|entity| entity.cooked = true);
            prefab
        }));
        self.entities.extend(items.entities.into_iter().map(|mut entity| {
            entity.cooked = true;
            entity
        }));
        self.folder = folder;
        Ok(())
    }

    pub(crate) fn ids(&self) -> Result<(), Error> {
        identifier(&self.id, "$.id")?;
        unique_ids(
            self.textures.iter().map(|texture| texture.id.as_str()),
            "$.textures",
        )?;
        unique_ids(
            self.materials.iter().map(|material| material.id.as_str()),
            "$.materials",
        )?;
        unique_ids(self.meshes.iter().map(|mesh| mesh.id.as_str()), "$.meshes")?;
        unique_ids(
            self.prefabs.iter().map(|prefab| prefab.id.as_str()),
            "$.prefabs",
        )?;
        Ok(())
    }

    #[cfg(feature = "dev")]
    pub fn schema_json(registry: &engine_registry::Registry, kinds: &crate::SceneKinds) -> String {
        let mut schema =
            serde_json::to_value(schemars::schema_for!(Self)).expect("schema_serializes");
        if let Some(definitions) = schema
            .get_mut("$defs")
            .and_then(serde_json::Value::as_object_mut)
        {
            let sections = crate::environment::section_schemas(registry, definitions);
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

fn sized(source: &str, path: &str) -> Result<(), Error> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(Error::new(ErrorCode::SourceTooLarge, path)
            .at_most(MAX_SOURCE_BYTES as u64)
            .got(source.len()));
    }
    Ok(())
}

fn fresh<'a>(
    held: impl Iterator<Item = &'a str>,
    added: impl Iterator<Item = &'a str>,
    section: &str,
) -> Result<(), Error> {
    let mut known: BTreeSet<&str> = held.collect();
    for (index, id) in added.enumerate() {
        let path = format!("{section}[{index}].id");
        identifier(id, &path)?;
        if !known.insert(id) {
            return Err(Error::new(ErrorCode::DuplicateId, path)
                .expected(Expected::Unique)
                .got(id));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "dev", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneMesh {
    #[cfg_attr(
        feature = "dev",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub id: String,
    #[cfg_attr(
        feature = "dev",
        schemars(
            length(min = 1, max = 256),
            regex(pattern = r"^[A-Za-z0-9_-][A-Za-z0-9_./-]*$")
        )
    )]
    pub uri: String,
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
    #[serde(skip)]
    pub(crate) cooked: bool,
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
