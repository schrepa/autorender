use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::Error;
use engine_registry::{Registry, SettingsEntry};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SceneEnvironment {
    pub sections: BTreeMap<String, Value>,
}

impl SceneEnvironment {
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.sections
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(key, _)| key.as_str())
    }

    pub fn value(&self, key: &str) -> Option<&Value> {
        self.sections.get(key).filter(|value| !value.is_null())
    }

    pub fn unlisted(&self, world: &World, path: &str) -> Vec<Error> {
        self.sections
            .iter()
            .filter(|(key, value)| !value.is_null() && listed(world, key).is_none())
            .map(|(key, _)| {
                Error::new(ErrorCode::SectionUnlisted, format!("{path}.{key}"))
                    .got(key)
                    .variants(registered(world).map(|section| section.key))
            })
            .collect()
    }

    pub(crate) fn apply_all(&self, world: &mut World, path: &str) -> Result<(), Error> {
        let sections: Vec<SettingsEntry> = registered(world).copied().collect();
        for section in sections {
            let value = self
                .value(section.key)
                .cloned()
                .unwrap_or_else(|| section.default_value());
            section.set_at(world, &value, &format!("{path}.{}", section.key))?;
        }
        Ok(())
    }
}

fn registered(world: &World) -> impl Iterator<Item = &SettingsEntry> {
    world
        .get_resource::<Registry>()
        .into_iter()
        .flat_map(Registry::sections)
}

fn listed(world: &World, key: &str) -> Option<SettingsEntry> {
    world.get_resource::<Registry>()?.section(key).copied()
}

#[cfg(feature = "dev")]
impl schemars::JsonSchema for SceneEnvironment {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SceneEnvironment".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        })
    }
}

#[cfg(feature = "dev")]
pub(crate) fn section_schemas(
    registry: &Registry,
    definitions: &mut serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    let mut properties = serde_json::Map::new();
    for section in registry.sections() {
        let mut schema =
            serde_json::to_value(registry.section_schema(section.key)).expect("schema_serializes");
        let Some(fields) = schema.as_object_mut() else {
            continue;
        };
        fields.remove("$schema");
        if let Some(Value::Object(nested)) = fields.remove("$defs") {
            for (name, definition) in nested {
                definitions.entry(name).or_insert(definition);
            }
        }
        let title = fields
            .remove("title")
            .and_then(|title| title.as_str().map(str::to_owned))
            .unwrap_or_else(|| {
                let name = section.type_name;
                name.rsplit("::").next().unwrap_or(name).to_owned()
            });
        definitions.insert(title.clone(), schema);
        properties.insert(
            section.key.to_owned(),
            serde_json::json!({
                "anyOf": [{ "$ref": format!("#/$defs/{title}") }, { "type": "null" }],
            }),
        );
    }
    properties
}
