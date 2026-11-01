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
    pub fn current(world: &World) -> Self {
        let mut sections = BTreeMap::new();
        for section in registered(world) {
            sections.insert(section.key.to_owned(), section.current(world));
        }
        Self { sections }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.sections
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(key, _)| key.as_str())
    }

    pub fn ordered_keys(&self, world: &World) -> Vec<String> {
        let mut keys: Vec<String> = Vec::new();
        for section in registered(world) {
            if self.value(section.key).is_some() {
                keys.push(section.key.to_owned());
            }
        }
        for key in self.keys() {
            if !keys.iter().any(|known| known == key) {
                keys.push(key.to_owned());
            }
        }
        keys
    }

    pub fn value(&self, key: &str) -> Option<&Value> {
        self.sections.get(key).filter(|value| !value.is_null())
    }

    pub fn check(&self, world: &World, path: &str) -> Result<(), Error> {
        self.check_in(world.get_resource::<Registry>(), path)
    }

    pub(crate) fn check_in(&self, registry: Option<&Registry>, path: &str) -> Result<(), Error> {
        for key in self.keys() {
            if let Some(section) = registry.and_then(|registry| registry.section(key)) {
                section.check(&self.sections[key], &format!("{path}.{key}"))?;
            }
        }
        Ok(())
    }

    pub fn unlisted(&self, world: &World, path: &str) -> Vec<Error> {
        self.keys()
            .filter(|key| listed(world, key).is_none())
            .map(|key| {
                Error::new(ErrorCode::SectionUnlisted, format!("{path}.{key}"))
                    .got(key)
                    .variants(registered(world).map(|section| section.key))
            })
            .collect()
    }

    pub fn apply_sections(&self, world: &mut World, path: &str) -> Result<Vec<Error>, Error> {
        self.check(world, path)?;
        let skipped = self.unlisted(world, path);
        for key in self.keys() {
            let Some(section) = listed(world, key) else {
                continue;
            };
            section.set_at(world, &self.sections[key], &format!("{path}.{key}"))?;
        }
        Ok(skipped)
    }

    pub(crate) fn apply_all(&self, world: &mut World, path: &str) -> Result<(), Error> {
        self.apply_where(world, path, |_| true)
    }

    pub(crate) fn apply_where(
        &self,
        world: &mut World,
        path: &str,
        applies: impl Fn(&str) -> bool,
    ) -> Result<(), Error> {
        let sections: Vec<SettingsEntry> = registered(world)
            .filter(|section| applies(section.key))
            .copied()
            .collect();
        for section in sections {
            match self.value(section.key) {
                Some(value) => {
                    section.set_at(world, value, &format!("{path}.{}", section.key))?;
                }
                None => section.reset(world),
            }
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

#[cfg(feature = "schema")]
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

#[cfg(feature = "schema")]
pub fn section_schemas(
    registry: &Registry,
    definitions: &mut serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    let mut properties = serde_json::Map::new();
    for section in registry.sections() {
        let mut schema = serde_json::to_value(section.schema()).expect("schema_serializes");
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
