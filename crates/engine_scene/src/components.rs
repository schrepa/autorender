use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{ComponentKind, Error, Placed, SceneComponents, ScenePath};
use engine_registry::MembersOf;
use engine_settings::ErrorCode;
use serde_json::Value;

use crate::SceneObject;

#[derive(Clone, Default)]
pub struct SceneKinds {
    listed: Vec<ComponentKind>,
}

impl SceneKinds {
    pub fn of(world: &World) -> Self {
        Self {
            listed: world
                .get_resource::<MembersOf<SceneComponents>>()
                .map_or_else(Vec::new, |members| members.all().to_vec()),
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.listed.iter().map(ComponentKind::key)
    }

    fn find(&self, key: &str) -> Option<&ComponentKind> {
        self.listed.iter().find(|kind| kind.key() == key)
    }

    fn each_known(
        &self,
        object: &SceneObject,
        path: &str,
        mut visit: impl FnMut(&ComponentKind, &Value, &str) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for (name, value) in object.authored() {
            if let Some(kind) = self.find(name) {
                visit(kind, value, &format!("{path}.{name}"))?;
            }
        }
        Ok(())
    }

    pub(crate) fn check(&self, object: &SceneObject, path: &str) -> Result<(), Error> {
        self.each_known(object, path, |kind, value, at| kind.validate(value, at))
    }

    pub(crate) fn check_placed(
        &self,
        object: &SceneObject,
        placed: Placed,
        path: &str,
    ) -> Result<(), Error> {
        self.each_known(object, path, |kind, value, at| kind.validate_placed(value, placed, at))
    }

    pub(crate) fn paths(&self, object: &SceneObject, path: &str) -> Result<Vec<ScenePath>, Error> {
        let mut paths = Vec::new();
        self.each_known(object, path, |kind, value, at| {
            paths.extend(kind.scene_paths(value, at)?);
            Ok(())
        })?;
        Ok(paths)
    }

    pub(crate) fn files<'a>(&'a self, object: &'a SceneObject) -> impl Iterator<Item = String> + 'a {
        object
            .authored()
            .filter_map(|(name, value)| self.find(name).map(|kind| kind.listed_files(value)))
            .flatten()
    }

    pub(crate) fn unlisted(&self, object: &SceneObject, path: &str, skipped: &mut Vec<Error>) {
        for (name, _) in object.authored() {
            if self.find(name).is_none() {
                skipped.push(
                    Error::new(ErrorCode::SectionUnlisted, format!("{path}.{name}"))
                        .got(name)
                        .variants(self.keys()),
                );
            }
        }
    }

    pub(crate) fn read(
        &self,
        world: &mut World,
        entity: Entity,
        object: &SceneObject,
        path: &str,
    ) -> Result<(), Error> {
        for (name, value) in object.authored() {
            let Some(kind) = self.find(name) else {
                continue;
            };
            kind.read(world, entity, value)
                .map_err(|error| error.within(&format!("{path}.{name}")))?;
        }
        Ok(())
    }
}

impl SceneObject {
    pub fn component(&self, name: &str) -> Option<&Value> {
        self.components.get(name).filter(|value| !value.is_null())
    }

    pub(crate) fn authored(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.components
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(name, value)| (name.as_str(), value))
    }
}

#[cfg(feature = "schema")]
pub(crate) fn component_schemas(
    kinds: &SceneKinds,
    definitions: &mut serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    kinds
        .listed
        .iter()
        .map(|kind| (kind.key().to_owned(), component_property(kind, definitions)))
        .collect()
}

#[cfg(feature = "schema")]
fn component_property(
    kind: &ComponentKind,
    definitions: &mut serde_json::Map<String, Value>,
) -> Value {
    let mut schema = kind.json_schema();
    let Some(fields) = schema.as_object_mut() else {
        return Value::Bool(true);
    };
    fields.remove("$schema");
    if let Some(Value::Object(nested)) = fields.remove("$defs") {
        for (name, definition) in nested {
            definitions.entry(name).or_insert(definition);
        }
    }
    let listed = fields.get("type").and_then(Value::as_str) == Some("array");
    let title = fields
        .remove("title")
        .and_then(|title| title.as_str().map(str::to_owned));
    let shape = match title {
        Some(title) if !listed => {
            definitions.insert(title.clone(), schema);
            serde_json::json!({ "$ref": format!("#/$defs/{title}") })
        }
        Some(_) | None => schema,
    };
    serde_json::json!({ "anyOf": [shape, { "type": "null" }] })
}
