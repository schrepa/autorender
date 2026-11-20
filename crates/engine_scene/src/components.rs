use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use std::sync::Arc;

use engine_core::{Assets, ComponentKind, Error, Prepared, SceneCx, scene_components};
use engine_registry::{Registry, SettingsEntry};
use engine_settings::ErrorCode;
use serde_json::Value;

use crate::SceneObject;

#[derive(Clone)]
pub struct SceneKinds {
    listed: Vec<ComponentKind>,
    sections: Vec<SettingsEntry>,
    assets: Arc<Assets>,
}

pub(crate) struct Component {
    kind: ComponentKind,
    at: String,
    value: Prepared,
}

impl SceneKinds {
    pub fn of(world: &World) -> Self {
        Self {
            listed: scene_components(world).cloned().collect(),
            sections: world
                .get_resource::<Registry>()
                .into_iter()
                .flat_map(Registry::sections)
                .copied()
                .collect(),
            assets: Arc::new(
                world
                    .get_resource::<Assets>()
                    .map_or_else(Assets::default, Assets::fresh),
            ),
        }
    }

    pub(crate) fn section(&self, key: &str) -> Option<SettingsEntry> {
        self.sections
            .iter()
            .find(|section| section.key == key)
            .copied()
    }

    pub(crate) fn scratch(&self) -> Assets {
        self.assets.fresh()
    }

    pub fn keys(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.listed.iter().map(ComponentKind::key)
    }

    fn find(&self, key: &str) -> Option<&ComponentKind> {
        self.listed.iter().find(|kind| kind.key() == key)
    }

    pub(crate) fn prepare(
        &self,
        object: &SceneObject,
        path: &str,
        cx: &SceneCx<'_>,
    ) -> Result<Vec<Component>, Error> {
        let mut components = Vec::new();
        for (name, value) in object.authored() {
            let Some(kind) = self.find(name) else {
                continue;
            };
            let at = format!("{path}.{name}");
            let value = kind.prepare_value(value, cx, &at)?;
            components.push(Component {
                kind: kind.clone(),
                at,
                value,
            });
        }
        Ok(components)
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
        cx: &SceneCx<'_>,
    ) -> Result<(), Error> {
        for component in self.prepare(object, path, cx)? {
            component.insert(world, entity)?;
        }
        Ok(())
    }
}

impl Component {
    pub(crate) fn insert(self, world: &mut World, entity: Entity) -> Result<(), Error> {
        self.kind
            .insert_prepared(world, entity, self.value)
            .map_err(|error| error.within(&self.at))
    }
}

impl SceneObject {
    pub(crate) fn authored(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.components
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(name, value)| (name.as_str(), value))
    }
}

#[cfg(feature = "dev")]
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

#[cfg(feature = "dev")]
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
