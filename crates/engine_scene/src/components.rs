use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{ComponentKind, Error, SceneComponents};
use engine_registry::MembersOf;
use engine_settings::ErrorCode;
use serde_json::Value;

use crate::SceneObject;

pub(crate) struct Kinds {
    listed: Vec<ComponentKind>,
}

impl Kinds {
    pub(crate) fn of(world: &World) -> Self {
        Self {
            listed: world
                .get_resource::<MembersOf<SceneComponents>>()
                .map_or_else(Vec::new, |members| members.all().to_vec()),
        }
    }

    fn find(&self, key: &str) -> Option<&ComponentKind> {
        self.listed.iter().find(|kind| kind.key() == key)
    }

    fn keys(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.listed.iter().map(ComponentKind::key)
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
            (kind.scene().0)(world, entity, value)
                .map_err(|error| error.within(&format!("{path}.{name}")))?;
        }
        Ok(())
    }
}

impl SceneObject {
    pub fn component(&self, name: &str) -> Option<&Value> {
        self.components.get(name).filter(|value| !value.is_null())
    }

    fn authored(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.components
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(name, value)| (name.as_str(), value))
    }
}
