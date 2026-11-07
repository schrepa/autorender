use bevy_ecs::world::World;
use engine_core::Error;
use engine_registry::Registry;

use crate::{SceneDocument, SceneKinds};

impl SceneDocument {
    pub(crate) fn admit(&self, world: &World, kinds: &SceneKinds) -> Result<Vec<Error>, Error> {
        let mut skipped = Vec::new();
        let lists =
            [("$.entities".to_owned(), &self.entities)]
                .into_iter()
                .chain(self.prefabs.iter().enumerate().map(|(index, prefab)| {
                    (format!("$.prefabs[{index}].entities"), &prefab.entities)
                }));
        for (list, objects) in lists {
            for (index, object) in objects.iter().enumerate() {
                kinds.unlisted(object, &format!("{list}[{index}]"), &mut skipped);
            }
        }
        let registry = world.get_resource::<Registry>();
        for (index, look) in self.looks.iter().enumerate() {
            let path = format!("$.looks[{index}].environment");
            look.environment.check_in(registry, &path)?;
            skipped.extend(look.environment.unlisted(world, &path));
        }
        if let Some(environment) = &self.environment {
            environment.check_in(registry, "$.environment")?;
            skipped.extend(environment.unlisted(world, "$.environment"));
        }
        Ok(skipped)
    }
}
