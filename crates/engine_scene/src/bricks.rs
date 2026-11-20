use bevy_ecs::world::World;
use engine_core::Error;

use crate::{SceneDocument, SceneKinds};

impl SceneDocument {
    pub(crate) fn unlisted(&self, world: &World, kinds: &SceneKinds) -> Vec<Error> {
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
        if let Some(environment) = &self.environment {
            skipped.extend(environment.unlisted(world, "$.environment"));
        }
        skipped
    }
}
