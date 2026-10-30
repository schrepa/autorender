use engine_core::bevy_ecs::prelude::World;
use engine_core::{BrickRegistry, Error, missing_brick};

use crate::SceneDocument;

pub(crate) fn added(world: &World, name: &str) -> bool {
    world
        .get_resource::<BrickRegistry>()
        .is_some_and(|registry| registry.has(name))
}

impl SceneDocument {
    pub(crate) fn check_bricks(&self, world: &World) -> Result<(), Error> {
        let lists =
            [("$.entities".to_owned(), &self.entities)]
                .into_iter()
                .chain(self.prefabs.iter().enumerate().map(|(index, prefab)| {
                    (format!("$.prefabs[{index}].entities"), &prefab.entities)
                }))
                .chain([("$.streamed".to_owned(), &self.streamed)]);
        for (list, objects) in lists {
            for (index, object) in objects.iter().enumerate() {
                object.admit_components(world, &format!("{list}[{index}]"))?;
            }
        }
        if !self.streamed.is_empty() && !added(world, "world_streaming") {
            return Err(missing_brick("world_streaming", "$.streamed"));
        }
        if let Some(index) = self.along.iter().position(|along| along.streamed.is_some())
            && !added(world, "world_streaming")
        {
            return Err(missing_brick(
                "world_streaming",
                format!("$.along[{index}].streamed"),
            ));
        }
        for (index, look) in self.looks.iter().enumerate() {
            look.environment
                .check(world, &format!("$.looks[{index}].environment"))?;
        }
        if let Some(environment) = &self.environment {
            environment.check(world, "$.environment")?;
        }
        Ok(())
    }
}
