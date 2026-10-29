use engine_core::bevy_ecs::prelude::World;
use engine_core::{BrickRegistry, Error, missing_brick};

use crate::{SceneDocument, SceneObject};

macro_rules! object_bricks {
    ($(
        $(#[$doc:meta])*
        $field:ident: $type:ty {
            brick: $brick:expr,
            shows: $shows:literal,
            rule: $rule:ident $(($($arg:expr),*))?,
            $kind:ident
        },
    )*) => {
        impl SceneObject {
            fn bricks(&self) -> impl Iterator<Item = (&'static str, &'static str)> {
                [$((stringify!($field), self.$field.is_some(), $brick),)*]
                    .into_iter()
                    .filter_map(|(field, authored, brick)| {
                        brick.filter(|_| authored).map(|brick| (field, brick))
                    })
            }
        }
    };
}

crate::components::entity_components!(object_bricks);

fn added(world: &World, name: &str) -> bool {
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
                if let Some((field, brick)) =
                    object.bricks().find(|&(_, brick)| !added(world, brick))
                {
                    return Err(missing_brick(brick, format!("{list}[{index}].{field}")));
                }
                if let Some(terrain) = &object.terrain {
                    terrain.check_layers(world, &format!("{list}[{index}].terrain"))?;
                }
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
