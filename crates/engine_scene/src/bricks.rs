use bevy_ecs::world::World;
use engine_core::Error;
use engine_registry::Registry;
use engine_settings::ErrorCode;

use crate::{SceneDocument, SceneKinds, SceneObject, environment::STREAMED};

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
        if kinds.streamed().is_some() {
            for (index, value) in self.streamed.iter().enumerate() {
                if let Ok(object) = serde_json::from_value::<SceneObject>(value.clone()) {
                    kinds.unlisted(&object, &format!("$.streamed[{index}]"), &mut skipped);
                }
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
        if kinds.streamed().is_none() {
            for (field, values) in [("streamed", &self.streamed), ("along", &self.along)] {
                if !values.is_empty() {
                    skipped.push(
                        Error::new(ErrorCode::SectionUnlisted, format!("$.{field}"))
                            .got(STREAMED)
                            .variants(registry.into_iter().flat_map(|registry| {
                                registry.sections().map(|section| section.key)
                            })),
                    );
                }
            }
        }
        Ok(skipped)
    }
}
