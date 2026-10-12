//! What a document needs of the engine's features: the plugins whose
//! components and sections it authors.

use engine_core::bevy_ecs::prelude::World;
use engine_core::{EnvironmentSection, Feature, has_feature};
use engine_settings::ErrorCode;

use crate::{SceneDocument, SceneEnvironment, SceneError, SceneObject, SectionVisitor};

/// This module's part of `entity_components!`: the features an object
/// needs for the components listed, after its terrain's.
macro_rules! object_features {
    ($(
        $(#[$doc:meta])*
        $field:ident: $type:ty { feature: $feature:ident, shows: $shows:literal, direct },
    )*) => {
        impl SceneObject {
            /// The features this object authors, by field.
            fn features(&self) -> impl Iterator<Item = (&'static str, Feature)> {
                [
                    ("terrain", self.terrain.is_some(), Some(Feature::Terrain)),
                    $((
                        stringify!($field),
                        self.$field.is_some(),
                        crate::components::component_feature!($feature),
                    ),)*
                    ("animation", self.animation.is_some(), Some(Feature::Animation)),
                    ("skin", self.skin.is_some(), Some(Feature::Animation)),
                    ("collider", self.collider.is_some(), Some(Feature::Collision)),
                ]
                .into_iter()
                .filter_map(|(field, authored, feature)| {
                    feature.filter(|_| authored).map(|feature| (field, feature))
                })
            }
        }
    };
}

crate::components::entity_components!(object_features);

/// The error of `path` authoring `feature`, whose plugin the engine lacks.
fn missing(feature: Feature, path: String) -> SceneError {
    SceneError::new(
        ErrorCode::MissingPlugin,
        path,
        format!(
            "{} is not added: add it, or EnginePlugins, to the engine",
            feature.plugin()
        ),
    )
}

impl SceneDocument {
    /// Fails at the first component, streamed list or environment section
    /// the document authors whose feature `world`'s engine lacks.
    pub(crate) fn check_features(&self, world: &World) -> Result<(), SceneError> {
        let lists =
            [("$.entities".to_owned(), &self.entities)]
                .into_iter()
                .chain(self.prefabs.iter().enumerate().map(|(index, prefab)| {
                    (format!("$.prefabs[{index}].entities"), &prefab.entities)
                }))
                .chain([("$.streamed".to_owned(), &self.streamed)]);
        for (list, objects) in lists {
            for (index, object) in objects.iter().enumerate() {
                if let Some((field, feature)) = object
                    .features()
                    .find(|&(_, feature)| !has_feature(world, feature))
                {
                    return Err(missing(feature, format!("{list}[{index}].{field}")));
                }
            }
        }
        if !self.streamed.is_empty() && !has_feature(world, Feature::WorldStreaming) {
            return Err(missing(Feature::WorldStreaming, "$.streamed".into()));
        }
        if let Some(environment) = &self.environment {
            environment.check_features(world, "$.environment")?;
        }
        Ok(())
    }
}

impl SceneEnvironment {
    /// Fails at the first section it names, under `path`, whose feature
    /// `world`'s engine lacks.
    pub fn check_features(&self, world: &World, path: &str) -> Result<(), SceneError> {
        struct Lacking<'w>(&'w World, Option<(&'static str, Feature)>);
        impl SectionVisitor for Lacking<'_> {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, section: Option<T>) {
                if self.1.is_none()
                    && section.is_some()
                    && let Some(feature) = T::FEATURE
                    && !has_feature(self.0, feature)
                {
                    self.1 = Some((name, feature));
                }
            }
        }
        let mut lacking = Lacking(world, None);
        self.visit(&mut lacking);
        match lacking.1 {
            Some((name, feature)) => Err(missing(feature, format!("{path}.{name}"))),
            None => Ok(()),
        }
    }
}
