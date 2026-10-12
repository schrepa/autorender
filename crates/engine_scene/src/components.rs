//! The list of components scenes author on entities, and how each joins
//! the entity spawned for an object authoring it.

use engine_core::bevy_ecs::world::EntityWorldMut;
use engine_core::{Deformer, Floating, Grounded, InterpolateTransform, NavAgent, TerrainPrefetch};
use engine_settings::ErrorCode;

use crate::SceneError;

/// The components scenes author on entities that this list declares, in
/// the order spawning adds them. Each entry, after the field's doc comment,
/// is `field: Type { feature: Variant, shows: "Inspector", direct },`: the
/// [`SceneObject`](crate::SceneObject) field authoring the component, its
/// type, the [`Feature`](engine_core::Feature) whose plugin it needs (`Core`
/// for none), where inspecting its entity shows it (an inspector, or one
/// field of one) and `direct`: scenes author the engine's own component.
/// Each type is a [`SceneComponent`]. Code over these components passes a
/// macro that receives this list (`entity_components!(callback)`) instead
/// of writing them out: the object's fields, the features it needs, where
/// it may author them, spawning and [`COMPONENT_SHOWS`].
macro_rules! entity_components {
    ($then:ident) => {
        $then! {
            /// Rides the sea. Roots only.
            floating: engine_core::Floating { feature: Ocean, shows: "Floating", direct },
            /// Presses tracks into deformable ground.
            deformer: engine_core::Deformer { feature: Deformation, shows: "Deformer", direct },
            /// Names it a place the camera may be sent to, whose
            /// terrain workers generate ahead of need.
            prefetch: engine_core::TerrainPrefetch {
                feature: Terrain,
                shows: "TerrainPrefetch",
                direct
            },
            /// Stands on the terrain: its translation's y becomes
            /// `offset` meters above the ground under it. Roots only.
            grounded: engine_core::Grounded { feature: Core, shows: "Grounded", direct },
            /// Walks over the terrain to a destination. Roots only.
            nav_agent: engine_core::NavAgent { feature: Navigation, shows: "NavAgent.agent", direct },
        }
    };
}

pub(crate) use entity_components;

/// A listed component's feature: none for the core's own (`Core`).
macro_rules! component_feature {
    (Core) => {
        None
    };
    ($feature:ident) => {
        Some(engine_core::Feature::$feature)
    };
}

pub(crate) use component_feature;

/// Where in a document's hierarchy entities may author a component.
pub(crate) enum Placement {
    /// On any entity.
    Anywhere,
    /// On roots only: authored under a parent, it fails with the code and
    /// the rule.
    Roots(ErrorCode, &'static str),
}

/// A component scenes author on entities, as `entity_components!` lists
/// it: where entities may author it and how it joins the entity spawned
/// for an object authoring it.
pub(crate) trait SceneComponent {
    /// Where entities may author it.
    const PLACEMENT: Placement;

    /// Adds it to `entity`, spawned for the object authoring it.
    fn spawn(&self, entity: &mut EntityWorldMut<'_>);
}

impl SceneComponent for Floating {
    const PLACEMENT: Placement = Placement::Roots(
        ErrorCode::InvalidFloating,
        "only roots ride the sea; float the placement or root instead",
    );

    fn spawn(&self, entity: &mut EntityWorldMut<'_>) {
        entity.insert(*self);
    }
}

impl SceneComponent for Deformer {
    const PLACEMENT: Placement = Placement::Anywhere;

    fn spawn(&self, entity: &mut EntityWorldMut<'_>) {
        entity.insert(*self);
    }
}

impl SceneComponent for TerrainPrefetch {
    const PLACEMENT: Placement = Placement::Anywhere;

    fn spawn(&self, entity: &mut EntityWorldMut<'_>) {
        entity.insert(*self);
    }
}

impl SceneComponent for Grounded {
    // Grounded objects stand on the ground themselves.
    const PLACEMENT: Placement = Placement::Roots(
        ErrorCode::InvalidGrounded,
        "only roots stand on the ground; ground the placement or root instead",
    );

    fn spawn(&self, entity: &mut EntityWorldMut<'_>) {
        entity.insert(*self);
    }
}

impl SceneComponent for NavAgent {
    // Agents walk unparented.
    const PLACEMENT: Placement = Placement::Roots(
        ErrorCode::InvalidNavAgent,
        "only roots walk; give the placement or root the agent instead",
    );

    // Agents move each tick: they present smoothly.
    fn spawn(&self, entity: &mut EntityWorldMut<'_>) {
        entity.insert((*self, InterpolateTransform));
    }
}

/// This module's part of `entity_components!`: where an object authors the
/// listed components, spawning them, and [`COMPONENT_SHOWS`].
macro_rules! components {
    ($(
        $(#[$doc:meta])*
        $field:ident: $type:ty { feature: $feature:ident, shows: $shows:literal, direct },
    )*) => {
        impl crate::SceneObject {
            /// Fails at the first listed component this object, at authored
            /// path `path`, authors where its [`Placement`] forbids.
            pub(crate) fn check_placement(&self, path: &str) -> Result<(), SceneError> {
                $(
                    if self.$field.is_some()
                        && self.parent.is_some()
                        && let Placement::Roots(code, rule) = <$type as SceneComponent>::PLACEMENT
                    {
                        return Err(SceneError::new(
                            code,
                            format!("{path}.{}", stringify!($field)),
                            rule,
                        ));
                    }
                )*
                Ok(())
            }

            /// Adds the listed components this object authors to `entity`,
            /// spawned for it, in the list's order.
            pub(crate) fn spawn_components(&self, entity: &mut EntityWorldMut<'_>) {
                $(
                    if let Some(component) = &self.$field {
                        component.spawn(entity);
                    }
                )*
            }
        }

        /// Where inspecting an entity shows each component the list of
        /// components declares, by the field authoring it: an inspector, or
        /// one field of one. Development tools check that whatever scenes
        /// author shows.
        pub const COMPONENT_SHOWS: &[(&str, &str)] = &[$((stringify!($field), $shows),)*];
    };
}

entity_components!(components);
