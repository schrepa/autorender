//! The list of components scenes author on entities, and how each joins
//! the entity spawned for an object authoring it.

use engine_core::bevy_ecs::world::EntityWorldMut;

/// The components scenes author on entities that this list declares, in
/// the order spawning adds them. Each entry, after the field's doc comment,
/// is `field: Type { feature: Variant, shows: "Inspector", direct },`: the
/// [`SceneObject`](crate::SceneObject) field authoring the component, its
/// type, the [`Feature`](engine_core::Feature) whose plugin it needs, where
/// inspecting its entity shows it (an inspector, or one field of one) and
/// `direct`: scenes author the engine's own component, which spawns as
/// authored. Code over these components passes a macro that receives this
/// list (`entity_components!(callback)`) instead of writing them out: the
/// object's fields, the features it needs, spawning and [`COMPONENT_SHOWS`].
macro_rules! entity_components {
    ($then:ident) => {
        $then! {
            /// Rides the sea. Roots only.
            floating: engine_core::Floating { feature: Ocean, shows: "Floating", direct },
        }
    };
}

pub(crate) use entity_components;

/// A component scenes author on entities, as `entity_components!` lists
/// it: how it joins the entity spawned for an object authoring it.
pub(crate) trait SceneComponent {
    /// Adds it to `entity`, spawned for the object authoring it.
    fn spawn(&self, entity: &mut EntityWorldMut<'_>);
}

/// This module's part of `entity_components!`: each direct component's
/// [`SceneComponent`], spawning them, and [`COMPONENT_SHOWS`].
macro_rules! components {
    ($(
        $(#[$doc:meta])*
        $field:ident: $type:ty { feature: $feature:ident, shows: $shows:literal, direct },
    )*) => {
        $(
            impl SceneComponent for $type {
                fn spawn(&self, entity: &mut EntityWorldMut<'_>) {
                    entity.insert(*self);
                }
            }
        )*

        impl crate::SceneObject {
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
