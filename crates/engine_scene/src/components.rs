//! The list of components scenes author on entities, and how each is
//! prepared and joins the entity spawned for an object authoring it.

use std::collections::BTreeMap;

use engine_core::bevy_ecs::world::EntityWorldMut;
use engine_core::glam::{Mat4, Vec3};
use engine_core::{
    Deformer, Floating, Grounded, InterpolateTransform, Mesh, NavAgent, TerrainPrefetch, Tint,
};
use engine_settings::ErrorCode;

use crate::instance::Lookups;
use crate::{SceneDocument, SceneError, SceneFiles};

/// The components scenes author on entities that this list declares, in
/// the order spawning adds them. Each entry, after the field's doc comment,
/// is `field: Type { feature: Variant, shows: "Inspector", rule: RULE,
/// kind },`: the [`SceneObject`](crate::SceneObject) field authoring the
/// component, its authored type, the [`Feature`](engine_core::Feature)
/// whose plugin it needs (`Core` for none), where inspecting its entity
/// shows it (an inspector, or one field of one), the field's rule, and
/// `direct` when scenes author the engine's own component or `mirror` when
/// the authored type mirrors it. Each type is a [`SceneComponent`]. Code
/// over these components passes a macro that receives this list
/// (`entity_components!(callback)`) instead of writing them out: the
/// object's fields, the features it needs, where it may author them,
/// preparing, spawning, streamed items' reach and [`COMPONENT_SHOWS`].
macro_rules! entity_components {
    ($then:ident) => {
        $then! {
            /// Local collision shape, independent of rendering.
            collider: crate::SceneCollider {
                feature: Collision,
                shows: "Collider",
                rule: any,
                mirror
            },
            /// A point or spot light.
            light: crate::SceneLight { feature: Core, shows: "LocalLight", rule: any, mirror },
            /// A particle emitter.
            particles: crate::SceneEmitter {
                feature: Core,
                shows: "ParticleEmitter",
                rule: any,
                mirror
            },
            /// Multiplies this object's color and opacity channel by channel,
            /// each factor finite and in `[0, 1]`.
            tint: [f32; 4] { feature: Core, shows: "Tint", rule: range(0.0, 1.0), mirror },
            /// Projects a material onto the opaque surfaces inside this entity's
            /// box.
            decal: crate::SceneDecal { feature: Core, shows: "Decal", rule: any, mirror },
            /// Rides the sea. Roots only.
            floating: engine_core::Floating {
                feature: Ocean,
                shows: "Floating",
                rule: any,
                direct
            },
            /// Presses tracks into deformable ground.
            deformer: engine_core::Deformer {
                feature: Deformation,
                shows: "Deformer",
                rule: any,
                direct
            },
            /// Names it a place the camera may be sent to, whose
            /// terrain workers generate ahead of need.
            prefetch: engine_core::TerrainPrefetch {
                feature: Terrain,
                shows: "TerrainPrefetch",
                rule: any,
                direct
            },
            /// Stands on the terrain: its translation's y becomes
            /// `offset` meters above the ground under it. Roots only.
            grounded: engine_core::Grounded {
                feature: Core,
                shows: "Grounded",
                rule: any,
                direct
            },
            /// Walks over the terrain to a destination. Roots only.
            nav_agent: engine_core::NavAgent {
                feature: Navigation,
                shows: "NavAgent.agent",
                rule: any,
                direct
            },
            /// One of this document's meshes bent along a curve.
            spline: crate::SceneSpline { feature: Core, shows: "SplineMesh", rule: any, mirror },
            /// Captures this entity's surroundings into an environment map that
            /// surfaces inside its world box reflect.
            reflection_probe: crate::SceneReflectionProbe {
                feature: Core,
                shows: "ReflectionProbe",
                rule: any,
                mirror
            },
            /// A heightfield terrain whose tiles stream around the camera.
            terrain: crate::SceneTerrain {
                feature: Terrain,
                shows: "Terrain.terrain",
                rule: any,
                mirror
            },
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

/// Where in a document entities may author a component.
pub(crate) enum Placement {
    /// On any entity.
    Anywhere,
    /// On roots only: authored under a parent, it fails with the code and
    /// the rule.
    Roots(ErrorCode, &'static str),
    /// On the document's entities and prefabs only: authored on a streamed
    /// entity, it fails with the code and the rule.
    Entities(ErrorCode, &'static str),
}

/// Where a streamed entity stands in its item, which its components reach
/// beyond, and the document's meshes they may name.
pub(crate) struct Reach<'a> {
    /// The entity's world matrix.
    pub world: Mat4,
    /// The item's position: its root's.
    pub position: Vec3,
    /// The entity's distance from it.
    pub from: f32,
    /// The document's meshes, and their indices by authored ID.
    pub meshes: (&'a [Mesh], &'a BTreeMap<&'a str, usize>),
}

impl Reach<'_> {
    /// The diagonal of the bounds of the document's mesh `id`; none, 0.
    pub fn mesh_size(&self, id: &str) -> f32 {
        let (meshes, indices) = self.meshes;
        indices.get(id).map_or(0.0, |&mesh| {
            let bounds = meshes[mesh].bounds();
            (bounds.max() - bounds.min()).length()
        })
    }

    /// The distance from the item's position to the farthest corner of the
    /// entity's local box from `min` to `max`.
    pub fn of_box(&self, min: Vec3, max: Vec3) -> f32 {
        (0..8)
            .map(|corner| {
                let pick = |bit: u32, low: f32, high: f32| {
                    if corner & (1 << bit) == 0 { low } else { high }
                };
                let point = Vec3::new(
                    pick(0, min.x, max.x),
                    pick(1, min.y, max.y),
                    pick(2, min.z, max.z),
                );
                self.world.transform_point3(point).distance(self.position)
            })
            .fold(0.0, engine_core::math::max)
    }
}

/// A component scenes author on entities, as `entity_components!` lists
/// it: where entities may author it, what preparing a document makes of
/// it, how it joins the entity spawned for an object authoring it, and how
/// far beyond its entity it reaches.
pub(crate) trait SceneComponent {
    /// What preparing the authored value makes for spawning.
    type Prepared;

    /// Where entities may author it.
    const PLACEMENT: Placement;

    /// The authored value, at authored path `path` of `document`, prepared
    /// to spawn; its own rules are checked with the entity's. `files`
    /// supplies the files it reads, or with `None` they read as
    /// placeholders.
    fn prepare(
        &self,
        path: &str,
        document: &SceneDocument,
        files: Option<&SceneFiles>,
    ) -> Result<Self::Prepared, SceneError>;

    /// Adds the component `prepared` makes to `entity`, spawned for the
    /// object authoring it; `lookups` holds the document's resident assets.
    fn spawn(prepared: &Self::Prepared, entity: &mut EntityWorldMut<'_>, lookups: &Lookups);

    /// Hands `extend` how far from a streamed item's position the
    /// component reaches, when it reaches beyond its entity standing at
    /// `reach`.
    fn reach(_prepared: &Self::Prepared, _reach: &Reach<'_>, _extend: &mut impl FnMut(f32)) {}
}

impl SceneComponent for [f32; 4] {
    type Prepared = Tint;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<Tint, SceneError> {
        Ok(Tint(*self))
    }

    fn spawn(prepared: &Tint, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for Floating {
    type Prepared = Self;

    const PLACEMENT: Placement = Placement::Roots(
        ErrorCode::InvalidFloating,
        "only roots ride the sea; float the placement or root instead",
    );

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<Self, SceneError> {
        Ok(*self)
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for Deformer {
    type Prepared = Self;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<Self, SceneError> {
        Ok(*self)
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for TerrainPrefetch {
    type Prepared = Self;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<Self, SceneError> {
        Ok(*self)
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for Grounded {
    type Prepared = Self;

    // Grounded objects stand on the ground themselves.
    const PLACEMENT: Placement = Placement::Roots(
        ErrorCode::InvalidGrounded,
        "only roots stand on the ground; ground the placement or root instead",
    );

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<Self, SceneError> {
        Ok(*self)
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for NavAgent {
    type Prepared = Self;

    // Agents walk unparented.
    const PLACEMENT: Placement = Placement::Roots(
        ErrorCode::InvalidNavAgent,
        "only roots walk; give the placement or root the agent instead",
    );

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<Self, SceneError> {
        Ok(*self)
    }

    // Agents move each tick: they present smoothly.
    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert((*prepared, InterpolateTransform));
    }
}

/// This module's part of `entity_components!`: where an object authors the
/// listed components, preparing and spawning them, their reach, and
/// [`COMPONENT_SHOWS`].
macro_rules! components {
    ($(
        $(#[$doc:meta])*
        $field:ident: $type:ty {
            feature: $feature:ident,
            shows: $shows:literal,
            rule: $rule:ident $(($($arg:expr),*))?,
            $kind:ident
        },
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

            /// Fails at the first listed component this streamed object, at
            /// authored path `path`, authors that only the document's
            /// entities may.
            pub(crate) fn check_streamed(&self, path: &str) -> Result<(), SceneError> {
                $(
                    if self.$field.is_some()
                        && let Placement::Entities(code, rule) =
                            <$type as SceneComponent>::PLACEMENT
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

            /// The listed components this object, at authored path `path`
            /// of `document`, authors, prepared in the list's order with the
            /// scene's `files`.
            pub(crate) fn prepare_components(
                &self,
                path: &str,
                document: &SceneDocument,
                files: Option<&SceneFiles>,
            ) -> Result<PreparedComponents, SceneError> {
                Ok(PreparedComponents {
                    $($field: self
                        .$field
                        .as_ref()
                        .map(|component| {
                            <$type as SceneComponent>::prepare(
                                component,
                                &format!("{path}.{}", stringify!($field)),
                                document,
                                files,
                            )
                        })
                        .transpose()?,)*
                })
            }
        }

        /// The listed components an object authors, prepared.
        pub(crate) struct PreparedComponents {
            $($field: Option<<$type as SceneComponent>::Prepared>,)*
        }

        impl PreparedComponents {
            /// Adds the components to `entity`, spawned for the object
            /// authoring them, in the list's order; `lookups` holds the
            /// document's resident assets.
            pub(crate) fn spawn(&self, entity: &mut EntityWorldMut<'_>, lookups: &Lookups) {
                $(
                    if let Some(prepared) = &self.$field {
                        <$type as SceneComponent>::spawn(prepared, entity, lookups);
                    }
                )*
            }

            /// Hands `extend` how far each component reaching beyond its
            /// streamed entity, standing at `reach`, reaches.
            pub(crate) fn reach(&self, reach: &Reach<'_>, extend: &mut impl FnMut(f32)) {
                $(
                    if let Some(prepared) = &self.$field {
                        <$type as SceneComponent>::reach(prepared, reach, extend);
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
