use std::collections::BTreeMap;

use engine_core::bevy_ecs::world::EntityWorldMut;
use engine_core::glam::{Mat4, Vec3};
use engine_core::{Error, Expected, Grounded, InterpolateTransform, Mesh, Tint};
use engine_navigation::NavAgent;
use engine_settings::ErrorCode;
use engine_terrain::TerrainPrefetch;

use crate::instance::Lookups;
use crate::{SceneDocument, SceneFiles};

macro_rules! entity_components {
    ($then:ident) => {
        $then! {
            collider: crate::SceneCollider {
                brick: Some("collision"),
                shows: "Collider",
                rule: any,
                mirror
            },
            light: crate::SceneLight { brick: None, shows: "LocalLight", rule: any, mirror },
            particles: crate::SceneEmitter {
                brick: None,
                shows: "ParticleEmitter",
                rule: any,
                mirror
            },
            tint: [f32; 4] { brick: None, shows: "Tint", rule: range(0.0, 1.0), mirror },
            prefetch: engine_terrain::TerrainPrefetch {
                brick: Some("terrain"),
                shows: "TerrainPrefetch",
                rule: any,
                direct
            },
            grounded: engine_core::Grounded {
                brick: None,
                shows: "Grounded",
                rule: any,
                direct
            },
            rigid_body: engine_physics::RigidBody {
                brick: Some("physics"),
                shows: "RigidBody.body",
                rule: any,
                direct
            },
            nav_agent: engine_navigation::NavAgent {
                brick: Some("navigation"),
                shows: "NavAgent.agent",
                rule: any,
                direct
            },
            spline: crate::SceneSpline { brick: None, shows: "SplineMesh", rule: any, mirror },
            reflection_probe: crate::SceneReflectionProbe {
                brick: None,
                shows: "ReflectionProbe",
                rule: any,
                mirror
            },
            terrain: crate::SceneTerrain {
                brick: Some("terrain"),
                shows: "Terrain.terrain",
                rule: any,
                mirror
            },
        }
    };
}

pub(crate) use entity_components;

pub(crate) enum Placement {
    Anywhere,
    Roots(ErrorCode),
    Entities(ErrorCode),
}

pub(crate) struct Reach<'a> {
    pub world: Mat4,
    pub position: Vec3,
    pub from: f32,
    pub meshes: (&'a [Mesh], &'a BTreeMap<&'a str, usize>),
}

impl Reach<'_> {
    pub fn mesh_size(&self, id: &str) -> f32 {
        let (meshes, indices) = self.meshes;
        indices.get(id).map_or(0.0, |&mesh| {
            let bounds = meshes[mesh].bounds();
            (bounds.max() - bounds.min()).length()
        })
    }

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

pub(crate) trait SceneComponent {
    type Prepared;

    const PLACEMENT: Placement;

    fn prepare(
        &self,
        path: &str,
        document: &SceneDocument,
        files: Option<&SceneFiles>,
    ) -> Result<Self::Prepared, Error>;

    fn spawn(prepared: &Self::Prepared, entity: &mut EntityWorldMut<'_>, lookups: &Lookups);

    fn reach(_prepared: &Self::Prepared, _reach: &Reach<'_>, _extend: &mut impl FnMut(f32)) {}
}

impl SceneComponent for [f32; 4] {
    type Prepared = Tint;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(&self, _: &str, _: &SceneDocument, _: Option<&SceneFiles>) -> Result<Tint, Error> {
        Ok(Tint(*self))
    }

    fn spawn(prepared: &Tint, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for TerrainPrefetch {
    type Prepared = Self;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(&self, _: &str, _: &SceneDocument, _: Option<&SceneFiles>) -> Result<Self, Error> {
        Ok(*self)
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for Grounded {
    type Prepared = Self;

    const PLACEMENT: Placement = Placement::Roots(ErrorCode::InvalidGrounded);

    fn prepare(&self, _: &str, _: &SceneDocument, _: Option<&SceneFiles>) -> Result<Self, Error> {
        Ok(*self)
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for engine_physics::RigidBody {
    type Prepared = Self;

    const PLACEMENT: Placement = Placement::Roots(ErrorCode::InvalidRigidBody);

    fn prepare(&self, _: &str, _: &SceneDocument, _: Option<&SceneFiles>) -> Result<Self, Error> {
        Ok(*self)
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}

impl SceneComponent for NavAgent {
    type Prepared = Self;

    const PLACEMENT: Placement = Placement::Roots(ErrorCode::InvalidNavAgent);

    fn prepare(&self, _: &str, _: &SceneDocument, _: Option<&SceneFiles>) -> Result<Self, Error> {
        Ok(*self)
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert((*prepared, InterpolateTransform));
    }
}

macro_rules! components {
    ($(
        $(#[$doc:meta])*
        $field:ident: $type:ty {
            brick: $brick:expr,
            shows: $shows:literal,
            rule: $rule:ident $(($($arg:expr),*))?,
            $kind:ident
        },
    )*) => {
        impl crate::SceneObject {
            pub(crate) fn check_placement(&self, path: &str) -> Result<(), Error> {
                $(
                    if self.$field.is_some()
                        && self.parent.is_some()
                        && let Placement::Roots(code) = <$type as SceneComponent>::PLACEMENT
                    {
                        return Err(Error::new(code, format!("{path}.{}", stringify!($field)))
                            .expected(Expected::Absent));
                    }
                )*
                Ok(())
            }

            pub(crate) fn check_streamed(&self, path: &str) -> Result<(), Error> {
                $(
                    if self.$field.is_some()
                        && let Placement::Entities(code) = <$type as SceneComponent>::PLACEMENT
                    {
                        return Err(Error::new(code, format!("{path}.{}", stringify!($field)))
                            .expected(Expected::Absent));
                    }
                )*
                Ok(())
            }

            pub(crate) fn prepare_components(
                &self,
                path: &str,
                document: &SceneDocument,
                files: Option<&SceneFiles>,
            ) -> Result<PreparedComponents, Error> {
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

        pub(crate) struct PreparedComponents {
            $($field: Option<<$type as SceneComponent>::Prepared>,)*
        }

        impl PreparedComponents {
            pub(crate) fn spawn(&self, entity: &mut EntityWorldMut<'_>, lookups: &Lookups) {
                $(
                    if let Some(prepared) = &self.$field {
                        <$type as SceneComponent>::spawn(prepared, entity, lookups);
                    }
                )*
            }

            pub(crate) fn reach(&self, reach: &Reach<'_>, extend: &mut impl FnMut(f32)) {
                $(
                    if let Some(prepared) = &self.$field {
                        <$type as SceneComponent>::reach(prepared, reach, extend);
                    }
                )*
            }
        }

        pub const COMPONENT_SHOWS: &[(&str, &str)] = &[$((stringify!($field), $shows),)*];
    };
}

entity_components!(components);
