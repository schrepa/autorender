use std::collections::BTreeMap;
use std::sync::{PoisonError, RwLock};

use engine_core::bevy_ecs::prelude::Resource;
use engine_core::bevy_ecs::world::{EntityWorldMut, World};
use engine_core::glam::{Mat4, Vec3};
use engine_core::smoke::{SmokeEntry, SmokeResult, SmokeSession};
use engine_core::{Bricks, Error, Expected, Grounded, InterpolateTransform, List, Mesh, Tint};
use engine_navigation::NavAgent;
use engine_settings::ErrorCode;
use engine_terrain::TerrainPrefetch;
use serde_json::Value;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Anywhere,
    Roots(ErrorCode),
    Entities(ErrorCode),
}

pub struct Reach<'a> {
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

pub struct ComponentContext<'a> {
    pub path: &'a str,
    pub document: &'a SceneDocument,
    pub files: Option<&'a SceneFiles>,
}

pub trait PreparedComponent: Send + Sync + 'static {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, lookups: &Lookups);

    fn reach(&self, _reach: &Reach<'_>, _extend: &mut dyn FnMut(f32)) {}
}

pub type Prepared = Result<Box<dyn PreparedComponent>, Error>;

pub struct ComponentKind {
    pub name: &'static str,
    pub shows: &'static str,
    pub placement: Placement,
    pub schema: fn() -> Value,
    pub example: fn() -> Value,
    pub validate: fn(&Value, &str) -> Result<(), Error>,
    pub validate_world: fn(&Value, Mat4, &str) -> Result<(), Error>,
    pub prepare: fn(&Value, &ComponentContext<'_>) -> Prepared,
    pub files: fn(&Value) -> Vec<String>,
    pub admit: fn(&Value, &World, &str) -> Result<(), Error>,
}

impl std::fmt::Debug for ComponentKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ComponentKind")
            .field("name", &self.name)
            .field("shows", &self.shows)
            .field("placement", &self.placement)
            .finish_non_exhaustive()
    }
}

pub fn valid(_value: &Value, _path: &str) -> Result<(), Error> {
    Ok(())
}

pub fn valid_world(_value: &Value, _world: Mat4, _path: &str) -> Result<(), Error> {
    Ok(())
}

pub fn no_files(_value: &Value) -> Vec<String> {
    Vec::new()
}

pub fn admitted(_value: &Value, _world: &World, _path: &str) -> Result<(), Error> {
    Ok(())
}

#[macro_export]
macro_rules! component_schema {
    ($ty:ty) => {{
        #[cfg(feature = "schema")]
        fn component_schema() -> ::serde_json::Value {
            ::serde_json::to_value(::schemars::schema_for!($ty))
                .unwrap_or(::serde_json::Value::Null)
        }
        #[cfg(not(feature = "schema"))]
        fn component_schema() -> ::serde_json::Value {
            ::serde_json::Value::Null
        }
        component_schema
    }};
}

pub static CORE_COMPONENTS: &[&ComponentKind] = &[];

static KINDS: RwLock<Vec<(&'static str, &'static ComponentKind)>> = RwLock::new(Vec::new());

pub fn register_components(brick: &'static str, kinds: &[&'static ComponentKind]) {
    let mut known = KINDS.write().unwrap_or_else(PoisonError::into_inner);
    for &kind in kinds {
        match known.iter_mut().find(|(_, held)| held.name == kind.name) {
            Some(held) => *held = (brick, kind),
            None => known.push((brick, kind)),
        }
    }
}

pub fn component_kind(name: &str) -> Option<(&'static str, &'static ComponentKind)> {
    component_kinds()
        .into_iter()
        .find(|(_, kind)| kind.name == name)
}

pub fn component_kinds() -> Vec<(&'static str, &'static ComponentKind)> {
    let mut kinds: Vec<(&'static str, &'static ComponentKind)> = CORE_COMPONENTS
        .iter()
        .map(|&kind| (engine_core::CORE_ROW, kind))
        .collect();
    kinds.extend(
        KINDS
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .copied(),
    );
    kinds
}

pub fn component_names() -> Vec<&'static str> {
    component_kinds()
        .into_iter()
        .map(|(_, kind)| kind.name)
        .collect()
}

#[derive(Resource, Default)]
pub struct SceneComponents {
    kinds: Vec<(&'static str, &'static ComponentKind)>,
}

impl SceneComponents {
    pub fn add(&mut self, brick: &'static str, kinds: &[&'static ComponentKind]) {
        for &kind in kinds {
            match self
                .kinds
                .iter_mut()
                .find(|(_, held)| held.name == kind.name)
            {
                Some(held) => *held = (brick, kind),
                None => self.kinds.push((brick, kind)),
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<(&'static str, &'static ComponentKind)> {
        self.kinds
            .iter()
            .copied()
            .find(|(_, kind)| kind.name == name)
    }

    pub fn kinds(&self) -> &[(&'static str, &'static ComponentKind)] {
        &self.kinds
    }
}

pub struct SceneComponentsList;

impl List for SceneComponentsList {
    const NAME: &'static str = "scene_components";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_scene_components",
        brick: Self::NAME,
        gpu: false,
        run: scene_components,
    }];
}

pub trait AddSceneComponents {
    fn scene_components(&mut self, kinds: &[&'static ComponentKind]) -> &mut Self;
}

impl AddSceneComponents for Bricks<'_> {
    fn scene_components(&mut self, kinds: &[&'static ComponentKind]) -> &mut Self {
        let brick = self.name();
        register_components(brick, kinds);
        let world = self.world_mut();
        if !world.contains_resource::<SceneComponents>() {
            world.init_resource::<SceneComponents>();
        }
        world.resource_mut::<SceneComponents>().add(brick, kinds);
        self.list::<SceneComponentsList>()
    }
}

fn scene_components(_session: &mut dyn SmokeSession) -> SmokeResult {
    Ok(())
}
