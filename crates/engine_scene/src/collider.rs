use engine_collision::{Collider, ColliderShape, CollisionLayers, validate_collider_transform};
use engine_core::Error;
use engine_core::bevy_ecs::world::EntityWorldMut;
use engine_core::glam::Mat4;
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

use crate::components::{Placement, SceneComponent};
use crate::instance::Lookups;
use crate::{SceneDocument, SceneFiles};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SceneColliderShape {
    Cuboid {
        #[cfg_attr(
            feature = "schema",
            schemars(inner(range(min = engine_collision::MIN_COLLIDER_EXTENT, max = engine_collision::MAX_COLLIDER_EXTENT)))
        )]
        half_extents: [f32; 3],
    },
    Sphere {
        #[cfg_attr(
            feature = "schema",
            schemars(range(min = engine_collision::MIN_COLLIDER_EXTENT, max = engine_collision::MAX_COLLIDER_EXTENT))
        )]
        radius: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneCollider {
    pub shape: SceneColliderShape,
    #[serde(default = "default_memberships")]
    pub memberships: u32,
    #[serde(default = "default_filter")]
    pub filter: u32,
    #[serde(default)]
    pub sensor: bool,
}

impl engine_settings::Setting for SceneCollider {}

fn default_memberships() -> u32 {
    1
}
fn default_filter() -> u32 {
    u32::MAX
}

impl SceneColliderShape {
    fn runtime(self) -> ColliderShape {
        match self {
            Self::Cuboid { half_extents } => ColliderShape::Cuboid {
                half_extents: half_extents.into(),
            },
            Self::Sphere { radius } => ColliderShape::Sphere { radius },
        }
    }
}

impl SceneCollider {
    pub(crate) fn runtime(self, path: &str) -> Result<Collider, Error> {
        Collider::new(self.shape.runtime())
            .map(|collider| {
                collider
                    .with_layers(CollisionLayers {
                        memberships: self.memberships,
                        filter: self.filter,
                    })
                    .with_sensor(self.sensor)
            })
            .map_err(|error| {
                error
                    .coded(ErrorCode::InvalidCollider)
                    .within(&format!("{path}.shape"))
            })
    }

    pub(crate) fn validate_world(self, matrix: Mat4, path: &str) -> Result<(), Error> {
        validate_collider_transform(self.shape.runtime(), matrix).map_err(|error| {
            error
                .coded(ErrorCode::InvalidColliderTransform)
                .within(path)
        })
    }
}

impl SceneComponent for SceneCollider {
    type Prepared = Collider;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(
        &self,
        path: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<Collider, Error> {
        self.runtime(path)
    }

    fn spawn(prepared: &Collider, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }
}
