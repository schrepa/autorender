use engine_core::{Error, ErrorCode, Placed};
use engine_places::physics::{Collider, ColliderShape, MAX_COLLIDER_EXTENT, MIN_COLLIDER_EXTENT};
use glam::Mat4;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::types::{validate_collider_transform, validate_shape};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum SceneColliderShape {
    Cuboid {
        #[schemars(inner(range(min = MIN_COLLIDER_EXTENT, max = MAX_COLLIDER_EXTENT)))]
        half_extents: [f32; 3],
    },
    Sphere {
        #[schemars(range(min = MIN_COLLIDER_EXTENT, max = MAX_COLLIDER_EXTENT))]
        radius: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct SceneCollider {
    pub shape: SceneColliderShape,
    #[serde(default = "default_memberships")]
    pub memberships: u32,
    #[serde(default = "default_filter")]
    pub filter: u32,
    #[serde(default)]
    pub sensor: bool,
}

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

impl From<&Collider> for SceneCollider {
    fn from(collider: &Collider) -> Self {
        let layers = collider.layers();
        Self {
            shape: match collider.shape() {
                ColliderShape::Cuboid { half_extents } => SceneColliderShape::Cuboid {
                    half_extents: half_extents.to_array(),
                },
                ColliderShape::Sphere { radius } => SceneColliderShape::Sphere { radius },
            },
            memberships: layers.memberships,
            filter: layers.filter,
            sensor: collider.is_sensor(),
        }
    }
}

impl SceneCollider {
    fn runtime(self, path: &str) -> Result<(), Error> {
        validate_shape(self.shape.runtime()).map_err(|error| {
            error
                .coded(ErrorCode::InvalidCollider)
                .at(format!("{path}.shape"))
        })
    }

    fn validate_world(self, matrix: Mat4, path: &str) -> Result<(), Error> {
        validate_collider_transform(self.shape.runtime(), matrix)
            .map_err(|error| error.coded(ErrorCode::InvalidColliderTransform).at(path))
    }
}

pub(crate) fn schema() -> Value {
    serde_json::to_value(schemars::schema_for!(SceneCollider)).unwrap_or(Value::Null)
}

pub(crate) fn example() -> Value {
    json!({ "shape": { "kind": "cuboid", "half_extents": [0.5, 0.5, 0.5] } })
}

pub(crate) fn checked(collider: &Collider, path: &str) -> Result<(), Error> {
    SceneCollider::from(collider).runtime(path)
}

pub(crate) fn checked_world(collider: &Collider, placed: Placed, path: &str) -> Result<(), Error> {
    SceneCollider::from(collider).validate_world(placed.world, path)
}
