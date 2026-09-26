use engine_collision::{Collider, ColliderShape, CollisionLayers, validate_collider_transform};
use engine_core::glam::Mat4;
use serde::{Deserialize, Serialize};

use crate::SceneError;

/// Explicit local collision geometry. Parent transforms apply as full affine
/// matrices; rendering meshes never supply implicit collider dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SceneColliderShape {
    Cuboid {
        /// Positive half sizes on local axes, each in [0.0001, 10000].
        half_extents: [f32; 3],
    },
    Sphere {
        /// Local radius in [0.0001, 10000]. Nonuniform transforms form ellipsoids.
        #[cfg_attr(feature = "schema", schemars(range(min = 0.0001, max = 10000.0)))]
        radius: f32,
    },
}

/// A collider or sensor on any authored node, including prefab instance roots.
/// Zero layer masks are valid and intentionally disable matching pairs.
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
    pub(crate) fn runtime(self, path: &str) -> Result<Collider, SceneError> {
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
                SceneError::new(
                    "invalid_collider",
                    format!("{path}.shape"),
                    error.to_string(),
                )
            })
    }

    pub(crate) fn validate_world(self, matrix: Mat4, path: &str) -> Result<(), SceneError> {
        validate_collider_transform(self.shape.runtime(), matrix)
            .map_err(|error| SceneError::new("invalid_collider_transform", path, error.to_string()))
    }
}
