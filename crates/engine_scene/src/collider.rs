use engine_collision::{Collider, ColliderShape, CollisionLayers, validate_collider_transform};
use engine_core::glam::Mat4;
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

use crate::SceneError;

/// Explicit local collision geometry. Parent transforms apply as full affine
/// matrices; rendering meshes never supply implicit collider dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SceneColliderShape {
    Cuboid {
        /// Positive half sizes on local axes.
        #[cfg_attr(
            feature = "schema",
            schemars(inner(range(min = engine_core::MIN_COLLIDER_EXTENT, max = engine_core::MAX_COLLIDER_EXTENT)))
        )]
        half_extents: [f32; 3],
    },
    Sphere {
        /// Local radius. Nonuniform transforms form ellipsoids.
        #[cfg_attr(
            feature = "schema",
            schemars(range(min = engine_core::MIN_COLLIDER_EXTENT, max = engine_core::MAX_COLLIDER_EXTENT))
        )]
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

/// Collision validates colliders as they become components.
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
                    ErrorCode::InvalidCollider,
                    format!("{path}.shape"),
                    error.to_string(),
                )
            })
    }

    pub(crate) fn validate_world(self, matrix: Mat4, path: &str) -> Result<(), SceneError> {
        validate_collider_transform(self.shape.runtime(), matrix).map_err(|error| {
            SceneError::new(ErrorCode::InvalidColliderTransform, path, error.to_string())
        })
    }
}
