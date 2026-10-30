use engine_core::bevy_ecs::world::EntityWorldMut;
use engine_core::glam::Mat4;
use engine_core::{Error, parse_value_at};
use engine_scene::{
    ComponentContext, ComponentKind, Lookups, Placement, Prepared, PreparedComponent, admitted,
    component_schema, no_files, no_paths,
};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Collider, ColliderShape, CollisionLayers, validate_collider_transform};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SceneColliderShape {
    Cuboid {
        #[cfg_attr(
            feature = "schema",
            schemars(inner(range(min = crate::MIN_COLLIDER_EXTENT, max = crate::MAX_COLLIDER_EXTENT)))
        )]
        half_extents: [f32; 3],
    },
    Sphere {
        #[cfg_attr(
            feature = "schema",
            schemars(range(min = crate::MIN_COLLIDER_EXTENT, max = crate::MAX_COLLIDER_EXTENT))
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
                    .at(format!("{path}.shape"))
            })
    }

    pub(crate) fn validate_world(self, matrix: Mat4, path: &str) -> Result<(), Error> {
        validate_collider_transform(self.shape.runtime(), matrix)
            .map_err(|error| error.coded(ErrorCode::InvalidColliderTransform).at(path))
    }
}

pub static COLLIDER: ComponentKind = ComponentKind {
    name: "collider",
    shows: "Collider",
    placement: Placement::Anywhere,
    schema: component_schema!(SceneCollider),
    example,
    validate: checked,
    validate_world: checked_world,
    prepare: prepared,
    files: no_files,
    admit: admitted,
    paths: no_paths,
};

fn authored(value: &Value, path: &str) -> Result<SceneCollider, Error> {
    parse_value_at(value.clone(), path)
}

fn example() -> Value {
    json!({ "shape": { "kind": "cuboid", "half_extents": [0.5, 0.5, 0.5] } })
}

fn checked(value: &Value, path: &str) -> Result<(), Error> {
    authored(value, path)?.runtime(path).map(|_| ())
}

fn checked_world(value: &Value, matrix: Mat4, path: &str) -> Result<(), Error> {
    authored(value, path)?.validate_world(matrix, path)
}

fn prepared(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    let collider = authored(value, context.path)?.runtime(context.path)?;
    Ok(Box::new(collider))
}

impl PreparedComponent for Collider {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*self);
    }
}
