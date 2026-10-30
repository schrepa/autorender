use bevy_ecs::world::EntityWorldMut;
use engine_scene::{
    ComponentContext, ComponentKind, Lookups, Placement, Prepared, PreparedComponent, admitted,
    no_files, valid_world,
};
use engine_settings::{Error, ErrorCode, parse_value_at};
use serde_json::{Value, json};

use crate::RigidBody;

pub static RIGID_BODY: ComponentKind = ComponentKind {
    name: "rigid_body",
    shows: "RigidBody.body",
    placement: Placement::Roots(ErrorCode::InvalidRigidBody),
    schema: engine_scene::component_schema!(RigidBody),
    example,
    validate,
    validate_world: valid_world,
    prepare,
    files: no_files,
    admit: admitted,
};

struct PreparedBody(RigidBody);

impl PreparedComponent for PreparedBody {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(self.0);
    }
}

fn parsed(value: &Value, path: &str) -> Result<RigidBody, Error> {
    let body: RigidBody = parse_value_at(value.clone(), path)?;
    body.validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))?;
    Ok(body)
}

fn example() -> Value {
    json!({})
}

fn validate(value: &Value, path: &str) -> Result<(), Error> {
    parsed(value, path).map(|_| ())
}

fn prepare(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    Ok(Box::new(PreparedBody(parsed(value, context.path)?)))
}
