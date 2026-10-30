use engine_core::bevy_ecs::world::EntityWorldMut;
use engine_core::{Error, ErrorCode, parse_value_at};
use engine_scene::{
    ComponentContext, ComponentKind, Lookups, Placement, Prepared, PreparedComponent, admitted,
    no_files, valid_world,
};
use serde_json::{Value, json};

use crate::Pond;

pub static POND: ComponentKind = ComponentKind {
    name: "pond",
    shows: "Pond",
    placement: Placement::Anywhere,
    schema: engine_scene::component_schema!(Pond),
    example,
    validate,
    validate_world: valid_world,
    prepare,
    files: no_files,
    admit: admitted,
};

fn example() -> Value {
    json!({"center": [0.0, 0.0], "size": [16.0, 16.0], "surface": 0.0})
}

fn parsed(value: &Value, path: &str) -> Result<Pond, Error> {
    let pond: Pond = parse_value_at(value.clone(), path)?;
    pond.validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))?;
    Ok(pond)
}

fn validate(value: &Value, path: &str) -> Result<(), Error> {
    parsed(value, path).map(drop)
}

fn prepare(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    Ok(Box::new(parsed(value, context.path)?))
}

impl PreparedComponent for Pond {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _lookups: &Lookups) {
        entity.insert(*self);
    }
}
