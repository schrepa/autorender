use engine_core::{ComponentKind, ComponentKindOf, Error, ErrorCode, Expected, Placed};
use engine_places::navigation::NavAgent;
use serde_json::{Value, json};

use crate::agents::SceneNavAgent;

pub(crate) fn nav_agent() -> ComponentKindOf<NavAgent> {
    ComponentKind::of::<NavAgent>("nav_agent")
        .check(validate)
        .check_placed(roots)
        .schema(schema)
        .example(example)
}

fn schema() -> Value {
    serde_json::to_value(schemars::schema_for!(SceneNavAgent)).unwrap_or(Value::Null)
}

fn example() -> Value {
    json!({})
}

fn validate(agent: &NavAgent, path: &str) -> Result<(), Error> {
    SceneNavAgent::from(*agent)
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

fn roots(_: &NavAgent, placed: Placed, path: &str) -> Result<(), Error> {
    if placed.root {
        return Ok(());
    }
    Err(Error::new(ErrorCode::InvalidNavAgent, path).expected(Expected::Absent))
}
