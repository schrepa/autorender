use engine_core::{ComponentKind, ComponentKindOf, Error, ErrorCode, Placement};
use engine_places::navigation::NavAgent;
use serde_json::{Value, json};

use crate::agents::SceneNavAgent;
use crate::dev::schema_of;

pub(crate) fn nav_agent() -> ComponentKindOf<NavAgent> {
    ComponentKind::of::<NavAgent>("nav_agent")
        .check(validate)
        .placement(Placement::Roots(ErrorCode::InvalidNavAgent))
        .schema(schema)
        .example(example)
}

fn schema() -> Value {
    schema_of::<SceneNavAgent>()
}

fn example() -> Value {
    json!({})
}

fn validate(agent: &NavAgent, path: &str) -> Result<(), Error> {
    SceneNavAgent::from(*agent)
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}
