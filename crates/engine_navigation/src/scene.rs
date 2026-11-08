use engine_core::{ComponentKind, ComponentKindOf, Error, ErrorCode, Placement};
use engine_places_simulation::navigation::NavAgent;
use serde_json::Value;

use crate::agents::SceneNavAgent;
use crate::dev::schema_of;

pub const INVALID_NAV_AGENT: ErrorCode = ErrorCode::brick("invalid_nav_agent");

pub(crate) fn nav_agent() -> ComponentKindOf<NavAgent> {
    ComponentKind::of::<NavAgent>("nav_agent")
        .check(validate)
        .placement(Placement::Roots(INVALID_NAV_AGENT))
        .schema(schema)
}

fn schema() -> Value {
    schema_of::<SceneNavAgent>()
}

fn validate(agent: &NavAgent, path: &str) -> Result<(), Error> {
    SceneNavAgent::from(*agent)
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}
