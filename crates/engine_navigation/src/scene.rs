use bevy_ecs::world::EntityWorldMut;
use engine_core::{Error, ErrorCode, InterpolateTransform, parse_value_at};
use engine_scene::{
    ComponentContext, ComponentKind, Lookups, Placement, Prepared, PreparedComponent, admitted,
    no_files, valid_world,
};
use serde_json::{Value, json};

use crate::NavAgent;

pub static NAV_AGENT: ComponentKind = ComponentKind {
    name: "nav_agent",
    shows: "NavAgent.agent",
    placement: Placement::Roots(ErrorCode::InvalidNavAgent),
    schema: engine_scene::component_schema!(NavAgent),
    example,
    validate,
    validate_world: valid_world,
    prepare,
    files: no_files,
    admit: admitted,
};

fn example() -> Value {
    json!({})
}

fn agent(value: &Value, path: &str) -> Result<NavAgent, Error> {
    let agent: NavAgent = parse_value_at(value.clone(), path)?;
    agent
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))?;
    Ok(agent)
}

fn validate(value: &Value, path: &str) -> Result<(), Error> {
    agent(value, path)?;
    Ok(())
}

fn prepare(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    Ok(Box::new(agent(value, context.path)?))
}

impl PreparedComponent for NavAgent {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert((*self, InterpolateTransform));
    }
}
