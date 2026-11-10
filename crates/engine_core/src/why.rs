use bevy_ecs::entity::Entity;
use bevy_ecs::system::SystemParamItem;
use bevy_ecs::world::World;
use engine_registry::{BrickStates, OperationScene, Registry, SceneEntity};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    DrawnState, DrawnTable, Error, ErrorCode, Expected, GlobalTransforms, Mover, Operation,
    Readiness, Refusals,
};

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Named {
    Id(String),
    Bits(u64),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WhyRequest {
    #[serde(default)]
    entity: Option<Named>,
    #[serde(default)]
    brick: Option<String>,
}

pub(crate) struct Why;

impl Operation for Why {
    const NAME: &'static str = "why";
    const CONTROL: bool = false;
    type Params = &'static World;
    type Request = WhyRequest;

    fn run(world: SystemParamItem<'_, '_, Self::Params>, request: WhyRequest) -> Result<Value, Error> {
        match (request.entity, request.brick) {
            (Some(named), None) => entity_answer(world, &named),
            (None, Some(brick)) => brick_answer(world, &brick),
            (Some(_), Some(_)) => Err(Error::new(ErrorCode::InvalidArguments, "$.brick")
                .expected(Expected::Absent)),
            (None, None) => Err(Error::new(ErrorCode::MissingField, "$.entity")
                .expected(Expected::Present)),
        }
    }

    fn schema() -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "entity": { "type": ["string", "integer"] },
                "brick": { "type": "string" }
            }
        })
    }
}

fn found(world: &World, named: &Named) -> Option<Entity> {
    let entity = match named {
        Named::Id(id) => world.get_resource::<OperationScene>()?.entity(id)?,
        Named::Bits(bits) => Entity::try_from_bits(*bits)?,
    };
    world.entities().contains(entity).then_some(entity)
}

fn drawn_view(state: &DrawnState) -> Value {
    match state {
        DrawnState::Drawn { slot } => json!({ "state": "drawn", "slot": slot }),
        DrawnState::Hidden => json!({ "state": "hidden" }),
        DrawnState::Unplaced => json!({ "state": "unplaced" }),
        DrawnState::Waiting => json!({ "state": "waiting" }),
        DrawnState::Refused(_) => json!({ "state": "refused" }),
    }
}

fn entity_answer(world: &World, named: &Named) -> Result<Value, Error> {
    let entity = found(world, named).ok_or_else(|| {
        let got = match named {
            Named::Id(id) => json!(id),
            Named::Bits(bits) => json!(bits),
        };
        Error::new(ErrorCode::UnknownEntity, "$.entity").got(got)
    })?;
    let refused = world
        .get_resource::<Refusals>()
        .map_or(&[][..], |refusals| refusals.of(entity));
    let drawn = world
        .get_resource::<DrawnTable>()
        .and_then(|table| table.state(entity))
        .map(drawn_view);
    let node = world
        .get_resource::<GlobalTransforms>()
        .and_then(|globals| globals.word(entity));
    let authored = world.get::<SceneEntity>(entity);
    Ok(json!({
        "entity": entity.to_bits(),
        "id": authored.map(|authored| authored.authored_id.as_str()),
        "name": authored.map(|authored| authored.name.as_str()),
        "doing": refused.is_empty() && node.is_none_or(|word| word.active),
        "refused": refused,
        "node": node,
        "drawn": drawn,
        "mover": world.get::<Mover>(entity),
    }))
}

fn brick_answer(world: &World, brick: &str) -> Result<Value, Error> {
    let registry = world
        .get_resource::<Registry>()
        .ok_or_else(|| Error::new(ErrorCode::MissingResource, "$.brick").got(brick))?;
    let id = registry.id(brick).ok_or_else(|| {
        Error::new(ErrorCode::UnknownValue, "$.brick")
            .got(brick)
            .variants(registry.bricks().iter().map(|record| record.package))
    })?;
    let states = world.get_resource::<BrickStates>();
    let state = states.and_then(|states| states.state(id));
    let needs: Vec<&str> = states
        .map(|states| states.needs(id))
        .unwrap_or_default()
        .iter()
        .filter_map(|&need| registry.record(need).map(|record| record.package))
        .collect();
    let failed: Vec<&Error> = world
        .get_resource::<Readiness>()
        .into_iter()
        .flat_map(|readiness| readiness.failed.iter())
        .filter(|(by, _)| by == brick)
        .map(|(_, error)| error)
        .collect();
    let prefix = format!("$.bricks.{brick}.");
    let problems: Vec<&Error> = registry
        .problems()
        .iter()
        .filter(|problem| problem.path.starts_with(&prefix))
        .collect();
    Ok(json!({
        "brick": brick,
        "doing": state.is_some_and(engine_registry::BrickState::on) && failed.is_empty(),
        "state": state,
        "needs": needs,
        "failed": failed,
        "problems": problems,
    }))
}
