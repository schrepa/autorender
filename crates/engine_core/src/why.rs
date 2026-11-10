use bevy_ecs::system::SystemParamItem;
use bevy_ecs::world::World;
use engine_registry::{BrickStateView, BrickStates, OperationScene, Registry, SceneEntity};
use serde::{Deserialize, Serialize};

use crate::{
    Clock, DrawnState, DrawnTable, Error, ErrorCode, Expected, GlobalTransforms, Mobility, Mover,
    NodeWord, Operation, Pace, Readiness, Refusal, Refusals, Unsettled,
};

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct WhyRequest {
    #[serde(default)]
    entity: Option<String>,
    #[serde(default)]
    brick: Option<String>,
}

pub(crate) struct Why;

#[derive(Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub(crate) enum WhyAnswer {
    Entity(EntityWhy),
    Brick(BrickWhy),
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct RefusedPart {
    part: String,
    reason: Refusal,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct NodeView {
    active: bool,
    visible: bool,
    real_clock: bool,
    moving: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DrawnWhy {
    Drawn { slot: u32 },
    Hidden,
    Unplaced,
    Waiting,
    Refused,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct MoverView {
    by: String,
    frame_paced: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct EntityWhy {
    id: String,
    name: Option<String>,
    doing: bool,
    refused: Vec<RefusedPart>,
    node: Option<NodeView>,
    drawn: Option<DrawnWhy>,
    mover: Option<MoverView>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct BrickWhy {
    brick: String,
    doing: bool,
    state: Option<BrickStateView>,
    needs: Vec<String>,
    waiting: Vec<Unsettled>,
    failed: Vec<Refusal>,
    problems: Vec<Refusal>,
}

impl Operation for Why {
    const NAME: &'static str = "why";
    const CONTROL: bool = false;
    type Params = &'static World;
    type Request = WhyRequest;
    type Reply = WhyAnswer;

    fn run(
        world: SystemParamItem<'_, '_, Self::Params>,
        request: WhyRequest,
    ) -> Result<WhyAnswer, Error> {
        match (request.entity, request.brick) {
            (Some(id), None) => entity_answer(world, &id).map(WhyAnswer::Entity),
            (None, Some(brick)) => brick_answer(world, &brick).map(WhyAnswer::Brick),
            (Some(_), Some(_)) => {
                Err(Error::new(ErrorCode::InvalidArguments, "$.brick").expected(Expected::Absent))
            }
            (None, None) => {
                Err(Error::new(ErrorCode::MissingField, "$.entity").expected(Expected::Present))
            }
        }
    }
}

fn drawn_view(state: &DrawnState) -> DrawnWhy {
    match state {
        DrawnState::Drawn { slot } => DrawnWhy::Drawn { slot: *slot },
        DrawnState::Hidden => DrawnWhy::Hidden,
        DrawnState::Unplaced => DrawnWhy::Unplaced,
        DrawnState::Waiting => DrawnWhy::Waiting,
        DrawnState::Refused(_) => DrawnWhy::Refused,
    }
}

fn node_view(word: NodeWord) -> NodeView {
    NodeView {
        active: word.active,
        visible: word.visible,
        real_clock: match word.clock {
            Clock::Game => false,
            Clock::Real => true,
        },
        moving: match word.mobility {
            Mobility::Static => false,
            Mobility::Moving => true,
        },
    }
}

fn mover_view(mover: &Mover) -> MoverView {
    MoverView {
        by: mover.by.name().to_owned(),
        frame_paced: match mover.pace {
            Pace::Tick => false,
            Pace::Frame => true,
        },
    }
}

fn entity_answer(world: &World, id: &str) -> Result<EntityWhy, Error> {
    let entity = world
        .get_resource::<OperationScene>()
        .and_then(|scene| scene.entity(id))
        .filter(|&entity| world.entities().contains(entity))
        .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(id))?;
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
    Ok(EntityWhy {
        id: id.to_owned(),
        name: world
            .get::<SceneEntity>(entity)
            .map(|authored| authored.name.clone()),
        doing: refused.is_empty() && node.is_none_or(|word| word.active),
        refused: refused
            .iter()
            .map(|refused| RefusedPart {
                part: refused.part.to_owned(),
                reason: Refusal::of(&refused.reason),
            })
            .collect(),
        node: node.map(node_view),
        drawn,
        mover: world.get::<Mover>(entity).map(mover_view),
    })
}

fn brick_answer(world: &World, brick: &str) -> Result<BrickWhy, Error> {
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
    let needs: Vec<String> = states
        .map(|states| states.needs(id))
        .unwrap_or_default()
        .iter()
        .filter_map(|&need| {
            registry
                .record(need)
                .map(|record| record.package.to_owned())
        })
        .collect();
    let readiness = world.get_resource::<Readiness>();
    let waiting: Vec<Unsettled> = readiness
        .into_iter()
        .flat_map(|readiness| readiness.of(brick))
        .cloned()
        .collect();
    let failed: Vec<Refusal> = readiness
        .into_iter()
        .flat_map(|readiness| readiness.failed.iter())
        .filter(|(by, _)| by == brick)
        .map(|(_, error)| Refusal::of(error))
        .collect();
    let prefix = format!("$.bricks.{brick}.");
    let problems: Vec<Refusal> = registry
        .problems()
        .iter()
        .chain(registry.unmet())
        .filter(|problem| problem.path.starts_with(&prefix))
        .map(Refusal::of)
        .collect();
    Ok(BrickWhy {
        brick: brick.to_owned(),
        doing: state.is_some_and(engine_registry::BrickState::on) && failed.is_empty(),
        state: state.map(engine_registry::BrickState::view),
        needs,
        waiting,
        failed,
        problems,
    })
}
