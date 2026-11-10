#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Error, ErrorCode, Moves, Pace, Parent, Placement, Rule,
    SceneComponents, Schedule, Setting, Source, Transform, settings,
};
use engine_places_simulation::ground::{
    Ground, GroundProbe, GroundSurface, Grounded, GroundedAt,
};
use engine_places_simulation::navigation::{NavStep, NavWalk};
use glam::Vec2;
use serde::Serialize;
use serde_json::Value;

const OFFSET: Rule = Rule::range(-10_000.0, 10_000.0);

pub const INVALID_GROUNDED: ErrorCode = ErrorCode::brick("invalid_grounded");

settings! {
    #[section("grounding")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneGrounding")]
    pub struct Grounding {
        pub deck_above: f32 = 1.0 => range(0.0, 10_000.0),
    }
}

pub struct GroundingBrick;

impl Brick for GroundingBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Grounding>()
            .errors(&[INVALID_GROUNDED])
            .join::<SceneComponents>(
                ComponentKind::of::<Grounded>("grounded")
                    .check(check_grounded)
                    .placement(Placement::Roots(INVALID_GROUNDED))
                    .schema(grounded_schema)
                    .view::<GroundedAt, GroundedView>(grounded_view),
                None,
            )
            .moves::<Grounded>(Pace::Tick)
            .systems(Schedule::FixedSimulation, steer)
            .systems(Schedule::WorldSync, settle);
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct GroundedView {
    pub offset: f32,
    pub surface: Option<GroundSurface>,
}

fn grounded_view(grounded: &Grounded, at: Option<&GroundedAt>) -> GroundedView {
    GroundedView {
        offset: grounded.offset,
        surface: at.and_then(|at| at.surface()),
    }
}

fn check_grounded(grounded: &Grounded, path: &str) -> Result<(), Error> {
    grounded
        .offset
        .check("offset", OFFSET)
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

fn grounded_schema() -> Value {
    let mut schema = schemars::schema_for!(Grounded);
    engine_core::state(&mut schema, "offset", OFFSET);
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

type GroundedQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Grounded,
        &'static mut GroundedAt,
        Option<&'static NavWalk>,
    ),
    Without<Parent>,
>;

type SteeredQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Grounded,
        &'static mut GroundedAt,
        &'static mut NavWalk,
    ),
    Without<Parent>,
>;

fn holds(at: GroundedAt, (position, offset): (Vec2, f32), revision: u64) -> bool {
    match at {
        GroundedAt::Unsettled => false,
        GroundedAt::Stood {
            position: held,
            offset: kept,
            revision: seen,
            ..
        }
        | GroundedAt::Unsupported {
            position: held,
            offset: kept,
            revision: seen,
        } => held == position && kept == offset && seen == revision,
    }
}

fn ground_under(
    (ground, settings): (&Ground, &Grounding),
    (entity, at, feet): (Entity, GroundedAt, f32),
    (position, offset): (Vec2, f32),
    walked: Option<f32>,
) -> Option<(GroundedAt, Option<f32>)> {
    let revision = ground.revision();
    if holds(at, (position, offset), revision) {
        return None;
    }
    let probe = GroundProbe::at(position).excluding(entity);
    let near = probe.below(walked.unwrap_or(feet) + settings.deck_above);
    let floor = match ground.down(near).ready().flatten() {
        Some(point) => Some(point),
        None if walked.is_none() => ground.down(probe).ready().flatten(),
        None => None,
    };
    let stood = match (floor, walked) {
        (Some(point), _) => Some((point.height, Some(point.surface))),
        (None, Some(walked)) => Some((walked, None)),
        (None, None) => None,
    };
    Some(match stood {
        Some((height, surface)) => (
            GroundedAt::Stood {
                position,
                offset,
                surface,
                revision,
            },
            Some(height + offset),
        ),
        None => (
            GroundedAt::Unsupported {
                position,
                offset,
                revision,
            },
            None,
        ),
    })
}

fn stand(
    (ground, settings): (&Ground, &Grounding),
    entity: Entity,
    transform: &mut Mut<'_, Transform>,
    (grounded, at): (&Grounded, &mut Mut<'_, GroundedAt>),
    walked: Option<f32>,
) {
    let position = Vec2::new(transform.translation.x, transform.translation.z);
    let feet = transform.translation.y - grounded.offset;
    let Some((next, height)) = ground_under(
        (ground, settings),
        (entity, **at, feet),
        (position, grounded.offset),
        walked,
    ) else {
        return;
    };
    if let Some(height) = height
        && transform.translation.y != height
    {
        transform.translation.y = height;
    }
    **at = next;
}

fn steer(
    (ground, settings): (Res<Ground>, Res<Grounding>),
    mut moves: Moves<GroundingBrick>,
    mut query: SteeredQuery,
) {
    for (entity, grounded, mut at, mut walk) in &mut query {
        let Some(step) = walk.step.filter(|step| !step.taken) else {
            continue;
        };
        let Some(mut transform) = moves.get_mut(entity) else {
            continue;
        };
        walk.step = Some(NavStep {
            taken: true,
            ..step
        });
        transform.translation.x = step.to.x;
        transform.translation.z = step.to.z;
        if transform.rotation != step.facing {
            transform.rotation = step.facing;
        }
        stand(
            (&ground, &settings),
            entity,
            &mut transform,
            (grounded, &mut at),
            Some(step.to.y),
        );
    }
}

fn settle(
    (ground, settings): (Res<Ground>, Res<Grounding>),
    mut moves: Moves<GroundingBrick>,
    mut query: GroundedQuery,
) {
    for (entity, grounded, mut at, walk) in &mut query {
        let Some(mut transform) = moves.get_mut(entity) else {
            continue;
        };
        let position = Vec2::new(transform.translation.x, transform.translation.z);
        let walked = walk.and_then(|walk| walk.walked(position));
        stand(
            (&ground, &settings),
            entity,
            &mut transform,
            (grounded, &mut at),
            walked,
        );
    }
}
