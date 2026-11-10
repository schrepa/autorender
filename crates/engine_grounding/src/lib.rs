#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Error, ErrorCode, Expected, GlobalTransforms, Mover, Moves, Pace,
    Parent, Placement, Refusals, Rule, SceneComponents, Schedule, Setting, Settling, Slot, Source,
    Transform, mover_of, settings,
};
use engine_places_simulation::ground::{
    Ground, GroundPoint, GroundProbe, GroundRead, GroundSurface, Grounded, GroundedAt,
};
use engine_places_simulation::navigation::{NavStep, NavWalk};
use glam::Vec2;
use serde::Serialize;
use serde_json::Value;

const OFFSET: Rule = Rule::range(-10_000.0, 10_000.0);

pub const INVALID_GROUNDED: ErrorCode = ErrorCode::brick("invalid_grounded");

pub const GROUNDED_PART: &str = "grounded";

settings! {
    #[section("grounding")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneGrounding")]
    pub struct Grounding {
        pub deck_above: f32 = 1.0 => range(0.0, 10_000.0),
    }
}

#[derive(Resource, Default)]
struct GroundWaits {
    settling: Settling,
}

pub struct GroundingBrick;

impl Brick for GroundingBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Grounding>()
            .requires::<Ground>()
            .resource::<GroundWaits>()
            .errors(&[INVALID_GROUNDED])
            .join::<SceneComponents>(
                ComponentKind::of::<Grounded>(GROUNDED_PART)
                    .check(check_grounded)
                    .placement(Placement::Roots(INVALID_GROUNDED))
                    .schema(grounded_schema)
                    .view::<GroundedAt, GroundedView>(grounded_view),
                None,
            )
            .moves::<Grounded>(Pace::Tick)
            .systems(Schedule::FixedSimulation, steer)
            .systems(Schedule::WorldSync, settle)
            .systems(Schedule::ViewSync, count_waiting);
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
        GroundedAt::Unsettled | GroundedAt::Waiting => false,
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

struct Placed {
    at: GroundedAt,
    height: Option<f32>,
    verdict: Result<(), Error>,
}

fn unplaced(read: GroundRead<Option<GroundPoint>>, position: Vec2) -> Error {
    match read {
        GroundRead::Off => Error::new(ErrorCode::PlaceNotFilled, "$.grounded")
            .expected(Expected::Present)
            .got(Ground::NAME),
        GroundRead::Ready(_) | GroundRead::NotLoaded => {
            Error::new(ErrorCode::InvalidPosition, "$.grounded").got(position.to_array())
        }
    }
}

fn ground_under(
    (ground, settings): (&Ground, &Grounding),
    (entity, at, feet): (Entity, GroundedAt, f32),
    (position, offset): (Vec2, f32),
    walked: Option<f32>,
) -> Option<Placed> {
    let revision = ground.revision();
    if holds(at, (position, offset), revision) {
        return None;
    }
    let probe = GroundProbe::at(position).excluding(entity);
    let near = probe.below(walked.unwrap_or(feet) + settings.deck_above);
    let read = match ground.down(near) {
        GroundRead::Ready(None) if walked.is_none() => ground.down(probe),
        read @ (GroundRead::Ready(_) | GroundRead::NotLoaded | GroundRead::Off) => read,
    };
    let stood = |surface: Option<GroundSurface>, height: f32| Placed {
        at: GroundedAt::Stood {
            position,
            offset,
            surface,
            revision,
        },
        height: Some(height + offset),
        verdict: Ok(()),
    };
    Some(match (read, walked) {
        (GroundRead::Ready(Some(point)), _) => stood(Some(point.surface), point.height),
        (GroundRead::Ready(None) | GroundRead::NotLoaded | GroundRead::Off, Some(walked)) => {
            stood(None, walked)
        }
        (GroundRead::NotLoaded, None) => Placed {
            at: GroundedAt::Waiting,
            height: None,
            verdict: Ok(()),
        },
        (GroundRead::Ready(None) | GroundRead::Off, None) => Placed {
            at: GroundedAt::Unsupported {
                position,
                offset,
                revision,
            },
            height: None,
            verdict: Err(unplaced(read, position)),
        },
    })
}

fn stand(
    (ground, settings): (&Ground, &Grounding),
    (entity, refusals): (Entity, &mut Refusals),
    transform: &mut Mut<'_, Transform>,
    (grounded, at): (&Grounded, &mut Mut<'_, GroundedAt>),
    walked: Option<f32>,
) {
    let position = Vec2::new(transform.translation.x, transform.translation.z);
    let feet = transform.translation.y - grounded.offset;
    let Some(placed) = ground_under(
        (ground, settings),
        (entity, **at, feet),
        (position, grounded.offset),
        walked,
    ) else {
        return;
    };
    if let Some(height) = placed.height
        && transform.translation.y != height
    {
        transform.translation.y = height;
    }
    **at = placed.at;
    refusals.judge(entity, GROUNDED_PART, placed.verdict);
}

fn steer(
    (ground, settings): (Res<Ground>, Res<Grounding>),
    mut moves: Moves<GroundingBrick>,
    mut query: SteeredQuery,
    mut refusals: ResMut<Refusals>,
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
            (entity, &mut refusals),
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
    mut refusals: ResMut<Refusals>,
) {
    for (entity, grounded, mut at, walk) in &mut query {
        let Some(mut transform) = moves.get_mut(entity) else {
            continue;
        };
        let position = Vec2::new(transform.translation.x, transform.translation.z);
        let walked = walk.and_then(|walk| walk.walked(position));
        stand(
            (&ground, &settings),
            (entity, &mut refusals),
            &mut transform,
            (grounded, &mut at),
            walked,
        );
    }
    refusals.keep_part(GROUNDED_PART, |entity| query.contains(entity));
}

type GroundedRoots<'w, 's> =
    Query<'w, 's, (Entity, &'static GroundedAt, &'static Mover), (With<Grounded>, Without<Parent>)>;

fn count_waiting(waits: Res<GroundWaits>, globals: Res<GlobalTransforms>, grounded: GroundedRoots) {
    let by = mover_of::<GroundingBrick>();
    let waiting = grounded
        .iter()
        .filter(|&(entity, at, mover)| {
            mover.by == by
                && globals.active(entity)
                && matches!(at, GroundedAt::Unsettled | GroundedAt::Waiting)
        })
        .count();
    waits
        .settling
        .left(u32::try_from(waiting).unwrap_or(u32::MAX));
}
