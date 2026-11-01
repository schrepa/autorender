#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Error, ErrorCode, Expected, Parent, Placed, Rule, SceneComponents,
    Schedule, Setting, Source, Transform,
};
use engine_places::ground::{Ground, Grounded, GroundedAt, HeightHeld};
use glam::Vec2;
use serde::Serialize;
use serde_json::{Value, json};

const OFFSET: Rule = Rule::range(-10_000.0, 10_000.0);

pub struct GroundingBrick;

impl Brick for GroundingBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Grounded>("grounded")
                    .check(check_grounded)
                    .check_placed(placed_grounded)
                    .schema(grounded_schema)
                    .example(grounded_example)
                    .view::<GroundedAt, GroundedView>(grounded_view),
                None,
            )
            .systems(Schedule::WorldSync, settle);
    }
}

#[derive(Serialize)]
pub struct GroundedView {
    pub offset: f32,
    pub level: Option<u32>,
}

fn grounded_view(grounded: &Grounded, at: Option<&GroundedAt>) -> GroundedView {
    GroundedView {
        offset: grounded.offset,
        level: at.and_then(|at| at.level),
    }
}

fn check_grounded(grounded: &Grounded, path: &str) -> Result<(), Error> {
    grounded
        .offset
        .check("offset", OFFSET)
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

fn placed_grounded(_: &Grounded, placed: Placed, path: &str) -> Result<(), Error> {
    if placed.root {
        return Ok(());
    }
    Err(Error::new(ErrorCode::InvalidGrounded, path).expected(Expected::Absent))
}

fn grounded_schema() -> Value {
    let mut schema = schemars::schema_for!(Grounded);
    engine_core::state(&mut schema, "offset", OFFSET);
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

fn grounded_example() -> Value {
    json!({})
}

type GroundedQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static Grounded,
        &'static mut Transform,
        &'static mut GroundedAt,
        Option<&'static HeightHeld>,
    ),
    Without<Parent>,
>;

fn settle(ground: Res<Ground>, mut query: GroundedQuery) {
    let revision = ground.revision();
    for (grounded, mut transform, mut at, held) in &mut query {
        if held.is_some_and(|held| held.0) {
            continue;
        }
        let position = Vec2::new(transform.translation.x, transform.translation.z);
        let stale = *at == GroundedAt::default()
            || at.position != position
            || at.offset != grounded.offset
            || (at.revision != revision && at.level != Some(0));
        if !stale {
            continue;
        }
        let sample = ground.sample(position);
        if let Some(height) = sample.map(|sample| sample.height + grounded.offset)
            && transform.translation.y != height
        {
            transform.translation.y = height;
        }
        *at = GroundedAt {
            position,
            offset: grounded.offset,
            revision,
            level: sample.map(|sample| sample.level),
        };
    }
}
