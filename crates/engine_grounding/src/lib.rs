#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Parent, SceneComponents, Schedule, Source, Transform,
};
use engine_places::ground::{Ground, Grounded, GroundedAt, HeightHeld};
use glam::Vec2;

pub struct GroundingBrick;

impl Brick for GroundingBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(ComponentKind::of::<Grounded>("grounded"), None)
            .systems(Schedule::WorldSync, settle);
    }
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

fn settle(ground: Res<Ground>, mut revision: Local<u64>, mut query: GroundedQuery) {
    if ground.is_changed() {
        *revision += 1;
    }
    let revision = *revision;
    for (grounded, mut transform, mut at, held) in &mut query {
        if held.is_some_and(|held| held.0) {
            continue;
        }
        let position = Vec2::new(transform.translation.x, transform.translation.z);
        let stale = at.position != position
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
