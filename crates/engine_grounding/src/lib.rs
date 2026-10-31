#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Join, Parent, Transform, WorldSync, WorldSyncSystems};
use engine_places::{Ground, HeightHeld};
use engine_scene::{GROUNDED, Grounded, SceneComponentsList};
use glam::Vec2;
use serde::Serialize;

pub struct GroundingBrick;

impl Brick for GroundingBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join(Join::<SceneComponentsList>::new().cpu(&GROUNDED))
            .systems(WorldSync, settle.in_set(WorldSyncSystems::Ground))
            .component_inspector::<Grounded, GroundedView>(grounded_view);
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct GroundedAt {
    pub position: Vec2,
    pub offset: f32,
    pub revision: u64,
    pub level: Option<u32>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GroundedView {
    pub offset: f32,
    pub level: Option<u32>,
}

fn grounded_view(world: &World, entity: Entity, grounded: &Grounded) -> GroundedView {
    let at = world.get::<GroundedAt>(entity);
    GroundedView {
        offset: grounded.offset,
        level: at.and_then(|at| at.level),
    }
}

pub(crate) type GroundedQuery = QueryState<
    (
        Entity,
        &'static Grounded,
        &'static Transform,
        Option<&'static GroundedAt>,
        Option<&'static HeightHeld>,
    ),
    Without<Parent>,
>;

pub(crate) fn settle(world: &mut World, query: &mut GroundedQuery) {
    if !world.contains_resource::<Ground>() {
        return;
    }
    let revision = Ground::revision_in(world);
    let mut settled = Vec::new();
    for (entity, grounded, transform, at, held) in query.iter(world) {
        if held.is_some_and(|held| held.0) {
            continue;
        }
        let position = Vec2::new(transform.translation.x, transform.translation.z);
        let stale = at.is_none_or(|at| {
            at.position != position
                || at.offset != grounded.offset
                || (at.revision != revision && at.level != Some(0))
        });
        if !stale {
            continue;
        }
        let sample = Ground::sample_in(world, position.x, position.y);
        settled.push((
            entity,
            sample.map(|sample| sample.height + grounded.offset),
            GroundedAt {
                position,
                offset: grounded.offset,
                revision,
                level: sample.map(|sample| sample.level),
            },
        ));
    }
    for (entity, height, at) in settled {
        let mut entity = world.entity_mut(entity);
        if let Some(height) = height
            && let Some(mut transform) = entity.get_mut::<Transform>()
            && transform.translation.y != height
        {
            transform.translation.y = height;
        }
        entity.insert(at);
    }
}
