use bevy_ecs::prelude::*;
use engine_settings::settings;
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::{Parent, Transform};

settings! {
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct Grounded {
        pub offset: f32 = 0.0 => range(-10_000.0, 10_000.0),
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct GroundedAt {
    pub position: Vec2,
    pub offset: f32,
    pub revision: u64,
    pub level: Option<u32>,
}

pub(crate) type GroundedQuery = QueryState<
    (
        Entity,
        &'static Grounded,
        &'static Transform,
        Option<&'static GroundedAt>,
        Option<&'static crate::HeightHeld>,
    ),
    Without<Parent>,
>;

pub(crate) fn settle(world: &mut World, query: &mut GroundedQuery) {
    let Some(ground) = world.get_resource::<crate::Ground>().copied() else {
        return;
    };
    let revision = (ground.revision)(world);
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
        let sample = (ground.sample)(world, position.x, position.y);
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
