//! Grounded objects: entities standing on the terrain, as
//! authors and generators place props by x and z alone. At extraction,
//! before transforms refresh, a [`Grounded`] entity's translation rises or
//! sinks to its offset above the ground under it, sampled from the finest
//! terrain tile resident there; it samples again as finer tiles arrive,
//! and when its x, z or offset changes. Streamed items
//! stand on the ground the frame they spawn.

use bevy_ecs::prelude::*;
use engine_settings::settings;
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::{Parent, Terrains, Transform};

settings! {
    /// Stands its entity on the terrain: its translation's y is
    /// kept `offset` meters above the ground under its x and z. Unparented
    /// entities only; without a resident tile under it (or over a hole) the
    /// entity keeps its translation, as walking navigation agents keep theirs:
    /// their walks stand them, on bridges' decks too. Requires
    /// version 9.
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct Grounded {
        /// Meters above the ground its translation stands.
        pub offset: f32 = 0.0 => range(-10_000.0, 10_000.0),
    }
}

/// Where a grounded entity last sampled the ground: its x and z, the
/// offset it applied, the terrains' revision then, and the level of the
/// tile that answered (`None` while none did).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct GroundedAt {
    pub position: Vec2,
    pub offset: f32,
    pub revision: u64,
    pub level: Option<u32>,
}

/// The unparented grounded entities, which [`Grounding::settle`] reads
/// each extraction.
pub(crate) struct Grounding {
    query: GroundedQuery,
}

type GroundedQuery = QueryState<
    (
        Entity,
        &'static Grounded,
        &'static Transform,
        Option<&'static GroundedAt>,
        Option<&'static crate::NavWalk>,
    ),
    Without<Parent>,
>;

impl Grounding {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }

    /// Stands every grounded entity that moved, or whose ground may have
    /// become finer, on the terrain.
    pub(crate) fn settle(&mut self, world: &mut World) {
        let Some(terrains) = world.get_resource::<Terrains>() else {
            return;
        };
        let revision = terrains.revision();
        let mut settled = Vec::new();
        for (entity, grounded, transform, at, walk) in self.query.iter(world) {
            if walk.is_some_and(|walk| walk.status == crate::NavStatus::Walking) {
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
            let sample = terrains
                .sample(position.x, position.y)
                .filter(|sample| !sample.hole);
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
}
