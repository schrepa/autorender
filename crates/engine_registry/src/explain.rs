use std::panic::Location;

use bevy_ecs::entity::Entity;
use bevy_ecs::query::FilteredAccessSet;
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::World;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::Registry;
use crate::entries::access_of;

#[derive(Clone, Copy, Debug)]
pub enum ExplainAsk<'a> {
    Entity { id: &'a str, entity: Entity },
    Frame { stats: &'a Value },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExplainRecord {
    pub brick: &'static str,
    pub decided: &'static str,
    #[serde(flatten)]
    pub facts: Map<String, Value>,
}

impl ExplainRecord {
    pub fn new(decided: &'static str) -> Self {
        Self {
            brick: "",
            decided,
            facts: Map::new(),
        }
    }

    #[must_use]
    pub fn of(mut self, brick: &'static str) -> Self {
        self.brick = brick;
        self
    }

    #[must_use]
    pub fn with(mut self, fact: &str, value: impl Serialize) -> Self {
        self.facts.insert(
            fact.to_owned(),
            serde_json::to_value(value).expect("facts_serialize"),
        );
        self
    }
}

#[derive(Clone, Copy)]
pub struct ExplainEntry {
    pub brick: &'static str,
    pub at: &'static Location<'static>,
    run: fn(&mut World, &ExplainAsk<'_>) -> Option<ExplainRecord>,
    access: fn(&mut World) -> FilteredAccessSet,
}

impl ExplainEntry {
    #[track_caller]
    pub fn of<P: SystemParam + 'static>(
        run: fn(&mut World, &ExplainAsk<'_>) -> Option<ExplainRecord>,
    ) -> Self {
        Self {
            brick: "",
            at: Location::caller(),
            run,
            access: access_of::<P>,
        }
    }

    pub fn explain(&self, world: &mut World, ask: &ExplainAsk<'_>) -> Option<ExplainRecord> {
        (self.run)(world, ask).map(|record| record.of(self.brick))
    }

    pub fn access(&self, world: &mut World) -> FilteredAccessSet {
        (self.access)(world)
    }
}

pub fn explain(world: &mut World, ask: &ExplainAsk<'_>) -> Vec<ExplainRecord> {
    let entries: Vec<ExplainEntry> = world
        .get_resource::<Registry>()
        .map(|registry| registry.explainers().copied().collect())
        .unwrap_or_default();
    entries
        .iter()
        .filter_map(|entry| entry.explain(world, ask))
        .collect()
}
