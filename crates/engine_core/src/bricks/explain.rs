use std::panic::Location;

use bevy_ecs::prelude::{Entity, World};
use serde::Serialize;
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawState {
    Unread,
    Culled,
    Level(u32),
}

#[derive(Clone, Debug)]
pub struct ExplainedEntity {
    pub id: String,
    pub entity: Entity,
    pub slot: Option<u32>,
    pub drawn: DrawState,
}

#[derive(Clone, Debug)]
pub enum ExplainRequest {
    Entity(ExplainedEntity),
    Frame { stats: Value },
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
            serde_json::to_value(value).expect("facts serialize"),
        );
        self
    }
}

pub type Explainer = fn(&World, &ExplainRequest) -> Option<ExplainRecord>;

#[derive(Clone, Copy)]
pub struct ExplainEntry {
    pub brick: &'static str,
    pub at: &'static Location<'static>,
    explain: Explainer,
}

impl ExplainEntry {
    pub(super) fn of(
        brick: &'static str,
        explain: Explainer,
        at: &'static Location<'static>,
    ) -> Self {
        Self { brick, at, explain }
    }

    pub fn explain(&self, world: &World, request: &ExplainRequest) -> Option<ExplainRecord> {
        (self.explain)(world, request).map(|record| record.of(self.brick))
    }
}

pub fn source_line(at: &Location<'_>) -> String {
    format!("{}:{}", at.file().replace('\\', "/"), at.line())
}
