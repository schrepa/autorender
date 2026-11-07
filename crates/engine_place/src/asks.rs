use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use serde::Serialize;
use serde_json::{Value, json};

use crate::Slot;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ApplyAt {
    TickEnd,
    StepStart,
    TickStart,
    FrameStart,
}

pub trait Ask: Send + Sync + 'static {
    const NAME: &'static str;
    const AT: ApplyAt;

    fn node(&self) -> Option<Entity> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Moment {
    pub tick: u64,
    pub frame: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Applied {
    pub node: Option<u64>,
    pub asked: Moment,
    pub applied: Moment,
}

#[derive(Resource)]
pub struct Asked<A: Ask> {
    pending: Vec<(A, Moment)>,
    applied: Vec<Applied>,
}

impl<A: Ask> Default for Asked<A> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            applied: Vec::new(),
        }
    }
}

impl<A: Ask> Slot for Asked<A> {
    const NAME: &'static str = A::NAME;
}

impl<A: Ask> Asked<A> {
    pub fn ask(&mut self, ask: A, at: Moment) {
        self.pending.push((ask, at));
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn pending(&self) -> impl ExactSizeIterator<Item = &A> + '_ {
        self.pending.iter().map(|(ask, _)| ask)
    }

    pub fn applied(&self) -> &[Applied] {
        &self.applied
    }

    pub fn take(&mut self, now: Moment) -> Vec<A> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        self.applied.clear();
        let pending = std::mem::take(&mut self.pending);
        pending
            .into_iter()
            .map(|(ask, asked)| {
                self.applied.push(Applied {
                    node: ask.node().map(Entity::to_bits),
                    asked,
                    applied: now,
                });
                ask
            })
            .collect()
    }

    pub fn view(&self) -> Value {
        let pending: Vec<Value> = self
            .pending
            .iter()
            .map(|(ask, asked)| json!({ "node": ask.node().map(Entity::to_bits), "asked": asked }))
            .collect();
        json!({
            "name": A::NAME,
            "at": A::AT,
            "pending": pending,
            "applied": self.applied,
        })
    }
}
