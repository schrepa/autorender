use bevy_ecs::prelude::Resource;
use serde::Serialize;

use crate::input::{stacked, stamp};
use crate::{Interest, List, Requests, Ticket};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum CursorMode {
    Drag = 0,
    #[default]
    Free = 1,
    Captured = 2,
}

impl CursorMode {
    pub const ALL: [(Self, &'static str); 3] = [
        (Self::Drag, "drag"),
        (Self::Free, "free"),
        (Self::Captured, "captured"),
    ];

    pub const fn name(self) -> &'static str {
        Self::ALL[self as usize].1
    }
}

pub type CursorAsk = (&'static str, CursorMode);

#[derive(Resource, Debug, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CursorRequests {
    #[serde(skip)]
    requests: Requests<CursorAsk, Ticket>,
    applied: CursorMode,
}

impl List for CursorRequests {
    const NAME: &'static str = "cursor";
    type Member = CursorAsk;
}

impl CursorRequests {
    pub fn ask(&mut self, by: &'static str, mode: CursorMode) -> Interest<CursorAsk> {
        let interest = self.requests.ask((by, mode));
        stamp(&mut self.requests);
        interest
    }

    pub fn wanted(&self) -> CursorMode {
        stacked(&self.requests)
            .last()
            .map_or_else(CursorMode::default, |&(_, mode)| mode)
    }

    pub fn asked(&self) -> Vec<CursorAsk> {
        stacked(&self.requests)
    }

    pub fn applied(&self) -> CursorMode {
        self.applied
    }

    #[doc(hidden)]
    pub fn apply(&mut self) {
        self.requests.settle(0.0);
        stamp(&mut self.requests);
        self.applied = self.wanted();
    }
}
