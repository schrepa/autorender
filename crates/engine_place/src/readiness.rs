use bevy_ecs::resource::Resource;
use serde::Serialize;

use crate::{Error, Progress};

pub const CORE_ROW: &str = "core";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Awaits {
    #[default]
    Work,
    View,
    Finished,
    Frames,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct Unsettled {
    pub brick: &'static str,
    pub left: u64,
    pub awaits: Awaits,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Readiness {
    pub loading: Progress,
    pub compiling: u32,
    pub settling: u32,
    pub finishing: u32,
    pub drawing: u32,
    pub unsettled: Vec<Unsettled>,
    pub failed: Vec<(String, Error)>,
}

impl Readiness {
    pub fn settled(&self) -> bool {
        self.worked() && self.drawing == 0
    }

    pub fn worked(&self) -> bool {
        self.loading.total == 0 && self.compiling == 0 && self.settling == 0
    }

    pub fn only_finishing(&self) -> bool {
        self.loading.total == 0 && self.settling == self.finishing
    }

    pub fn finish(&mut self, brick: &'static str, left: u32) {
        self.settling = self.settling.saturating_add(left);
        self.finishing = self.finishing.saturating_add(left);
        self.note(brick, u64::from(left), Awaits::Finished);
    }

    pub fn settle(&mut self, brick: &'static str, left: u32) {
        self.settling = self.settling.saturating_add(left);
        self.note(brick, u64::from(left), Awaits::Work);
    }

    pub fn load(&mut self, brick: &'static str, loading: Progress) {
        self.loading = self.loading.and(loading);
        let left = loading.left().max(u64::from(loading.total > 0));
        self.note(brick, left, Awaits::Work);
    }

    pub fn await_view(&mut self, brick: &'static str, left: u32) {
        self.note(brick, u64::from(left), Awaits::View);
    }

    pub fn await_frames(&mut self, brick: &'static str, left: u32) {
        self.drawing = self.drawing.saturating_add(left);
        self.note(brick, u64::from(left), Awaits::Frames);
    }

    pub fn fail(&mut self, by: &str, error: Error) {
        self.failed.push((by.to_owned(), error));
    }

    fn note(&mut self, brick: &'static str, left: u64, awaits: Awaits) {
        if left == 0 {
            return;
        }
        match self
            .unsettled
            .iter_mut()
            .find(|row| row.brick == brick && row.awaits == awaits)
        {
            Some(row) => row.left = row.left.saturating_add(left),
            None => self.unsettled.push(Unsettled {
                brick,
                left,
                awaits,
            }),
        }
    }
}
