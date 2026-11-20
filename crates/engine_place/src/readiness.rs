use bevy_ecs::resource::Resource;
use engine_assets::GroupReport;
use serde::Serialize;

use crate::{Error, Progress};

pub const CORE_ROW: &str = "core";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct Unsettled {
    pub brick: &'static str,
    pub left: u64,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "dev", derive(schemars::JsonSchema))]
pub struct Readiness {
    pub loading: Progress,
    pub compiling: u32,
    pub settling: u32,
    pub unsettled: Vec<Unsettled>,
    pub failed: Vec<(String, Error)>,
    pub groups: Vec<GroupReport>,
}

impl Readiness {
    pub fn settled(&self) -> bool {
        self.loading.total == 0 && self.compiling == 0 && self.settling == 0
    }

    pub fn settle(&mut self, brick: &'static str, left: u32) {
        self.settling = self.settling.saturating_add(left);
        self.note(brick, u64::from(left));
    }

    pub fn load(&mut self, brick: &'static str, loading: Progress) {
        self.loading = self.loading.and(loading);
        let left = loading.left().max(u64::from(loading.total > 0));
        self.note(brick, left);
    }

    pub fn fail(&mut self, by: &str, error: Error) {
        self.failed.push((by.to_owned(), error));
    }

    fn note(&mut self, brick: &'static str, left: u64) {
        if left == 0 {
            return;
        }
        match self.unsettled.iter_mut().find(|row| row.brick == brick) {
            Some(row) => row.left = row.left.saturating_add(left),
            None => self.unsettled.push(Unsettled { brick, left }),
        }
    }
}
