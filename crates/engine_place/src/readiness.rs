use bevy_ecs::resource::Resource;
use serde::Serialize;

use crate::{Error, Progress};

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Readiness {
    pub loading: Progress,
    pub compiling: u32,
    pub settling: u32,
    pub failed: Vec<(String, Error)>,
}

impl Readiness {
    pub fn settled(&self) -> bool {
        self.loading.total == 0 && self.compiling == 0 && self.settling == 0
    }

    pub fn settle(&mut self, left: u32) {
        self.settling = self.settling.saturating_add(left);
    }

    pub fn fail(&mut self, by: &str, error: Error) {
        self.failed.push((by.to_owned(), error));
    }
}
