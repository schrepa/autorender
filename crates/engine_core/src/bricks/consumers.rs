use serde::Serialize;

use super::List;
use crate::smoke::{SmokeEntry, SmokeResult, SmokeSession};

pub struct DetailConsumersList;

impl List for DetailConsumersList {
    const NAME: &'static str = "detail_consumers";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_detail_consumers",
        brick: Self::NAME,
        gpu: true,
        run: detail_consumers,
    }];
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Cut {
    pub asked: u32,
    pub kept: u32,
    pub cut: u32,
}

impl Cut {
    pub fn of(asked: u32, kept: u32) -> Self {
        let kept = kept.min(asked);
        Self {
            asked,
            kept,
            cut: asked - kept,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetailConsumer {
    pub name: &'static str,
    pub setting: &'static str,
    pub stat: &'static str,
}

fn detail_consumers(_session: &mut dyn SmokeSession) -> SmokeResult {
    Ok(())
}
