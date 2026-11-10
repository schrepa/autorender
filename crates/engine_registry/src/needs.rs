use engine_place::{AttachPoint, PassEntry};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadKind {
    Slot,
    Face,
    Target,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Read {
    pub reader: &'static str,
    pub kind: ReadKind,
    pub place: String,
    pub providers: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub(crate) struct LeftOut {
    pub(crate) package: &'static str,
    pub(crate) places: Vec<&'static str>,
    pub(crate) passes: Vec<PassEntry>,
}

impl LeftOut {
    pub(crate) fn provides(&self, place: &str) -> bool {
        self.places.contains(&place)
    }

    pub(crate) fn writes(&self, target: &str, before: Option<AttachPoint>) -> bool {
        self.passes.iter().any(|entry| {
            before.is_none_or(|at| entry.pass.at < at)
                && entry
                    .pass
                    .writes
                    .targets()
                    .iter()
                    .any(|written| written.name() == target)
        })
    }
}
