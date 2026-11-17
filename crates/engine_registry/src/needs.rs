use std::any::TypeId;

use serde::Serialize;

use crate::BrickId;
use crate::state::GateId;

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
    pub required: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PlaceUses {
    pub writers: Vec<&'static str>,
    pub readers: Vec<&'static str>,
}

impl PlaceUses {
    pub(crate) fn write(&mut self, brick: &'static str) {
        self.readers.retain(|&reader| reader != brick);
        if !self.writers.contains(&brick) {
            self.writers.push(brick);
        }
    }

    pub(crate) fn read(&mut self, brick: &'static str) {
        if !self.writers.contains(&brick) && !self.readers.contains(&brick) {
            self.readers.push(brick);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Accessed {
    pub(crate) place: TypeId,
    pub(crate) brick: &'static str,
    pub(crate) write: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RequiredFill {
    pub(crate) reader: BrickId,
    pub(crate) filler: GateId,
    pub(crate) place: &'static str,
}
