use serde::Serialize;

use crate::BrickId;
use crate::state::GateId;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Read {
    pub reader: &'static str,
    pub place: String,
    pub providers: Vec<&'static str>,
    pub required: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RequiredFill {
    pub(crate) reader: BrickId,
    pub(crate) filler: GateId,
    pub(crate) place: &'static str,
}
