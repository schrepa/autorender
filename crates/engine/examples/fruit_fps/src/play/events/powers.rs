use engine::prelude::Resource;

use crate::abilities::Event;
use crate::rules::{AbilitySlot, AbilityUse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NotLoaded,
    NoMatch,
    Phase,
    Dead,
    UnknownCharacter,
    NoCharges,
    UltimateNotReady,
}

impl Refusal {
    pub fn key(self) -> &'static str {
        match self {
            Refusal::NotLoaded => "not_loaded",
            Refusal::NoMatch => "no_match",
            Refusal::Phase => "phase",
            Refusal::Dead => "dead",
            Refusal::UnknownCharacter => "unknown_character",
            Refusal::NoCharges => "no_charges",
            Refusal::UltimateNotReady => "ultimate_not_ready",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Refused {
    pub slot: u32,
    pub ability: AbilitySlot,
    pub reason: Refusal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cast {
    pub slot: u32,
    pub ability: AbilitySlot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    Equipped,
    Cancelled,
    Cast,
}

impl Hold {
    pub fn key(self) -> &'static str {
        match self {
            Hold::Equipped => "equipped",
            Hold::Cancelled => "cancelled",
            Hold::Cast => "cast",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoldChange {
    pub slot: u32,
    pub ability: AbilitySlot,
    pub change: Hold,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct PowerEvents {
    pub tick: u64,
    pub accepted: Vec<AbilityUse>,
    pub casts: Vec<Cast>,
    pub refused: Vec<Refused>,
    pub holds: Vec<HoldChange>,
    pub events: Vec<Event>,
}
