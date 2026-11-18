use engine::prelude::{Component, Resource};
use glam::Vec3;

use crate::abilities::{ActorId, Danger, Drawable, EffectId, Effects, Modifiers, Preview};
use crate::rules::{AbilitySlot, Team};

pub const SOLID_ID_BASE: u32 = 0;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct Push(pub Vec3);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlotView {
    pub name: String,
    pub block: String,
    pub charges: u32,
    pub max: u32,
    pub cooldown_ticks: u32,
    pub ready: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PowerActor {
    pub slot: u32,
    pub team: Team,
    pub blindness: f32,
    pub sight_limit_m: Option<f32>,
    pub modifiers: Modifiers,
    pub revealed_to: [bool; 2],
    pub slots: [SlotView; 4],
    pub held: Option<AbilitySlot>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Held {
    pub slot: u32,
    pub ability: AbilitySlot,
    pub name: String,
    pub since: u64,
    pub preview: Option<Preview>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectHit {
    pub effect: EffectId,
    pub shooter: u32,
    pub damage: f32,
}

#[derive(Resource, Clone, Debug)]
pub struct Powers {
    pub effects: Effects,
    pub actors: Vec<PowerActor>,
    pub struck: Vec<EffectHit>,
    pub held: Vec<Held>,
}

impl Default for Powers {
    fn default() -> Self {
        Powers {
            effects: Effects::new(SOLID_ID_BASE),
            actors: Vec::new(),
            struck: Vec::new(),
            held: Vec::new(),
        }
    }
}

impl Powers {
    pub fn sight_blocked(&self, a: Vec3, b: Vec3) -> bool {
        self.effects.sight_blocked(a, b)
    }

    pub fn actor(&self, slot: u32) -> Option<&PowerActor> {
        self.actors.iter().find(|actor| actor.slot == slot)
    }

    pub fn held(&self, slot: u32) -> Option<&Held> {
        self.held.iter().find(|held| held.slot == slot)
    }

    pub fn previews(&self) -> Vec<&Preview> {
        self.held
            .iter()
            .filter_map(|held| held.preview.as_ref())
            .collect()
    }

    pub fn blindness(&self, slot: u32) -> f32 {
        self.effects.blindness(ActorId(slot))
    }

    pub fn sight_limit(&self, slot: u32) -> Option<f32> {
        self.effects.sight_limit(ActorId(slot))
    }

    pub fn modifiers(&self, slot: u32) -> Modifiers {
        self.effects.modifiers(ActorId(slot))
    }

    pub fn damage_taken(&self, slot: u32) -> f32 {
        self.modifiers(slot).damage_taken
    }

    pub fn revealed(&self, slot: u32, to: Team) -> bool {
        self.effects.revealed(ActorId(slot), to.index())
    }

    pub fn dangers(&self, team: Team) -> Vec<Danger> {
        self.effects.dangers(team.index())
    }

    pub fn drawables(&self) -> Vec<Drawable> {
        self.effects.drawables()
    }
}
