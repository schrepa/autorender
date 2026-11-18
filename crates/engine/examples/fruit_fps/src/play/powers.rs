use engine::prelude::{Component, Resource};
use glam::Vec3;

use crate::abilities::{
    self, AbilityId, AbilitySpec, ActorId, Aim, Danger, Drawable, EffectId, Effects, Modifiers,
    Preview, WorldHit,
};
use crate::play::world::Arena;
use crate::rules::{AbilitySlot, Team};
use crate::solids::{Hit, Kind};

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
    pub surfaces: Vec<(Kind, f32)>,
}

pub fn bounce_of(surfaces: &[(Kind, f32)], kind: Kind) -> f32 {
    surfaces
        .iter()
        .find(|(held, _)| *held == kind)
        .map_or(1.0, |(_, bounce)| *bounce)
}

pub fn world_hit(surfaces: &[(Kind, f32)], hit: &Hit) -> WorldHit {
    WorldHit {
        distance: hit.distance,
        normal: hit.normal,
        bounce: bounce_of(surfaces, hit.kind),
    }
}

pub struct ArenaView<'a> {
    pub arena: &'a Arena,
    pub surfaces: &'a [(Kind, f32)],
}

impl abilities::World for ArenaView<'_> {
    fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<WorldHit> {
        self.arena
            .ray(origin, direction, max)
            .map(|hit| world_hit(self.surfaces, &hit))
    }

    fn sphere_sweep(&self, centre: Vec3, radius: f32, motion: Vec3) -> Option<WorldHit> {
        self.arena
            .solids()
            .sphere_sweep(centre, radius, motion)
            .map(|hit| world_hit(self.surfaces, &hit))
    }

    fn add_box(
        &mut self,
        _id: u32,
        _ability: AbilityId,
        _centre: Vec3,
        _half_extents: Vec3,
        _axes: glam::Mat3,
    ) {
    }

    fn remove_box(&mut self, _id: u32) {}
}

impl Default for Powers {
    fn default() -> Self {
        Powers {
            effects: Effects::new(SOLID_ID_BASE),
            actors: Vec::new(),
            struck: Vec::new(),
            held: Vec::new(),
            surfaces: Vec::new(),
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

    pub fn landing(
        &self,
        arena: &Arena,
        caster: &abilities::Actor,
        spec: &AbilitySpec,
        aim: Aim,
    ) -> Option<Preview> {
        let view = ArenaView {
            arena,
            surfaces: &self.surfaces,
        };
        abilities::preview(&view, caster, spec, aim)
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
