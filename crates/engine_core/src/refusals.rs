use std::collections::BTreeMap;

use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use serde::Serialize;

use crate::Error;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Refused {
    pub part: &'static str,
    pub reason: Error,
}

#[derive(Resource, Default, Debug)]
pub struct Refusals {
    held: BTreeMap<Entity, Vec<Refused>>,
    revision: u64,
}

impl Refusals {
    pub fn refuse(&mut self, entity: Entity, part: &'static str, reason: Error) {
        let held = self.held.entry(entity).or_default();
        match held.iter_mut().find(|refused| refused.part == part) {
            Some(refused) if refused.reason == reason => return,
            Some(refused) => refused.reason = reason,
            None => held.push(Refused { part, reason }),
        }
        self.revision = self.revision.wrapping_add(1);
    }

    pub fn accept(&mut self, entity: Entity, part: &'static str) {
        let Some(held) = self.held.get_mut(&entity) else {
            return;
        };
        let before = held.len();
        held.retain(|refused| refused.part != part);
        if held.len() == before {
            return;
        }
        if held.is_empty() {
            self.held.remove(&entity);
        }
        self.revision = self.revision.wrapping_add(1);
    }

    pub fn judge(&mut self, entity: Entity, part: &'static str, verdict: Result<(), Error>) {
        match verdict {
            Ok(()) => self.accept(entity, part),
            Err(reason) => self.refuse(entity, part, reason),
        }
    }

    pub fn keep_part(&mut self, part: &'static str, mut keep: impl FnMut(Entity) -> bool) {
        let mut changed = false;
        self.held.retain(|&entity, held| {
            if !keep(entity) {
                let before = held.len();
                held.retain(|refused| refused.part != part);
                changed |= held.len() != before;
            }
            !held.is_empty()
        });
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    pub fn of(&self, entity: Entity) -> &[Refused] {
        self.held.get(&entity).map_or(&[], Vec::as_slice)
    }

    pub fn reason(&self, entity: Entity, part: &str) -> Option<&Error> {
        self.of(entity)
            .iter()
            .find(|refused| refused.part == part)
            .map(|refused| &refused.reason)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Entity, &Refused)> + '_ {
        self.held
            .iter()
            .flat_map(|(&entity, held)| held.iter().map(move |refused| (entity, refused)))
    }

    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.held.keys().copied()
    }

    pub fn len(&self) -> usize {
        self.held.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }
}

pub(crate) fn prune(world: &mut World) {
    let dead: Vec<Entity> = world.get_resource::<Refusals>().map_or_else(Vec::new, |refusals| {
        refusals
            .entities()
            .filter(|&entity| !world.entities().contains(entity))
            .collect()
    });
    if dead.is_empty() {
        return;
    }
    let mut refusals = world.resource_mut::<Refusals>();
    for entity in dead {
        refusals.held.remove(&entity);
    }
    refusals.revision = refusals.revision.wrapping_add(1);
}
