use bevy_ecs::component::ComponentId;
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

use super::{GlobalTransforms, tracking};
use crate::moves::MovingKinds;
use crate::{Mobility, Mover, Node, NodeWord};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Declared {
    node: Node,
    owned: Mobility,
}

impl Declared {
    fn read(world: &World, entity: Entity, kinds: &[ComponentId]) -> Self {
        let Ok(held) = world.get_entity(entity) else {
            return Self::default();
        };
        let moved = held.contains::<Mover>() || kinds.iter().any(|&kind| held.contains_id(kind));
        Self {
            node: held.get::<Node>().copied().unwrap_or_default(),
            owned: if moved {
                Mobility::Moving
            } else {
                Mobility::Static
            },
        }
    }

    fn word(self, parent: Option<NodeWord>) -> NodeWord {
        NodeWord::of(self.node, self.owned, parent)
    }
}

impl GlobalTransforms {
    pub(super) fn declare(&mut self, world: &mut World, rebuilt: bool) -> bool {
        if !world
            .resource::<tracking::Invalidation>()
            .touched
            .is_empty()
        {
            let mut invalidation = world.resource_mut::<tracking::Invalidation>();
            std::mem::swap(&mut self.touched, &mut invalidation.touched);
        }
        let world: &World = world;
        let kinds = world
            .get_resource::<MovingKinds>()
            .map_or(&[][..], |kinds| kinds.0.as_slice());
        let mut changed = rebuilt;
        if rebuilt {
            for (declared, &entity) in self.declared.iter_mut().zip(&self.entities) {
                *declared = Declared::read(world, entity, kinds);
            }
        } else {
            for &entity in &self.touched {
                let Some(&index) = self.indices.get(&entity) else {
                    continue;
                };
                let next = Declared::read(world, entity, kinds);
                if self.declared[index] != next {
                    self.declared[index] = next;
                    changed = true;
                }
            }
        }
        self.touched.clear();
        changed
    }

    pub(super) fn settle_words(&mut self, rebuilt: bool) -> bool {
        let _profile = engine_profile::span("hierarchy", "settle_words");
        let threshold = self.dense_threshold();
        let mut changed = rebuilt;
        for position in 0..self.order.len() {
            let index = self.order[position];
            let link = self.links[index];
            let parent = link
                .parent()
                .filter(|_| link.topology_valid())
                .map(|parent| self.words[parent]);
            let next = self.declared[index].word(parent);
            let before = self.words[index];
            if next == before {
                continue;
            }
            self.words[index] = next;
            if rebuilt {
                continue;
            }
            changed = true;
            if !before.active && next.active {
                self.dirty.mark(index, threshold);
                self.woken.push(self.entities[index]);
            }
        }
        if rebuilt {
            for entity in self.asleep.drain(..) {
                if let Some(&index) = self.indices.get(&entity)
                    && self.words[index].active
                {
                    self.woken.push(entity);
                }
            }
        }
        if changed {
            self.words_revision = self.words_revision.wrapping_add(1);
        }
        changed
    }

    pub(crate) fn take_woken(&mut self) -> Vec<Entity> {
        std::mem::take(&mut self.woken)
    }

    pub(crate) fn has_woken(&self) -> bool {
        !self.woken.is_empty()
    }

    pub fn word(&self, entity: Entity) -> Option<NodeWord> {
        let index = *self.indices.get(&entity)?;
        Some(self.words[index])
    }

    pub fn word_at(&self, index: usize) -> Option<NodeWord> {
        self.words.get(index).copied()
    }

    pub fn words_revision(&self) -> u64 {
        self.words_revision
    }

    pub fn active(&self, entity: Entity) -> bool {
        self.word(entity).is_none_or(|word| word.active)
    }

    pub fn drawn(&self, entity: Entity) -> bool {
        self.word(entity).is_none_or(NodeWord::drawn)
    }

    pub fn mobility(&self, entity: Entity) -> Option<Mobility> {
        self.word(entity).map(|word| word.mobility)
    }
}
