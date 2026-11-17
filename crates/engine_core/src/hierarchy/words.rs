use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

use super::{GlobalTransforms, tracking};
use crate::{Mobility, Mover, Node, NodeWord};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Declared {
    node: Node,
    owned: Mobility,
}

impl Declared {
    pub(super) fn read(world: &World, entity: Entity) -> Self {
        let Ok(held) = world.get_entity(entity) else {
            return Self::default();
        };
        Self {
            node: held.get::<Node>().copied().unwrap_or_default(),
            owned: if held.contains::<Mover>() {
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
    pub(super) fn declare(&mut self, world: &mut World) {
        if !world
            .resource::<tracking::Invalidation>()
            .touched
            .is_empty()
        {
            let mut invalidation = world.resource_mut::<tracking::Invalidation>();
            std::mem::swap(&mut self.touched, &mut invalidation.touched);
        }
        let world: &World = world;
        if self.words_stale {
            for (declared, &entity) in self.declared.iter_mut().zip(&self.entities) {
                *declared = Declared::read(world, entity);
            }
        } else {
            for &entity in &self.touched {
                let Some(&index) = self.indices.get(&entity) else {
                    continue;
                };
                let next = Declared::read(world, entity);
                if self.declared[index] != next {
                    self.declared[index] = next;
                    self.worded.push(index);
                }
            }
        }
        self.touched.clear();
    }

    fn word_from_parent(&self, index: usize) -> NodeWord {
        let link = self.links[index];
        let parent = link
            .parent()
            .filter(|_| link.topology_valid())
            .map(|parent| self.words[parent]);
        self.declared[index].word(parent)
    }

    pub(super) fn settle_words(&mut self) -> bool {
        let _profile = engine_profile::span("hierarchy", "settle_words");
        if std::mem::take(&mut self.words_stale) {
            self.worded.clear();
            for position in 0..self.slots.order.len() {
                let index = self.slots.order[position];
                self.words[index] = self.word_from_parent(index);
            }
            for entity in self.asleep.drain(..) {
                if let Some(&index) = self.indices.get(&entity)
                    && self.words[index].active
                {
                    self.woken.push(entity);
                }
            }
            return true;
        }
        let threshold = self.dense_threshold();
        let mut changed = false;
        let mut worded = std::mem::take(&mut self.worded);
        while let Some(index) = worded.pop() {
            if !self.live(index) {
                continue;
            }
            let next = self.word_from_parent(index);
            let before = self.words[index];
            if next == before {
                continue;
            }
            self.words[index] = next;
            self.placed.mark(index as u32);
            changed = true;
            if !before.active && next.active {
                self.dirty.mark(index, threshold);
                self.woken.push(self.entities[index]);
            }
            worded.extend(self.children(index));
        }
        self.worded = worded;
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
