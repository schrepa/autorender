use std::ops::Index;

use bevy_ecs::entity::Entity;

#[derive(Clone, Copy, Debug)]
struct Entry {
    entity: Entity,
    value: usize,
    at: u32,
}

const EMPTY: Entry = Entry {
    entity: Entity::PLACEHOLDER,
    value: usize::MAX,
    at: u32::MAX,
};

#[derive(Clone, Debug, Default)]
pub(crate) struct EntityIndices {
    entries: Vec<Entry>,
    occupied: Vec<u32>,
}

impl EntityIndices {
    #[inline]
    pub(crate) fn get(&self, entity: &Entity) -> Option<&usize> {
        let entry = self.entries.get(entity.index_u32() as usize)?;
        (entry.entity == *entity).then_some(&entry.value)
    }

    pub(crate) fn insert(&mut self, entity: Entity, value: usize) -> Option<usize> {
        let slot = entity.index_u32() as usize;
        if slot >= self.entries.len() {
            self.entries.resize(slot + 1, EMPTY);
        }
        let entry = &mut self.entries[slot];
        let previous = (entry.entity == entity).then_some(entry.value);
        if entry.entity == Entity::PLACEHOLDER {
            entry.at = self.occupied.len() as u32;
            self.occupied.push(slot as u32);
        }
        entry.entity = entity;
        entry.value = value;
        previous
    }

    pub(crate) fn remove(&mut self, entity: &Entity) -> Option<usize> {
        let slot = entity.index_u32() as usize;
        let entry = self.entries.get_mut(slot)?;
        if entry.entity != *entity {
            return None;
        }
        let (value, at) = (entry.value, entry.at as usize);
        *entry = EMPTY;
        self.occupied.swap_remove(at);
        if let Some(&moved) = self.occupied.get(at) {
            self.entries[moved as usize].at = at as u32;
        }
        Some(value)
    }

    pub(crate) fn clear(&mut self) {
        for slot in self.occupied.drain(..) {
            self.entries[slot as usize] = EMPTY;
        }
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        self.occupied.reserve(additional);
    }
}

impl Index<&Entity> for EntityIndices {
    type Output = usize;

    fn index(&self, entity: &Entity) -> &usize {
        self.get(entity).expect("entity_indexed")
    }
}
