//! Dense entity lookups for hot paths: each entity's index by its entity
//! index, instead of hashing `Entity` keys. A lookup reads one entry and
//! compares the stored entity, generation included, so entries of despawned
//! entities whose index was reused never match.

use std::ops::Index;

use bevy_ecs::entity::Entity;

#[derive(Clone, Copy, Debug)]
struct Entry {
    entity: Entity,
    value: usize,
}

const EMPTY: Entry = Entry {
    entity: Entity::PLACEHOLDER,
    value: usize::MAX,
};

/// A map from entities to indices with the calls of the `HashMap`s it
/// replaces. Memory follows the largest entity index inserted; clearing
/// costs as many writes as entries were inserted.
#[derive(Clone, Debug, Default)]
pub(crate) struct EntityIndices {
    entries: Vec<Entry>,
    // Entity indices holding entries, for proportional clearing.
    occupied: Vec<u32>,
}

impl EntityIndices {
    #[inline]
    pub(crate) fn get(&self, entity: &Entity) -> Option<&usize> {
        let entry = self.entries.get(entity.index_u32() as usize)?;
        (entry.entity == *entity).then_some(&entry.value)
    }

    /// Sets `entity`'s index, returning the one it replaces.
    pub(crate) fn insert(&mut self, entity: Entity, value: usize) -> Option<usize> {
        let slot = entity.index_u32() as usize;
        if slot >= self.entries.len() {
            self.entries.resize(slot + 1, EMPTY);
        }
        let entry = &mut self.entries[slot];
        let previous = if entry.entity == entity {
            Some(entry.value)
        } else {
            if entry.entity == Entity::PLACEHOLDER {
                self.occupied.push(slot as u32);
            }
            None
        };
        *entry = Entry { entity, value };
        previous
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
        self.get(entity).expect("entity has an index")
    }
}
