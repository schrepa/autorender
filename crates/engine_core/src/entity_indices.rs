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

    #[inline]
    #[cfg(test)]
    pub(crate) fn contains_key(&self, entity: &Entity) -> bool {
        self.get(entity).is_some()
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

    /// Entities holding entries (an entity whose index slot another took
    /// counts until the next clear).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.occupied.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.occupied.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.occupied.capacity()
    }
}

impl Index<&Entity> for EntityIndices {
    type Output = usize;

    fn index(&self, entity: &Entity) -> &usize {
        self.get(entity).expect("entity has an index")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups_match_by_generation_and_clearing_forgets_everything() {
        let entity = |index: u64, generation: u64| Entity::from_bits(index | generation << 32);
        let (a, b) = (entity(5, 0), entity(900, 0));
        let mut indices = EntityIndices::default();
        assert_eq!(indices.insert(a, 3), None);
        assert_eq!(indices.insert(b, 7), None);
        assert_eq!(indices.insert(a, 4), Some(3));
        assert_eq!(
            (indices.get(&a), indices[&b], indices.len()),
            (Some(&4), 7, 2)
        );
        // An entity reusing a despawned one's index, with a new generation,
        // does not find the old entity's entry, nor the old one its.
        let c = entity(5, 1);
        assert!(!indices.contains_key(&c));
        indices.insert(c, 9);
        assert!(!indices.contains_key(&a) && indices[&c] == 9);
        indices.clear();
        assert!(indices.is_empty() && indices.get(&b).is_none() && indices.get(&c).is_none());
    }
}
