//! Maps keyed by tile that no one walks in a hash table's order: that
//! order differs between native and Wasm builds and follows each table's
//! history, so what a frame does would too. A [`TileMap`] answers lookups
//! by key and hands out its tiles in key order alone.

use std::collections::hash_map::Entry;

use super::*;

/// A map keyed by tile: lookups by key, and its tiles in key order alone.
pub(super) struct TileMap<V>(HashMap<TileKey, V>);

impl<V> Default for TileMap<V> {
    fn default() -> Self {
        Self(HashMap::default())
    }
}

impl<V> TileMap<V> {
    pub(super) fn get(&self, key: &TileKey) -> Option<&V> {
        self.0.get(key)
    }

    pub(super) fn get_mut(&mut self, key: &TileKey) -> Option<&mut V> {
        self.0.get_mut(key)
    }

    pub(super) fn contains_key(&self, key: &TileKey) -> bool {
        self.0.contains_key(key)
    }

    pub(super) fn insert(&mut self, key: TileKey, value: V) -> Option<V> {
        self.0.insert(key, value)
    }

    pub(super) fn remove(&mut self, key: &TileKey) -> Option<V> {
        self.0.remove(key)
    }

    pub(super) fn entry(&mut self, key: TileKey) -> Entry<'_, TileKey, V> {
        self.0.entry(key)
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn clear(&mut self) {
        self.0.clear();
    }

    /// Its tiles, in key order.
    pub(super) fn keys(&self) -> Vec<TileKey> {
        let mut keys: Vec<TileKey> = self.0.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    /// Its tiles `wanted` holds for, in key order.
    pub(super) fn keys_where(&self, wanted: impl Fn(&TileKey, &V) -> bool) -> Vec<TileKey> {
        let mut keys: Vec<TileKey> = self
            .0
            .iter()
            .filter(|(key, value)| wanted(key, value))
            .map(|(&key, _)| key)
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Its tiles with their values, in key order.
    pub(super) fn iter(&self) -> impl Iterator<Item = (TileKey, &V)> {
        self.keys().into_iter().map(|key| (key, &self.0[&key]))
    }

    /// How many of its values `counts` holds for, which no order changes.
    pub(super) fn count(&self, counts: impl Fn(&V) -> bool) -> usize {
        self.0.values().filter(|value| counts(value)).count()
    }

    /// Keeps the tiles `keep` holds for, asking in key order.
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&TileKey, &mut V) -> bool) {
        for key in self.keys() {
            if let Some(value) = self.0.get_mut(&key)
                && !keep(&key, value)
            {
                self.0.remove(&key);
            }
        }
    }

    /// Takes every tile out, in key order.
    pub(super) fn drain(&mut self) -> Vec<(TileKey, V)> {
        let mut drained: Vec<(TileKey, V)> = self.0.drain().collect();
        drained.sort_unstable_by_key(|&(key, _)| key);
        drained
    }
}
