use super::*;

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

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn keys(&self) -> Vec<TileKey> {
        let mut keys: Vec<TileKey> = self.0.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

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

    pub(super) fn iter(&self) -> impl Iterator<Item = (TileKey, &V)> {
        self.keys().into_iter().map(|key| (key, &self.0[&key]))
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&TileKey, &mut V) -> bool) {
        for key in self.keys() {
            if let Some(value) = self.0.get_mut(&key)
                && !keep(&key, value)
            {
                self.0.remove(&key);
            }
        }
    }
}
