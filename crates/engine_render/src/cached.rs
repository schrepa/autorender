//! Values derived from inputs a key identifies, rebuilt when the inputs
//! change: mostly bind groups over the buffers and textures of given
//! generations, which every pass keeps between frames.

/// A value built for a key, rebuilt when asked for another key.
#[derive(Debug)]
pub(crate) struct Cached<K, V> {
    entry: Option<(K, V)>,
}

impl<K, V> Default for Cached<K, V> {
    fn default() -> Self {
        Self { entry: None }
    }
}

impl<K: PartialEq, V> Cached<K, V> {
    /// Whether the value is missing or was built for another key.
    pub(crate) fn stale(&self, key: &K) -> bool {
        self.entry.as_ref().is_none_or(|(built, _)| built != key)
    }

    /// Keeps `value`, built for `key`.
    pub(crate) fn set(&mut self, key: K, value: V) {
        self.entry = Some((key, value));
    }

    /// The value for `key`, built by `build` unless it already was.
    pub(crate) fn get_or_insert_with(&mut self, key: K, build: impl FnOnce() -> V) -> &V {
        if self.stale(&key) {
            self.entry = Some((key, build()));
        }
        &self.entry.as_ref().expect("built above").1
    }
}

impl<K, V> Cached<K, V> {
    /// The value last built, whatever its key.
    pub(crate) fn get(&self) -> Option<&V> {
        self.entry.as_ref().map(|(_, value)| value)
    }

    /// The key the value was last built for.
    pub(crate) fn key(&self) -> Option<&K> {
        self.entry.as_ref().map(|(key, _)| key)
    }

    /// Forgets the value.
    pub(crate) fn clear(&mut self) {
        self.entry = None;
    }
}
