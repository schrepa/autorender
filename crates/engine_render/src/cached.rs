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

#[cfg(test)]
mod tests {
    use super::Cached;

    #[test]
    fn stale_until_set_for_the_key() {
        let mut cached = Cached::default();
        assert!(cached.stale(&1));
        assert_eq!(cached.get(), None);
        cached.set(1, "one");
        assert!(!cached.stale(&1));
        assert!(cached.stale(&2));
        assert_eq!(cached.get(), Some(&"one"));
        assert_eq!(cached.key(), Some(&1));
        cached.set(2, "two");
        assert!(cached.stale(&1));
        assert_eq!(cached.get(), Some(&"two"));
    }

    #[test]
    fn builds_only_for_new_keys() {
        let mut cached = Cached::default();
        let mut builds = 0;
        let mut build = |value| {
            builds += 1;
            value
        };
        assert_eq!(*cached.get_or_insert_with(1, || build("one")), "one");
        assert_eq!(*cached.get_or_insert_with(1, || build("again")), "one");
        assert_eq!(*cached.get_or_insert_with(2, || build("two")), "two");
        assert_eq!(builds, 2);
    }

    #[test]
    fn cleared_values_are_stale() {
        let mut cached = Cached::default();
        cached.set((3, 4), 'a');
        cached.clear();
        assert!(cached.stale(&(3, 4)));
        assert_eq!(cached.get(), None);
    }
}
