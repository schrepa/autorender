pub use foldhash::fast::FixedState;

#[expect(clippy::disallowed_types, reason = "the fixed-hasher map")]
pub type HashMap<K, V> = std::collections::HashMap<K, V, FixedState>;

#[expect(clippy::disallowed_types, reason = "the fixed-hasher set")]
pub type HashSet<T> = std::collections::HashSet<T, FixedState>;
