//! Hash maps and sets that iterate in the same order on every run:
//! foldhash's fast hash with its fixed seed, where the standard library's
//! maps seed SipHash at random each run. Clippy refuses the standard ones
//! (clippy.toml). Keys a stranger chooses, such as a scene document's
//! names, belong in a `BTreeMap` instead: a known seed lets them collide on
//! purpose.

/// The fixed-seed hasher every map and set here hashes with:
/// `HashMap::with_capacity_and_hasher(capacity, FixedState::default())`.
pub use foldhash::fast::FixedState;

/// A hash map with a fixed hasher; `HashMap::default()` makes an empty one.
#[expect(clippy::disallowed_types, reason = "the fixed-hasher map")]
pub type HashMap<K, V> = std::collections::HashMap<K, V, FixedState>;

/// A hash set with a fixed hasher; `HashSet::default()` makes an empty one.
#[expect(clippy::disallowed_types, reason = "the fixed-hasher set")]
pub type HashSet<T> = std::collections::HashSet<T, FixedState>;
