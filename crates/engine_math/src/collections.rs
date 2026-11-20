use core::sync::atomic::{AtomicU64, Ordering};
use std::vec::Vec;

pub use foldhash::fast::FixedState;

#[expect(clippy::disallowed_types, reason = "the fixed-hasher map")]
pub type HashMap<K, V> = std::collections::HashMap<K, V, FixedState>;

#[expect(clippy::disallowed_types, reason = "the fixed-hasher set")]
pub type HashSet<T> = std::collections::HashSet<T, FixedState>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StoreId(u64);

impl StoreId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Clone, Debug)]
struct Ring<T> {
    entries: Vec<T>,
    first: u64,
    size: usize,
}

impl<T> Ring<T> {
    fn push(&mut self, entry: T) {
        if self.entries.len() >= self.size * 2 {
            let dropped = self.entries.len() - self.size;
            self.entries.drain(..dropped);
            self.first += dropped as u64;
        }
        self.entries.push(entry);
    }

    fn end(&self) -> u64 {
        self.first + self.entries.len() as u64
    }

    fn from(&self, seen: u64) -> Option<&[T]> {
        (self.first..=self.end())
            .contains(&seen)
            .then(|| &self.entries[(seen - self.first) as usize..])
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    store: StoreId,
    seen: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Since<'a, T> {
    Same,
    Listed(&'a [T]),
    All,
}

#[derive(Clone, Debug)]
pub struct Feed<T> {
    store: StoreId,
    log: Ring<T>,
    next: u64,
}

impl<T> Feed<T> {
    pub fn new(size: usize) -> Self {
        Self {
            store: StoreId::next(),
            log: Ring {
                entries: Vec::new(),
                first: 0,
                size: size.max(1),
            },
            next: 0,
        }
    }

    pub fn push(&mut self, entry: T) {
        self.log.push(entry);
        self.next += 1;
    }

    pub fn resize(&mut self, size: usize) {
        self.log.size = size.max(1);
    }

    pub fn since<'a>(&'a self, cursor: &mut Cursor) -> Since<'a, T> {
        let (store, seen) = (cursor.store, cursor.seen);
        cursor.store = self.store;
        cursor.seen = self.next;
        if store != self.store {
            return Since::All;
        }
        if seen == self.next {
            return Since::Same;
        }
        self.log.from(seen).map_or(Since::All, Since::Listed)
    }
}
