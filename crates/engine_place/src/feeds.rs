use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Serialize, Serializer};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StoreId(u64);

impl StoreId {
    pub fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Clone, Debug)]
pub struct Ring<T> {
    entries: Vec<T>,
    first: u64,
    size: usize,
}

impl<T> Ring<T> {
    fn new(size: usize) -> Self {
        Self {
            entries: Vec::new(),
            first: 0,
            size: size.max(1),
        }
    }

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

    fn clear(&mut self) {
        self.first = self.end();
        self.entries.clear();
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Cursor {
    store: StoreId,
    seen: u64,
    missed: u64,
}

impl Cursor {
    pub fn missed(&self) -> u64 {
        self.missed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Since<'a, T> {
    Same,
    Listed(&'a [T]),
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Events<'a, T> {
    Listed(&'a [T]),
    Missed(u64),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FeedView {
    pub store: StoreId,
    pub first: u64,
    pub next: u64,
    pub size: usize,
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
            log: Ring::new(size),
            next: 0,
        }
    }

    pub fn push(&mut self, entry: T) {
        self.log.push(entry);
        self.next += 1;
    }

    pub fn replace(&mut self) {
        self.store = StoreId::next();
        self.log.clear();
    }

    pub fn resize(&mut self, size: usize) {
        self.log.size = size.max(1);
    }

    pub fn store(&self) -> StoreId {
        self.store
    }

    pub fn next(&self) -> u64 {
        self.next
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

    pub fn events<'a>(&'a self, cursor: &mut Cursor) -> Events<'a, T> {
        if cursor.store != self.store {
            cursor.store = self.store;
            cursor.seen = self.log.first;
        }
        if cursor.seen < self.log.first {
            let missed = self.log.first - cursor.seen;
            cursor.missed += missed;
            cursor.seen = self.log.first;
            return Events::Missed(missed);
        }
        let listed = self.log.from(cursor.seen).unwrap_or(&[]);
        cursor.seen = self.next;
        Events::Listed(listed)
    }

    pub fn behind(&self, cursor: &Cursor) -> u64 {
        if cursor.store == self.store {
            self.next.saturating_sub(cursor.seen)
        } else {
            self.next.saturating_sub(self.log.first)
        }
    }
}

impl<T> Serialize for Feed<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        FeedView {
            store: self.store,
            first: self.log.first,
            next: self.next,
            size: self.log.size,
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "schema")]
impl<T> schemars::JsonSchema for Feed<T> {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "FeedView".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        FeedView::json_schema(generator)
    }
}
