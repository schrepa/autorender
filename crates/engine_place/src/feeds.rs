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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Changed<'a> {
    pub all: bool,
    pub listed: &'a [u32],
    pub upto: u64,
}

#[derive(Debug)]
pub struct Changes {
    listed: Vec<u32>,
    latest: Vec<u64>,
    first: u64,
    next: u64,
    all: Option<u64>,
    taken: Vec<AtomicU64>,
}

impl Changes {
    pub fn new(readers: usize) -> Self {
        Self {
            listed: Vec::new(),
            latest: Vec::new(),
            first: 0,
            next: 0,
            all: None,
            taken: (0..readers).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    fn read_past(&self, ordinal: u64) -> bool {
        self.taken
            .iter()
            .any(|taken| taken.load(Ordering::Relaxed) > ordinal)
    }

    pub fn mark(&mut self, index: u32) {
        let at = index as usize;
        if at >= self.latest.len() {
            self.latest.resize(at + 1, 0);
        }
        let latest = self.latest[at];
        if latest != 0 && !self.read_past(latest - 1) {
            return;
        }
        self.listed.push(index);
        self.next += 1;
        self.latest[at] = self.next;
    }

    pub fn mark_all(&mut self) {
        for &index in &self.listed {
            self.latest[index as usize] = 0;
        }
        self.listed.clear();
        self.all = Some(self.next);
        self.next += 1;
        self.first = self.next;
    }

    pub fn read(&self) -> Changed<'_> {
        Changed {
            all: self.all.is_some(),
            listed: &self.listed,
            upto: self.next,
        }
    }

    pub fn take(&self, reader: usize, upto: u64) {
        if let Some(taken) = self.taken.get(reader) {
            taken.fetch_max(upto, Ordering::Relaxed);
        }
    }

    pub fn settle(&mut self) {
        let least = self
            .taken
            .iter()
            .map(|taken| taken.load(Ordering::Relaxed))
            .min()
            .unwrap_or(self.next);
        if self.all.is_some_and(|at| least > at) {
            self.all = None;
        }
        let gone = (least.saturating_sub(self.first) as usize).min(self.listed.len());
        for (offset, &index) in self.listed[..gone].iter().enumerate() {
            let latest = &mut self.latest[index as usize];
            if *latest == self.first + offset as u64 + 1 {
                *latest = 0;
            }
        }
        self.listed.drain(..gone);
        self.first += gone as u64;
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
