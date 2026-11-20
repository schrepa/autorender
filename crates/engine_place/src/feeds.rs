use std::sync::atomic::{AtomicU64, Ordering};

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
