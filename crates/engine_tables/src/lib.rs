#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod bytes;
pub mod clouds;
pub mod sky;

pub use bytes::{f16_bits, f16_value};

use engine_math::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableKind {
    CloudNoise,
    SkyTables,
}

impl TableKind {
    pub const ALL: [Self; 2] = [Self::CloudNoise, Self::SkyTables];

    pub const fn name(self) -> &'static str {
        match self {
            Self::CloudNoise => "cloud_noise",
            Self::SkyTables => "sky_tables",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

pub fn run(kind: &str, input: &[u8]) -> Option<Result<Vec<u8>, String>> {
    let kind = TableKind::from_name(kind)?;
    Some(make(kind, input))
}

fn make(kind: TableKind, input: &[u8]) -> Result<Vec<u8>, String> {
    match kind {
        TableKind::CloudNoise => clouds::record(input),
        TableKind::SkyTables => sky::record(input),
    }
}

enum Running {
    Noise(Box<clouds::NoisePart>),
    Whole(TableKind, Vec<u8>),
}

#[derive(Default)]
pub struct TableWorker {
    running: HashMap<u64, Running>,
}

impl TableWorker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn runs(kind: &str) -> bool {
        TableKind::from_name(kind).is_some()
    }

    pub fn holds(&self, id: u64) -> bool {
        self.running.contains_key(&id)
    }

    pub fn start(&mut self, id: u64, kind: &str, input: &[u8]) -> Result<(), String> {
        let kind = TableKind::from_name(kind).ok_or_else(|| format!("unknown job kind {kind}"))?;
        let running = match kind {
            TableKind::CloudNoise => Running::Noise(Box::new(clouds::NoisePart::new(input)?)),
            TableKind::SkyTables => Running::Whole(kind, input.to_vec()),
        };
        self.running.insert(id, running);
        Ok(())
    }

    pub fn step(&mut self, id: u64, budget: u64) -> Result<Option<Vec<u8>>, String> {
        let Some(running) = self.running.get_mut(&id) else {
            return Err("no such job".into());
        };
        let output = match running {
            Running::Noise(part) => {
                let mut budget = budget.max(1);
                match part.advance(&mut budget) {
                    Some(record) => Ok(record),
                    None => return Ok(None),
                }
            }
            Running::Whole(kind, input) => make(*kind, input),
        };
        self.running.remove(&id);
        output.map(Some)
    }

    pub fn cancel(&mut self, id: u64) {
        self.running.remove(&id);
    }
}
