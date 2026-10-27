//! The static tables the renderer reads, as records: the noise clouds are
//! shaped from ([`clouds`]) and the physically based sky's transmittance
//! and multiple scattering for an air ([`sky`]); and the spread of the
//! wind's waves the sea's waves head by ([`waves`]).
//! CPU code makes each the same bits on every host (IEEE arithmetic without
//! fused operations, `libm`'s transcendental functions), wherever it runs:
//! on a host's workers, in place, or in the cook, which ships them. A
//! table's job is its record's identity and its input, and [`SOURCE`]
//! keys the record with its kind ([`TableKind`]); a record is a tag naming
//! what it holds, its sizes and its texels, which its table's module
//! decodes.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

// `SOURCE`, which keys the records table jobs make: hosts keep them, and
// the cook ships them, while this code stays the same.
include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod bytes;
pub mod clouds;
pub mod sky;
pub mod waves;

pub use bytes::{f16_bits, f16_value};

use engine_math::collections::HashMap;

/// The kinds of table jobs, which hosts carry by name: a slab of the
/// cloud noise, and the sky's tables for an air. Each one's input is its
/// identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableKind {
    CloudNoise,
    SkyTables,
}

impl TableKind {
    /// Every kind.
    pub const ALL: [Self; 2] = [Self::CloudNoise, Self::SkyTables];

    /// Its name, as hosts carry it and key the records they keep by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::CloudNoise => "cloud_noise",
            Self::SkyTables => "sky_tables",
        }
    }

    /// The kind named `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

/// The record of table job `kind` on `input`, made whole; `None` where
/// `kind` names no table job.
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

/// A table job a worker started: a slab of the cloud noise, made a texel a
/// work unit, or the sky's tables, made whole at their first step.
enum Running {
    Noise(Box<clouds::NoisePart>),
    Whole(TableKind, Vec<u8>),
}

/// The table jobs a worker runs, a budget of work units at a time.
#[derive(Default)]
pub struct TableWorker {
    running: HashMap<u64, Running>,
}

impl TableWorker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `kind` names a table job.
    pub fn runs(kind: &str) -> bool {
        TableKind::from_name(kind).is_some()
    }

    /// Whether it started job `id`, which has yet to finish.
    pub fn holds(&self, id: u64) -> bool {
        self.running.contains_key(&id)
    }

    /// Starts job `id` of `kind` on `input`, to [`Self::step`] until done;
    /// another of the same id stops.
    pub fn start(&mut self, id: u64, kind: &str, input: &[u8]) -> Result<(), String> {
        let kind = TableKind::from_name(kind).ok_or_else(|| format!("unknown job kind {kind}"))?;
        let running = match kind {
            TableKind::CloudNoise => Running::Noise(Box::new(clouds::NoisePart::new(input)?)),
            TableKind::SkyTables => Running::Whole(kind, input.to_vec()),
        };
        self.running.insert(id, running);
        Ok(())
    }

    /// Works on job `id` for about `budget` work units: its record once
    /// done, `None` while work is left. A job that failed or finished is
    /// gone.
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

    /// Stops job `id`, if it runs.
    pub fn cancel(&mut self, id: u64) {
        self.running.remove(&id);
    }
}
