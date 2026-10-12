//! `cargo xtask pure`: deterministic checks of engine code over tracked
//! fixtures, without a GPU or a browser. Each corpus runs the code its
//! fixtures exercise, checks the promises that code makes (a tile
//! generated in bands is the tile generated whole, a job worker's output
//! is the engine's own) and observes what it made: hashes and a few
//! readable numbers, which must equal the corpus's goldens in
//! tools/xtask/pure/goldens.json. An intended change shows as a reviewed
//! diff of that file, which `--bless` rewrites from a run whose promises
//! hold.

mod caches;
mod simplify;
mod terrain;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::Failure;

/// The goldens every corpus's observations must equal, by corpus.
const GOLDENS: &str = "tools/xtask/pure/goldens.json";

/// A set of checks over tracked fixtures: its name and what it runs.
pub(crate) struct Corpus {
    pub name: &'static str,
    pub run: fn(&Path, &mut Observations) -> Result<(), String>,
}

/// Every corpus, in the order `cargo xtask pure` runs them.
pub(crate) const CORPORA: &[Corpus] = &[
    Corpus {
        name: "terrain",
        run: terrain::run,
    },
    Corpus {
        name: "simplify",
        run: simplify::run,
    },
    Corpus {
        name: "caches",
        run: caches::run,
    },
];

/// What a corpus observed, by name, and the promises it found broken.
#[derive(Default)]
pub(crate) struct Observations {
    values: BTreeMap<String, Value>,
    broken: Vec<String>,
}

impl Observations {
    /// Observes `value` as `name`.
    pub fn record(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.values.insert(name.into(), value.into());
    }

    /// Observes `bytes` as `name` by their hash.
    pub fn hash(&mut self, name: impl Into<String>, bytes: &[u8]) {
        self.record(name, hash(bytes));
    }

    /// Notes the promise `what` broken unless it `holds`.
    pub fn check(&mut self, holds: bool, what: impl FnOnce() -> String) {
        if !holds {
            self.broken.push(what());
        }
    }
}

/// The first 16 hexadecimal digits of `bytes`' SHA-256: enough to tell any
/// change in a corpus's outputs.
pub(crate) fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `cargo xtask pure [--bless] [CORPUS...]`: runs the corpora named, or
/// every one, and fails (exit 3) when a promise broke or an observation
/// differs from its golden; `--bless` writes the run's observations as the
/// goldens instead, unless a promise broke.
pub fn pure(root: &Path, options: &[String]) -> Result<(), Failure> {
    let usage = || {
        format!(
            "usage: pure [--bless] [CORPUS...], corpora: {}",
            CORPORA
                .iter()
                .map(|corpus| corpus.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut bless = false;
    let mut chosen = Vec::new();
    for option in options {
        match option.as_str() {
            "--bless" => bless = true,
            name => match CORPORA.iter().find(|corpus| corpus.name == name) {
                Some(corpus) => chosen.push(corpus),
                None => return Err(usage().into()),
            },
        }
    }
    if chosen.is_empty() {
        chosen.extend(CORPORA);
    }
    let path = root.join(GOLDENS);
    let mut goldens: Map<String, Value> = match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text)
            .map_err(|error| format!("{GOLDENS} is not a JSON object: {error}"))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Map::new(),
        Err(error) => return Err(format!("cannot read {GOLDENS}: {error}").into()),
    };
    let mut problems = Vec::new();
    let mut summary = Vec::new();
    for corpus in chosen {
        let start = Instant::now();
        let mut observed = Observations::default();
        (corpus.run)(root, &mut observed)
            .map_err(|error| format!("pure {}: {error}", corpus.name))?;
        let seconds = start.elapsed().as_secs_f64();
        let mut differences = Vec::new();
        let golden = goldens
            .get(corpus.name)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for (name, value) in &observed.values {
            match golden.get(name) {
                Some(expected) if expected == value => {}
                Some(expected) => differences.push(format!("{name}: {expected} -> {value}")),
                None => differences.push(format!("{name}: new, {value}")),
            }
        }
        for name in golden.keys() {
            if !observed.values.contains_key(name) {
                differences.push(format!("{name}: no longer observed"));
            }
        }
        let result = match (observed.broken.is_empty(), differences.is_empty()) {
            (false, _) => "BROKEN",
            (true, true) => "equal",
            (true, false) if bless => "blessed",
            (true, false) => "DIFFERS",
        };
        summary.push(format!(
            "{:10} {result:8} {:5} observations {seconds:6.1} s",
            corpus.name,
            observed.values.len()
        ));
        for broken in &observed.broken {
            problems.push(format!("{}: {broken}", corpus.name));
        }
        if !observed.broken.is_empty() {
            continue;
        }
        if bless {
            goldens.insert(
                corpus.name.to_owned(),
                Value::Object(observed.values.into_iter().collect()),
            );
        } else {
            for difference in differences {
                problems.push(format!("{}: {difference}", corpus.name));
            }
        }
    }
    if bless {
        let text = serde_json::to_string_pretty(&goldens).expect("goldens serialize") + "\n";
        if let Some(folder) = path.parent() {
            fs::create_dir_all(folder)
                .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
        }
        fs::write(&path, text).map_err(|error| format!("cannot write {GOLDENS}: {error}"))?;
    }
    println!("{}", summary.join("\n"));
    if problems.is_empty() {
        return Ok(());
    }
    Err(Failure::found(format!(
        "{} problem(s); a change you intend: `cargo xtask pure --bless` and review the \
         diff of {GOLDENS}\n{}",
        problems.len(),
        problems.join("\n")
    )))
}
