mod caches;
mod gltf;
mod import;
mod requests;
mod scenes;
mod simplify;
mod tables;
mod terrain;
mod textures;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::Failure;

const GOLDENS: &str = "tools/xtask/pure/goldens.json";

pub(crate) struct Corpus {
    pub name: &'static str,
    pub run: fn(&Path, &mut Observations) -> Result<(), String>,
}

pub(crate) const CORPORA: &[Corpus] = &[
    Corpus {
        name: "terrain",
        run: terrain::run,
    },
    Corpus {
        name: "scenes",
        run: scenes::run,
    },
    Corpus {
        name: "requests",
        run: requests::run,
    },
    Corpus {
        name: "simplify",
        run: simplify::run,
    },
    Corpus {
        name: "gltf",
        run: gltf::run,
    },
    Corpus {
        name: "import",
        run: import::run,
    },
    Corpus {
        name: "caches",
        run: caches::run,
    },
    Corpus {
        name: "textures",
        run: textures::run,
    },
    Corpus {
        name: "tables",
        run: tables::run,
    },
];

const REPLAYED: &[(&str, &str)] = &[
    ("terrain", terrain::JOBS),
    ("tables", tables::JOBS),
    ("textures", textures::JOBS),
];

#[derive(Default)]
pub(crate) struct Observations {
    values: BTreeMap<String, Value>,
    broken: Vec<String>,
}

impl Observations {
    pub fn record(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.values.insert(name.into(), value.into());
    }

    pub fn hash(&mut self, name: impl Into<String>, bytes: &[u8]) {
        self.record(name, hash(bytes));
    }

    pub fn check(&mut self, holds: bool, what: impl FnOnce() -> String) {
        if !holds {
            self.broken.push(what());
        }
    }
}

pub(crate) fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub fn pure(root: &Path, options: &[String]) -> Result<(), Failure> {
    let usage = || {
        format!(
            "usage: pure [--bless] [--wasm] [CORPUS...], corpora: {}",
            CORPORA
                .iter()
                .map(|corpus| corpus.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let (mut bless, mut wasm) = (false, false);
    let mut chosen = Vec::new();
    for option in options {
        match option.as_str() {
            "--bless" => bless = true,
            "--wasm" => wasm = true,
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
    for corpus in &chosen {
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
    for &(name, jobs) in REPLAYED {
        if !(wasm && chosen.iter().any(|corpus| corpus.name == name)) {
            continue;
        }
        let start = Instant::now();
        let replayed = replay_in_wasm(root, jobs)?;
        let seconds = start.elapsed().as_secs_f64();
        let mismatched = replayed["mismatched"].as_u64().unwrap_or(0);
        summary.push(format!(
            "{:15} {:8} {:5} jobs {seconds:6.1} s",
            format!("{name} in Wasm"),
            if mismatched == 0 { "equal" } else { "DIFFERS" },
            replayed["jobs"]
        ));
        if mismatched > 0 {
            problems.push(format!(
                "{name} in Wasm: {mismatched} job(s) give other bits than natively: {}",
                replayed["mismatches"]
            ));
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

fn replay_in_wasm(root: &Path, jobs: &str) -> Result<Value, Failure> {
    crate::pipeline::build_wasm(root, true, crate::pipeline::QUICK)?;
    let jobs = root.join(jobs).display().to_string();
    println!("Running node web/scripts/pure-wasm.mjs {jobs}");
    let output = std::process::Command::new("node")
        .args(["web/scripts/pure-wasm.mjs", &jobs])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run node: {error}"))?;
    if !(output.status.success() || output.status.code() == Some(3)) {
        return Err(format!(
            "the Wasm replay stopped ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("the Wasm replay printed no report: {error}").into())
}
