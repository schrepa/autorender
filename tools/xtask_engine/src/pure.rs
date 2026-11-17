mod simplify;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use serde_json::{Map, Value, json};

use crate::Failure;
use crate::process::Project;
use crate::smoke::PLAYER;

const GOLDENS: &str = "tools/xtask_engine/pure/goldens.json";

pub(crate) const LISTED: &str = "crates/engine/examples/bare_game/generated/corpora.json";

pub(crate) struct Corpus {
    pub name: &'static str,
    pub run: fn(&Path, &mut Observations) -> Result<(), String>,
}

pub(crate) const CORPORA: &[Corpus] = &[Corpus {
    name: "simplify",
    run: simplify::run,
}];

pub(crate) fn listed(root: &Path) -> Result<Vec<Value>, String> {
    let text = fs::read_to_string(root.join(LISTED))
        .map_err(|error| format!("cannot read {LISTED}: {error}"))?;
    let listed: Value =
        serde_json::from_str(&text).map_err(|error| format!("{LISTED} is not JSON: {error}"))?;
    listed
        .as_array()
        .cloned()
        .ok_or_else(|| format!("{LISTED} is not a JSON array"))
}

pub(crate) fn names(root: &Path) -> Vec<String> {
    let games = listed(root).unwrap_or_default();
    CORPORA
        .iter()
        .map(|corpus| corpus.name.to_owned())
        .chain(
            games
                .iter()
                .filter_map(|corpus| corpus["name"].as_str().map(str::to_owned)),
        )
        .collect()
}

#[derive(Default)]
pub(crate) struct Observations {
    values: BTreeMap<String, Value>,
    broken: Vec<String>,
}

impl Observations {
    pub fn record(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.values.insert(name.into(), value.into());
    }

    pub fn check(&mut self, holds: bool, what: impl FnOnce() -> String) {
        if !holds {
            self.broken.push(what());
        }
    }
}

pub(crate) use crate::process::hash;

struct Observed {
    corpus: String,
    values: BTreeMap<String, Value>,
    broken: Vec<String>,
    jobs: Option<String>,
    seconds: f64,
}

impl Observed {
    fn reported(record: &Value) -> Option<Self> {
        Some(Self {
            corpus: record["corpus"].as_str()?.to_owned(),
            values: record["values"].as_object()?.clone().into_iter().collect(),
            broken: record["broken"]
                .as_array()?
                .iter()
                .map(|broken| broken.as_str().map(str::to_owned))
                .collect::<Option<_>>()?,
            jobs: record["jobs"].as_str().map(str::to_owned),
            seconds: record["seconds"].as_f64()?,
        })
    }
}

pub fn pure(root: &Path, options: &[String]) -> Result<(), Failure> {
    let known = names(root);
    let usage = || {
        format!(
            "usage: pure [--bless] [--wasm] [CORPUS...], corpora: {}",
            known.join(", ")
        )
    };
    let (mut bless, mut wasm) = (false, false);
    let (mut chosen, mut asked) = (Vec::new(), Vec::new());
    for option in options {
        match option.as_str() {
            "--bless" => bless = true,
            "--wasm" => wasm = true,
            name => match CORPORA.iter().find(|corpus| corpus.name == name) {
                Some(corpus) => chosen.push(corpus),
                None if known.iter().any(|listed| listed == name) => asked.push(name),
                None => return Err(usage().into()),
            },
        }
    }
    let every = chosen.is_empty() && asked.is_empty();
    if every {
        chosen.extend(CORPORA);
    }
    let mut observed = Vec::new();
    for corpus in chosen {
        let start = Instant::now();
        let mut observations = Observations::default();
        (corpus.run)(root, &mut observations)
            .map_err(|error| format!("pure {}: {error}", corpus.name))?;
        observed.push(Observed {
            corpus: corpus.name.to_owned(),
            values: observations.values,
            broken: observations.broken,
            jobs: None,
            seconds: start.elapsed().as_secs_f64(),
        });
    }
    if every || !asked.is_empty() {
        let names = if every {
            engine::smokes::ALL.to_owned()
        } else {
            asked.join(",")
        };
        observed.extend(reported(root, &names)?);
    }
    if let Some(twice) = observed.iter().enumerate().find(|(at, one)| {
        observed[..*at]
            .iter()
            .any(|other| other.corpus == one.corpus)
    }) {
        return Err(format!("two corpora are named {}", twice.1.corpus).into());
    }
    let path = root.join(GOLDENS);
    let mut goldens: Map<String, Value> = match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text)
            .map_err(|error| format!("{GOLDENS} is not a JSON object: {error}"))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Map::new(),
        Err(error) => return Err(format!("cannot read {GOLDENS}: {error}").into()),
    };
    let mut summary = Vec::new();
    let mut problems = judged(&observed, &mut goldens, bless, &mut summary);
    if wasm {
        for corpus in &observed {
            if let Some(jobs) = &corpus.jobs {
                replayed(root, &corpus.corpus, jobs, &mut summary, &mut problems)?;
            }
        }
    }
    if bless {
        let text = serde_json::to_string_pretty(&goldens).expect("goldens_serialize") + "\n";
        if let Some(folder) = path.parent() {
            fs::create_dir_all(folder)
                .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
        }
        fs::write(&path, text).map_err(|error| format!("cannot write {GOLDENS}: {error}"))?;
    }
    for record in &summary {
        println!("{record}");
    }
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

fn reported(root: &Path, names: &str) -> Result<Vec<Observed>, Failure> {
    let project = Project::read(root, PLAYER)?;
    let program = crate::pipeline::build_native(root, &project)?;
    let record = json!({
        "event": "running",
        "program": program,
        "arguments": ["--pure", names],
    });
    println!("{record}");
    let output = Command::new(&program)
        .args(["--pure", names])
        .current_dir(root)
        .output()
        .map_err(|error| format!("{program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} --pure {names} stopped ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["event"] == "corpus_observed")
        .map(|record| {
            Observed::reported(&record)
                .ok_or_else(|| format!("{program} --pure reported {record}").into())
        })
        .collect()
}

fn judged(
    observed: &[Observed],
    goldens: &mut Map<String, Value>,
    bless: bool,
    summary: &mut Vec<Value>,
) -> Vec<String> {
    let mut problems = Vec::new();
    for corpus in observed {
        let name = corpus.corpus.as_str();
        let mut differences = Vec::new();
        let golden = goldens
            .get(name)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for (key, value) in &corpus.values {
            match golden.get(key) {
                Some(expected) if expected == value => {}
                Some(expected) => differences.push(format!("{key}: {expected} -> {value}")),
                None => differences.push(format!("{key}: new, {value}")),
            }
        }
        for key in golden.keys() {
            if !corpus.values.contains_key(key) {
                differences.push(format!("{key}: no longer observed"));
            }
        }
        let result = match (corpus.broken.is_empty(), differences.is_empty()) {
            (false, _) => "broken",
            (true, true) => "equal",
            (true, false) if bless => "blessed",
            (true, false) => "differs",
        };
        summary.push(json!({
            "event": "corpus_result",
            "corpus": name,
            "result": result,
            "observations": corpus.values.len(),
            "seconds": corpus.seconds,
        }));
        for broken in &corpus.broken {
            problems.push(format!("{name}: {broken}"));
        }
        if !corpus.broken.is_empty() {
            continue;
        }
        if bless {
            goldens.insert(
                name.to_owned(),
                Value::Object(corpus.values.clone().into_iter().collect()),
            );
        } else {
            problems.extend(
                differences
                    .into_iter()
                    .map(|difference| format!("{name}: {difference}")),
            );
        }
    }
    problems
}

fn replayed(
    root: &Path,
    name: &str,
    jobs: &str,
    summary: &mut Vec<Value>,
    problems: &mut Vec<String>,
) -> Result<(), Failure> {
    let start = Instant::now();
    let replayed = replay_in_wasm(root, jobs)?;
    let seconds = start.elapsed().as_secs_f64();
    let mismatched = replayed["mismatched"].as_u64().unwrap_or(0);
    summary.push(json!({
        "event": "wasm_replay_result",
        "corpus": name,
        "result": if mismatched == 0 { "equal" } else { "differs" },
        "jobs": replayed["jobs"],
        "seconds": seconds,
    }));
    if mismatched > 0 {
        problems.push(format!(
            "{name} in Wasm: {mismatched} job(s) give other bits than natively: {}",
            replayed["mismatches"]
        ));
    }
    Ok(())
}

fn replay_in_wasm(root: &Path, jobs: &str) -> Result<Value, Failure> {
    crate::pipeline::build_wasm(root, true)?;
    let jobs = root.join(jobs).display().to_string();
    let record = serde_json::json!({
        "event": "running",
        "program": "node",
        "arguments": ["web/scripts/pure-wasm.mjs", jobs],
    });
    println!("{record}");
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
