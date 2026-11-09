use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine::smokes::{ALL, LIST};
use engine::{BrickList, Error, ErrorCode};
use engine_core::{Corpus, JobKind, JobRef, PureSession, RecordKey};
use engine_registry::{JobList, Registry};
use engine_worker::Worker;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(crate) fn run(names: &str, bricks: impl FnOnce(BrickList) -> BrickList) -> engine::Result<()> {
    let list = bricks(BrickList::default());
    let corpora = registered(&list)?;
    if names == LIST {
        let listed: Vec<Value> = corpora
            .iter()
            .map(|(owner, corpus)| {
                let replays: BTreeSet<&str> = corpus
                    .replays
                    .iter()
                    .map(|kind| kind.source().package())
                    .collect();
                json!({"name": corpus.name, "owner": owner, "replays": replays})
            })
            .collect();
        println!("{}", json!({"event": "corpora", "corpora": listed}));
        return Ok(());
    }
    let chosen = chosen(names, &corpora)?;
    let jobs = Arc::new(list.job_list()?);
    let root = std::env::current_dir()
        .map_err(|error| crate::host_error(ErrorCode::SourceIo, "--pure", error))?;
    for (owner, corpus) in chosen {
        let start = Instant::now();
        let mut session = Session::new(&root, &jobs);
        (corpus.run)(&mut session).map_err(|error| {
            Error::new(ErrorCode::GameRule, format!("$.corpora.{}", corpus.name)).got(error)
        })?;
        let written = if corpus.replays.is_empty() {
            None
        } else {
            Some(session.write(corpus.name)?)
        };
        let record = json!({
            "event": "corpus_observed",
            "corpus": corpus.name,
            "owner": owner,
            "values": session.values,
            "broken": session.broken,
            "jobs": written,
            "seconds": start.elapsed().as_secs_f64(),
        });
        println!("{record}");
    }
    Ok(())
}

fn registered(list: &BrickList) -> engine::Result<Vec<(&'static str, &'static Corpus)>> {
    let engine = engine::engine(|_| list.clone())?;
    let corpora: Vec<(&'static str, &'static Corpus)> = engine
        .app
        .world()
        .get_resource::<Registry>()
        .map(|registry| registry.corpora::<Corpus>().collect())
        .unwrap_or_default();
    for (at, (owner, corpus)) in corpora.iter().enumerate() {
        if let Some((first, _)) = corpora[..at]
            .iter()
            .find(|(_, earlier)| earlier.name == corpus.name)
        {
            return Err(
                Error::new(ErrorCode::NameTwice, format!("$.corpora.{}", corpus.name))
                    .got(*first)
                    .nearest(*owner),
            );
        }
    }
    Ok(corpora)
}

fn chosen<'c>(
    names: &str,
    corpora: &'c [(&'static str, &'static Corpus)],
) -> engine::Result<Vec<&'c (&'static str, &'static Corpus)>> {
    if names == ALL {
        return Ok(corpora.iter().collect());
    }
    let asked: Vec<&str> = names.split(',').collect();
    if let Some(unknown) = asked
        .iter()
        .find(|name| !corpora.iter().any(|(_, corpus)| corpus.name == **name))
    {
        let known = corpora.iter().map(|(_, corpus)| corpus.name);
        return Err(Error::new(ErrorCode::InvalidValue, "--pure")
            .got(*unknown)
            .variants(known.chain([ALL, LIST])));
    }
    Ok(corpora
        .iter()
        .filter(|(_, corpus)| asked.contains(&corpus.name))
        .collect())
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

struct Session {
    root: PathBuf,
    list: Arc<JobList>,
    worker: Worker,
    next: u64,
    log: Vec<Value>,
    values: BTreeMap<String, Value>,
    broken: Vec<String>,
}

impl Session {
    fn new(root: &Path, list: &Arc<JobList>) -> Self {
        let named: String = list
            .hash()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Self {
            root: root.to_path_buf(),
            list: Arc::clone(list),
            worker: Worker::new(Arc::clone(list), engine_worker::now_ms),
            next: 0,
            log: vec![json!({ "list": named })],
            values: BTreeMap::new(),
            broken: Vec::new(),
        }
    }

    fn write(&self, corpus: &str) -> engine::Result<String> {
        let jobs = format!("target/pure/{corpus}-jobs.json");
        let path = self.root.join(&jobs);
        if let Some(folder) = path.parent() {
            fs::create_dir_all(folder)
                .map_err(|error| crate::file_error("--pure", folder, &error))?;
        }
        fs::write(&path, Value::Array(self.log.clone()).to_string())
            .map_err(|error| crate::file_error("--pure", &path, &error))?;
        Ok(jobs)
    }
}

impl PureSession for Session {
    fn root(&self) -> &Path {
        &self.root
    }

    fn list(&self) -> &JobList {
        &self.list
    }

    fn record(&mut self, name: &str, value: Value) {
        self.values.insert(name.to_owned(), value);
    }

    fn hash(&mut self, name: &str, bytes: &[u8]) {
        self.record(name, digest(bytes).into());
    }

    fn digest(&self, bytes: &[u8]) -> String {
        digest(bytes)
    }

    fn check(&mut self, holds: bool, what: &dyn Fn() -> String) {
        if !holds {
            self.broken.push(what());
        }
    }

    fn share(&mut self, bytes: &[u8]) -> RecordKey {
        let key = RecordKey::shared(bytes);
        self.log
            .push(json!({"share": key.hex(), "bytes": STANDARD.encode(bytes)}));
        self.worker.share(key, bytes.to_vec());
        key
    }

    fn run(&mut self, kind: &'static JobKind, input: &[u8]) -> Result<Vec<u8>, String> {
        let output = self
            .worker
            .run(&JobRef::from(kind), input)
            .map_err(|error| error.to_string())?;
        self.log.push(json!({
            "job": kind.name(),
            "source": RecordKey::from_hash(kind.source().hash()).hex(),
            "input": STANDARD.encode(input),
            "output": digest(&output),
        }));
        Ok(output)
    }

    fn sliced(
        &mut self,
        kind: &'static JobKind,
        input: &[u8],
        slice: u64,
    ) -> Result<Vec<u8>, String> {
        self.next += 1;
        self.worker
            .start(self.next, &JobRef::from(kind), input)
            .map_err(|error| error.to_string())?;
        loop {
            let stepped = self
                .worker
                .step(self.next, slice)
                .map_err(|error| error.to_string())?;
            if let Some(output) = stepped {
                self.log.push(json!({
                    "job": kind.name(),
                    "source": RecordKey::from_hash(kind.source().hash()).hex(),
                    "input": STANDARD.encode(input),
                    "slice": slice,
                    "output": digest(&output),
                }));
                return Ok(output);
            }
        }
    }
}
