use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, TryLockError};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::cook::{Fault, fault};

#[derive(Serialize, Deserialize, Default)]
pub(crate) struct Record {
    pub(crate) sources: Sources,
    pub(crate) steps: BTreeMap<String, Kept>,
}

pub(crate) type Sources = BTreeMap<String, Seen>;

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Seen {
    size: u64,
    modified: u128,
    pub(crate) sha256: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Kept {
    pub(crate) kind: String,
    pub(crate) version: u32,
    pub(crate) params: String,
    pub(crate) inputs: Vec<(String, String)>,
    pub(crate) key: String,
    pub(crate) made: Made,
}

#[derive(Serialize, Deserialize, Clone, Default, PartialEq)]
pub(crate) struct Made {
    pub(crate) files: BTreeMap<String, Fact>,
    pub(crate) items: BTreeMap<String, BTreeMap<String, Value>>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub(crate) struct Fact {
    pub(crate) sha256: String,
    pub(crate) bytes: u64,
    pub(crate) published: bool,
}

#[derive(Serialize, Default)]
pub(crate) struct Changes {
    added: Vec<String>,
    changed: Vec<String>,
    removed: Vec<String>,
    same: Vec<String>,
}

pub(crate) struct Store {
    out: PathBuf,
    cook: PathBuf,
}

pub(crate) fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(crate) fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<_, _> = map
                .iter()
                .map(|(name, field)| (name.clone(), canonical(field)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

pub(crate) fn local(root: &Path, uri: &str) -> PathBuf {
    uri.split('/')
        .fold(root.to_owned(), |path, part| path.join(part))
}

pub(crate) fn made(dir: &Path) -> Option<Made> {
    serde_json::from_slice(&fs::read(dir.join("made")).ok()?).ok()
}

pub(crate) fn removed(result: io::Result<()>) -> io::Result<()> {
    match result {
        Err(error) if error.kind() != ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

pub(crate) fn seen(path: &Path, before: Option<&Seen>) -> io::Result<Seen> {
    let meta = fs::metadata(path)?;
    let (size, modified) = (meta.len(), meta.modified()?.duration_since(UNIX_EPOCH));
    let modified = modified.map_or(0, |since| since.as_nanos());
    match before {
        Some(before) if before.size == size && before.modified == modified => Ok(before.clone()),
        _ => Ok(Seen {
            size,
            modified,
            sha256: sha(&fs::read(path)?),
        }),
    }
}

fn load(path: &Path) -> io::Result<Record> {
    match fs::read(path) {
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(Record::default()),
        read => Ok(serde_json::from_slice(&read?)?),
    }
}

fn walk(dir: &Path, prefix: &str, found: &mut Vec<String>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let uri = format!("{prefix}{name}");
        if !entry.file_type()?.is_dir() {
            found.push(uri);
        } else if uri != ".cook" {
            walk(&entry.path(), &format!("{uri}/"), found)?;
        }
    }
    Ok(())
}

impl Record {
    fn published(&self) -> BTreeMap<&str, (&str, &str, &Fact)> {
        let mut published = BTreeMap::new();
        for (step, kept) in &self.steps {
            for (uri, fact) in kept.made.files.iter().filter(|(_, fact)| fact.published) {
                published.insert(uri.as_str(), (step.as_str(), kept.key.as_str(), fact));
            }
        }
        published
    }

    fn cooked(&self) -> String {
        let mut sections: BTreeMap<&str, BTreeMap<&str, &Value>> = BTreeMap::new();
        for (section, ids) in self.steps.values().flat_map(|kept| &kept.made.items) {
            let ids = ids.iter().map(|(id, item)| (id.as_str(), item));
            sections.entry(section).or_default().extend(ids);
        }
        let mut files = Vec::new();
        for (uri, (step, _, fact)) in self.published() {
            files.push(
                json!({"uri": uri, "sha256": fact.sha256, "bytes": fact.bytes, "step": step}),
            );
        }
        let mut items = Map::new();
        for (section, ids) in sections {
            items.insert(section.to_owned(), ids.into_values().cloned().collect());
        }
        let cooked = canonical(&json!({"files": files, "items": items}));
        format!("{cooked:#}\n")
    }
}

impl Store {
    pub(crate) fn open(out: &Path) -> Result<(Self, File), Fault> {
        let fail = |error: io::Error| fault("", "Output", &out.display().to_string(), error);
        let (out, cook) = (out.to_owned(), out.join(".cook"));
        let store = Self { out, cook };
        fs::create_dir_all(store.cook.join("steps")).map_err(fail)?;
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(store.cook.join("lock"));
        let lock = lock.map_err(fail)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(fault("", "Locked", ".cook/lock", "")),
            Err(TryLockError::Error(error)) => return Err(fail(error)),
        }
        removed(fs::remove_dir_all(store.cook.join("verify"))).map_err(fail)?;
        if store.cook.join("publishing").exists() {
            let pending = load(&store.cook.join("publishing")).map_err(fail)?;
            let last = load(&store.cook.join("record")).map_err(fail)?;
            store.settle(&last, &pending).map_err(fail)?;
        }
        Ok((store, lock))
    }

    pub(crate) fn staged(&self, step: &str, key: &str, verify: bool) -> PathBuf {
        let root = self.cook.join(if verify { "verify" } else { "steps" });
        root.join(sha(format!("{step}\n{key}").as_bytes()))
    }

    pub(crate) fn record(&self) -> Result<Record, Fault> {
        load(&self.cook.join("record")).map_err(|error| fault("", "Record", ".cook/record", error))
    }

    pub(crate) fn unowned(&self, record: &Record) -> Vec<Fault> {
        let mut found = Vec::new();
        if let Err(error) = walk(&self.out, "", &mut found) {
            return vec![fault("", "Output", "", error)];
        }
        let owned = record.published();
        found.retain(|uri| uri != "cooked.json" && !owned.contains_key(uri.as_str()));
        found
            .iter()
            .map(|uri| fault("", "Unowned", uri, ""))
            .collect()
    }

    pub(crate) fn publish(&self, last: &Record, next: &Record) -> io::Result<Changes> {
        self.replace(&self.cook.join("publishing"), &serde_json::to_vec(next)?)?;
        self.settle(last, next)
    }

    fn settle(&self, last: &Record, next: &Record) -> io::Result<Changes> {
        let (before, after) = (last.published(), next.published());
        let mut changes = Changes::default();
        for (&uri, &(step, key, fact)) in &after {
            let target = local(&self.out, uri);
            let same = before.get(uri).map(|(_, _, old)| old.sha256 == fact.sha256);
            let present = fs::metadata(&target).is_ok_and(|meta| meta.len() == fact.bytes);
            if same != Some(true) || !present {
                let (from, part) = (
                    self.staged(step, key, false).join("files"),
                    self.cook.join("part"),
                );
                fs::copy(local(&from, uri), &part)?;
                target.parent().map_or(Ok(()), fs::create_dir_all)?;
                fs::rename(&part, &target)?;
            }
            let list = match same {
                None => &mut changes.added,
                Some(false) => &mut changes.changed,
                Some(true) => &mut changes.same,
            };
            list.push(uri.to_owned());
        }
        for &uri in before.keys().filter(|uri| !after.contains_key(*uri)) {
            removed(fs::remove_file(local(&self.out, uri)))?;
            changes.removed.push(uri.to_owned());
        }
        let (cooked, manifest) = (next.cooked(), self.out.join("cooked.json"));
        if fs::read(&manifest).ok().as_deref() != Some(cooked.as_bytes()) {
            self.replace(&manifest, cooked.as_bytes())?;
        }
        fs::rename(self.cook.join("publishing"), self.cook.join("record"))?;
        let owned: BTreeSet<_> = next
            .steps
            .iter()
            .map(|(step, kept)| self.staged(step, &kept.key, false))
            .collect();
        for dir in fs::read_dir(self.cook.join("steps"))? {
            let dir = dir?.path();
            if !owned.contains(&dir) {
                fs::remove_dir_all(&dir)?;
            }
        }
        Ok(changes)
    }

    fn replace(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let part = self.cook.join("part");
        fs::write(&part, bytes)?;
        fs::rename(part, path)
    }
}
