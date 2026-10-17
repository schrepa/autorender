//! Verified source checkouts in the shared disposable store. A command holds
//! an exclusive OS lease for every tree it uses; concurrent processes cannot
//! replace its source or web outputs. Collection happens only after release.

use crate::measure::{ensure_commit, git};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Mutex,
};
use xtask::cache::{Entry, Store};

static USING: Mutex<BTreeMap<String, Entry>> = Mutex::new(BTreeMap::new());

pub fn checkout(root: &Path, sha: &str, pull: Option<u64>) -> Result<PathBuf, String> {
    ensure_commit(root, sha, pull)?;
    let store = Store::open(root)?;
    let repository = git(
        root,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )?;
    let identity = format!("{repository}:{sha}");
    let mut using = USING.lock().map_err(|e| e.to_string())?;
    if let Some(entry) = using.get(&identity) {
        return Ok(entry.path().join("source"));
    }
    let entry = store.entry("source", &identity)?;
    entry.require_capacity(0)?;
    let source = entry.path().join("source");
    if !source.exists() {
        git(root, &["worktree", "prune"])?;
        git(
            root,
            &[
                "worktree",
                "add",
                "--detach",
                &source.display().to_string(),
                sha,
            ],
        )?;
    }
    if git(&source, &["rev-parse", "HEAD"])? != sha {
        return Err(format!(
            "managed checkout {} has a different HEAD",
            source.display()
        ));
    }
    if !git(&source, &["status", "--porcelain"])?.is_empty() {
        return Err(format!(
            "managed checkout {} contains source edits; refusing to reuse or discard them",
            source.display()
        ));
    }
    using.insert(identity, entry);
    Ok(source)
}

/// Active trees stay pinned; inactive trees and compiler contexts share the
/// configured byte budget. Foreign targets and caller results are untouched.
pub fn prune(root: &Path) {
    let result = Store::open(root).and_then(|store| store.collect(0));
    match result {
        Ok(report) if report["within_budget"] == false => eprintln!("cache capacity: {report}"),
        Err(error) => eprintln!("cache collection: {error}"),
        _ => {}
    }
}

pub fn release() {
    if let Ok(mut using) = USING.lock() {
        using.clear();
    }
}
