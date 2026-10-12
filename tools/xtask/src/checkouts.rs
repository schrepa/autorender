//! The commits measure and check tasks build, run and serve: each checked
//! out under `target/measure/source/<sha>` (a check's development server
//! builds the page there, in the checkout's own target folder) and built
//! under `target/measure/builds/<sha>`. Besides the commits a run uses, the
//! checkouts and builds of the [`KEPT`] used most recently stay; older ones
//! go once a task has its commits ready ([`prune`]).

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::measure::{ensure_commit, git, iso};

/// Commits whose checkouts and builds stay besides those a run uses: the
/// most recently used. A checkout a check served holds a build of the page
/// (a few GB), a measured commit's build its program and symbols.
const KEPT: usize = 3;

/// The extension of a checkout or build being removed: renamed first, so
/// its folder is whole or gone, then deleted (by a later prune, where
/// something held a file open).
const REMOVING: &str = "removing";

/// The commits this process checked out, which it may still build, run or
/// serve: never removed while it runs.
static USING: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// `target/measure/NAME`.
fn folder(root: &Path, name: &str) -> PathBuf {
    root.join("target").join("measure").join(name)
}

/// The source at `sha`, checked out under `target/measure/source/` (anew
/// where a removal left part of it), recorded as used now and by this run.
pub fn checkout(root: &Path, sha: &str, pull: Option<u64>) -> Result<PathBuf, String> {
    ensure_commit(root, sha, pull)?;
    let source = folder(root, "source").join(sha);
    if !source.join("Cargo.toml").is_file() {
        if source.exists() {
            fs::remove_dir_all(&source)
                .map_err(|error| format!("cannot remove {}: {error}", source.display()))?;
            git(root, &["worktree", "prune"])?;
        }
        let path = source.display().to_string();
        git(
            root,
            &["worktree", "add", "--detach", "--force", &path, sha],
        )?;
    }
    USING
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(sha.to_owned());
    let used = folder(root, "used");
    fs::create_dir_all(&used).map_err(|error| error.to_string())?;
    fs::write(used.join(sha), iso(SystemTime::now()))
        .map_err(|error| format!("cannot write {}: {error}", used.display()))?;
    Ok(source)
}

/// Removes what earlier removals left, then the checkouts and builds of the
/// commits past the [`KEPT`] used most recently, but those this run uses.
/// A folder something holds open stays, said on stderr: a task never fails
/// for what it could not remove.
pub fn prune(root: &Path) {
    let using = USING.lock().unwrap_or_else(PoisonError::into_inner).clone();
    let mut removed = false;
    let mut commits = BTreeSet::new();
    for name in ["source", "builds"] {
        for path in folders(&folder(root, name)) {
            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
            if path
                .extension()
                .is_some_and(|extension| extension == REMOVING)
            {
                removed |= fs::remove_dir_all(&path).is_ok();
            } else if is_commit(&file_name) && !using.contains(file_name.as_ref()) {
                commits.insert(file_name.into_owned());
            }
        }
    }
    let used = |sha: &str| {
        fs::metadata(folder(root, "used").join(sha))
            .and_then(|metadata| metadata.modified())
            .unwrap_or(UNIX_EPOCH)
    };
    let mut others: Vec<(SystemTime, String)> =
        commits.into_iter().map(|sha| (used(&sha), sha)).collect();
    others.sort_by_key(|(used, _)| Reverse(*used));
    for (_, sha) in others.into_iter().skip(KEPT) {
        for name in ["source", "builds"] {
            removed |= remove(&folder(root, name).join(&sha));
        }
        let _ = fs::remove_file(folder(root, "used").join(&sha));
    }
    // Checkouts are git worktrees: forget those whose folders went.
    if removed && let Err(error) = git(root, &["worktree", "prune"]) {
        eprintln!("{error}");
    }
}

/// Whether `name` is a full commit id, as checkouts and builds are named.
fn is_commit(name: &str) -> bool {
    name.len() == 40 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// The folders in `folder`.
fn folders(folder: &Path) -> Vec<PathBuf> {
    fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect()
}

/// Removes the folder `path`, if any: renamed aside, then deleted; whether
/// it left where it was.
fn remove(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }
    let doomed = path.with_extension(REMOVING);
    if let Err(error) = fs::rename(path, &doomed) {
        eprintln!("Keeping {}, which is in use: {error}", path.display());
        return false;
    }
    if let Err(error) = fs::remove_dir_all(&doomed) {
        eprintln!(
            "Could not remove all of {} yet ({error}); a later task will",
            doomed.display()
        );
    }
    true
}
