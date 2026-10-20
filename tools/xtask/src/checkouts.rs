//! The commits measure and check tasks build, run and serve. A measured
//! commit builds in one reused checkout, `target/measure/tree`, where git
//! rewrites only the files that differ from the commit built before, so
//! cargo recompiles only what changed ([`reused`]); its build goes to
//! `target/measure/builds/<sha>` beside the files its runs read. A check
//! serves each commit from its own checkout, `target/measure/source/<sha>`,
//! whose page builds in its own target folder ([`checkout`]). Each process
//! leases the commits it uses ([`lease`]), which no task removes while that
//! process runs; besides those, the [`KEPT_BUILDS`] builds and
//! [`KEPT_CHECKOUTS`] checkouts used most recently stay, older ones going
//! once a task has its commits ready ([`prune`]).

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::measure::{ensure_commit, git, iso};

/// Builds kept besides those running tasks use: the most recently used. A
/// build holds a commit's program, its symbols and the files its runs read
/// (about 100 MB).
const KEPT_BUILDS: usize = 12;

/// Checkouts kept besides those running tasks use: the most recently used.
/// A checkout a check served holds a build of the page (a few GB).
const KEPT_CHECKOUTS: usize = 3;

/// The folders of a commit's tree its runs read (bench files, fixtures),
/// kept with its build.
pub const RUN_FILES: &[&str] = &["bench", "web/scripts/fixtures"];

/// The extension of a checkout or build being removed: renamed first, so
/// its folder is whole or gone, then deleted (by a later prune, where
/// something held a file open).
const REMOVING: &str = "removing";

/// How long a process waiting for another's lock waits before looking
/// again.
const LOCK_POLL: Duration = Duration::from_secs(1);

/// The commits this process leased.
static USING: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// `target/measure/NAME`.
fn folder(root: &Path, name: &str) -> PathBuf {
    root.join("target").join("measure").join(name)
}

/// Leases commit `sha` to this process, recorded as used now: its lease,
/// `target/measure/leases/<pid>`, lists the commits whose checkouts and
/// builds no task removes while it runs.
pub fn lease(root: &Path, sha: &str) -> Result<(), String> {
    let _registry = Lock::hold(root, "registry")?;
    let mut using = USING.lock().unwrap_or_else(PoisonError::into_inner);
    if using.insert(sha.to_owned()) {
        let leases = folder(root, "leases");
        fs::create_dir_all(&leases).map_err(|error| error.to_string())?;
        let listed = serde_json::to_string(&*using).expect("commits serialize");
        fs::write(leases.join(std::process::id().to_string()), listed)
            .map_err(|error| format!("cannot write {}: {error}", leases.display()))?;
    }
    let used = folder(root, "used");
    fs::create_dir_all(&used).map_err(|error| error.to_string())?;
    fs::write(used.join(sha), iso(SystemTime::now()))
        .map_err(|error| format!("cannot write {}: {error}", used.display()))
}

/// The source at `sha` in its own checkout under `target/measure/source/`
/// (anew where a removal left part of it), leased to this process.
pub fn checkout(root: &Path, sha: &str, pull: Option<u64>) -> Result<PathBuf, String> {
    ensure_commit(root, sha, pull)?;
    lease(root, sha)?;
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
    Ok(source)
}

/// The reused checkout measured commits build in, at a commit, which other
/// processes wait for until this drops.
pub struct Tree {
    path: PathBuf,
    _held: Lock,
}

impl Tree {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The reused checkout `target/measure/tree` at `sha`, leased to this
/// process and held until the result drops.
pub fn reused(root: &Path, sha: &str, pull: Option<u64>) -> Result<Tree, String> {
    ensure_commit(root, sha, pull)?;
    lease(root, sha)?;
    let held = Lock::hold(root, "tree")?;
    let path = folder(root, "tree");
    check_out(root, &path, sha)?;
    Ok(Tree { path, _held: held })
}

/// Checks `sha` out in `tree`, a checkout of `root` reused from commit to
/// commit (made the first time): git rewrites only the files that differ,
/// so builds there recompile only what changed.
pub fn check_out(root: &Path, tree: &Path, sha: &str) -> Result<(), String> {
    if !tree.join(".git").exists() {
        let path = tree.display().to_string();
        git(
            root,
            &["worktree", "add", "--detach", "--force", &path, sha],
        )?;
    } else if git(tree, &["rev-parse", "HEAD"])? != sha {
        git(tree, &["checkout", "--detach", "--force", sha])?;
    }
    Ok(())
}

/// Copies the folder `from` into `into`, every file under it.
pub fn copy_folder(from: &Path, into: &Path) -> Result<(), String> {
    fs::create_dir_all(into).map_err(|error| error.to_string())?;
    for entry in
        fs::read_dir(from).map_err(|error| format!("cannot read {}: {error}", from.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        let target = into.join(path.file_name().unwrap_or_default());
        if path.is_dir() {
            copy_folder(&path, &target)?;
        } else {
            fs::copy(&path, &target)
                .map_err(|error| format!("cannot copy {}: {error}", path.display()))?;
        }
    }
    Ok(())
}

/// Removes what earlier removals left, then the checkouts and builds of
/// the commits past the [`KEPT_CHECKOUTS`] and [`KEPT_BUILDS`] used most
/// recently, but those running processes lease. A folder something holds
/// open stays, said on stderr: a task never fails for what it could not
/// remove.
pub fn prune(root: &Path) {
    // Chosen and set aside while no process leases, then deleted.
    let doomed = match Lock::hold(root, "registry") {
        Ok(_registry) => doom(root),
        Err(error) => {
            eprintln!("{error}");
            return;
        }
    };
    let mut removed = false;
    for path in doomed {
        match fs::remove_dir_all(&path) {
            Ok(()) => removed = true,
            Err(error) => eprintln!(
                "Could not remove all of {} yet ({error}); a later task will",
                path.display()
            ),
        }
    }
    // Checkouts are git worktrees: forget those whose folders went.
    if removed && let Err(error) = git(root, &["worktree", "prune"]) {
        eprintln!("{error}");
    }
}

/// The folders to delete: those earlier removals left, and the checkouts
/// and builds past those kept, set aside.
fn doom(root: &Path) -> Vec<PathBuf> {
    let leased = leased(root);
    let used = |sha: &str| {
        fs::metadata(folder(root, "used").join(sha))
            .and_then(|metadata| metadata.modified())
            .unwrap_or(UNIX_EPOCH)
    };
    let mut doomed = Vec::new();
    for (name, kept) in [("source", KEPT_CHECKOUTS), ("builds", KEPT_BUILDS)] {
        let mut commits = Vec::new();
        for path in entries(&folder(root, name)).filter(|path| path.is_dir()) {
            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
            if path
                .extension()
                .is_some_and(|extension| extension == REMOVING)
            {
                doomed.push(path.clone());
            } else if is_commit(&file_name) && !leased.contains(file_name.as_ref()) {
                commits.push((used(&file_name), file_name.into_owned()));
            }
        }
        commits.sort_by_key(|(used, _)| Reverse(*used));
        for (_, sha) in commits.into_iter().skip(kept) {
            doomed.extend(set_aside(&folder(root, name).join(&sha)));
        }
    }
    // A commit with neither a checkout nor a build left is no longer used.
    for path in entries(&folder(root, "used")) {
        let sha = path.file_name().unwrap_or_default();
        if !folder(root, "source").join(sha).exists() && !folder(root, "builds").join(sha).exists()
        {
            let _ = fs::remove_file(&path);
        }
    }
    doomed
}

/// The commits running processes lease, this one's included; the leases of
/// processes that ended go.
fn leased(root: &Path) -> BTreeSet<String> {
    let mut leased = USING.lock().unwrap_or_else(PoisonError::into_inner).clone();
    let mine = std::process::id().to_string();
    for path in entries(&folder(root, "leases")) {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let Ok(pid) = name.parse::<u64>() else {
            continue;
        };
        if name == mine {
            continue;
        }
        if crate::heavy::alive(pid) {
            let listed = fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str::<Vec<String>>(&text).ok());
            leased.extend(listed.into_iter().flatten());
        } else {
            let _ = fs::remove_file(&path);
        }
    }
    leased
}

/// Whether `name` is a full commit id, as checkouts and builds are named.
fn is_commit(name: &str) -> bool {
    name.len() == 40 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// What `folder` holds.
fn entries(folder: &Path) -> impl Iterator<Item = PathBuf> {
    fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
}

/// The folder `path`, renamed aside to be deleted, if it is there and
/// nothing holds it open.
fn set_aside(path: &Path) -> Option<PathBuf> {
    if !path.exists() {
        return None;
    }
    let doomed = path.with_extension(REMOVING);
    match fs::rename(path, &doomed) {
        Ok(()) => Some(doomed),
        Err(error) => {
            eprintln!("Keeping {}, which is in use: {error}", path.display());
            None
        }
    }
}

/// A lock among this repository's processes: the folder
/// `target/measure/NAME.lock`, holding its owner's process id, released
/// when dropped; a lock whose owner ended is taken over.
struct Lock(PathBuf);

impl Lock {
    /// Takes lock `NAME`, waiting while a running process holds it.
    fn hold(root: &Path, name: &str) -> Result<Self, String> {
        let path = folder(root, &format!("{name}.lock"));
        let owner = path.join("owner.json");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut told = false;
        loop {
            match fs::create_dir(&path) {
                Ok(()) => {
                    let mine = json!({"pid": std::process::id(), "since": iso(SystemTime::now())});
                    fs::write(&owner, mine.to_string())
                        .map_err(|error| format!("cannot write {}: {error}", owner.display()))?;
                    return Ok(Self(path));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(format!("cannot make {}: {error}", path.display())),
            }
            let holder = || {
                fs::read_to_string(&owner)
                    .ok()
                    .and_then(|text| serde_json::from_str::<Value>(&text).ok())
                    .and_then(|held| held["pid"].as_u64())
            };
            let held = holder();
            let stale = match held {
                Some(pid) => !crate::heavy::alive(pid),
                // A holder that stopped between making the folder and
                // writing its file.
                None => fs::metadata(&path)
                    .and_then(|metadata| metadata.modified())
                    .ok()
                    .and_then(|made| made.elapsed().ok())
                    .is_some_and(|age| age > crate::heavy::OWNERLESS),
            };
            // Taken over only from the holder found stale, not one that
            // took it since.
            if stale && holder() == held {
                let _ = fs::remove_dir_all(&path);
                continue;
            }
            if !told {
                eprintln!(
                    "Waiting for {}, which process {} holds.",
                    path.display(),
                    held.map_or_else(|| "?".to_owned(), |pid| pid.to_string())
                );
                told = true;
            }
            std::thread::sleep(LOCK_POLL);
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
