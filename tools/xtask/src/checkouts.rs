use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::measure::{ensure_commit, git, iso};

const KEPT_BUILDS: usize = 12;

const KEPT_CHECKOUTS: usize = 3;

pub const RUN_FILES: &[&str] = &["bench"];

const REMOVING: &str = "removing";

const LOCK_POLL: Duration = Duration::from_secs(1);

const OWNERLESS: Duration = Duration::from_secs(60);

static USING: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

fn folder(root: &Path, name: &str) -> PathBuf {
    root.join("target").join("measure").join(name)
}

pub fn lease(root: &Path, sha: &str) -> Result<(), String> {
    let _registry = Lock::hold(root, "registry")?;
    let mut using = USING.lock().unwrap_or_else(PoisonError::into_inner);
    if using.insert(sha.to_owned()) {
        let leases = folder(root, "leases");
        fs::create_dir_all(&leases).map_err(|error| error.to_string())?;
        let listed = serde_json::to_string(&*using).expect("commits_serialize");
        fs::write(leases.join(std::process::id().to_string()), listed)
            .map_err(|error| format!("cannot write {}: {error}", leases.display()))?;
    }
    let used = folder(root, "used");
    fs::create_dir_all(&used).map_err(|error| error.to_string())?;
    fs::write(used.join(sha), iso(SystemTime::now()))
        .map_err(|error| format!("cannot write {}: {error}", used.display()))
}

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

pub struct Tree {
    path: PathBuf,
    _held: Lock,
}

impl Tree {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub fn reused(root: &Path, sha: &str, pull: Option<u64>) -> Result<Tree, String> {
    ensure_commit(root, sha, pull)?;
    lease(root, sha)?;
    let held = Lock::hold(root, "tree")?;
    let path = folder(root, "tree");
    check_out(root, &path, sha)?;
    Ok(Tree { path, _held: held })
}

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

pub fn prune(root: &Path) {
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
            Err(error) => {
                let record = json!({
                    "event": "checkout_removal_deferred",
                    "folder": path.display().to_string(),
                    "error": error.to_string(),
                });
                eprintln!("{record}");
            }
        }
    }
    if removed && let Err(error) = git(root, &["worktree", "prune"]) {
        eprintln!("{error}");
    }
}

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
    for path in entries(&folder(root, "used")) {
        let sha = path.file_name().unwrap_or_default();
        if !folder(root, "source").join(sha).exists() && !folder(root, "builds").join(sha).exists()
        {
            let _ = fs::remove_file(&path);
        }
    }
    doomed
}

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
        if alive(pid) {
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

fn is_commit(name: &str) -> bool {
    name.len() == 40 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn entries(folder: &Path) -> impl Iterator<Item = PathBuf> {
    fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
}

fn set_aside(path: &Path) -> Option<PathBuf> {
    if !path.exists() {
        return None;
    }
    let doomed = path.with_extension(REMOVING);
    match fs::rename(path, &doomed) {
        Ok(()) => Some(doomed),
        Err(error) => {
            let record = json!({
                "event": "checkout_in_use",
                "folder": path.display().to_string(),
                "error": error.to_string(),
            });
            eprintln!("{record}");
            None
        }
    }
}

struct Lock(PathBuf);

impl Lock {
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
                Some(pid) => !alive(pid),
                None => fs::metadata(&path)
                    .and_then(|metadata| metadata.modified())
                    .ok()
                    .and_then(|made| made.elapsed().ok())
                    .is_some_and(|age| age > OWNERLESS),
            };
            if stale && holder() == held {
                let _ = fs::remove_dir_all(&path);
                continue;
            }
            if !told {
                let record = json!({
                    "event": "lock_waiting",
                    "lock": path.display().to_string(),
                    "pid": held,
                });
                eprintln!("{record}");
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

fn alive(pid: u64) -> bool {
    let listed = if cfg!(windows) {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
            .output()
    } else {
        Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "pid="])
            .output()
    };
    match listed {
        Ok(output) if cfg!(windows) => {
            !output.status.success()
                || String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\""))
        }
        Ok(output) => {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).trim() == pid.to_string()
        }
        Err(_) => true,
    }
}
