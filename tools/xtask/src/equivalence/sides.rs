//! The two sides compared: each built, a commit in a reused checkout with
//! its build kept by commit, the working tree in place.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use serde_json::json;

use super::FOLDER;
use crate::bench;
use crate::measure::{self, git, read_json};

/// What runs: the Stress Lab natively, with its development tools, built
/// as builds nothing is timed on are.
const PACKAGE: &str = "stress-lab";

const PROGRAM: &str = "stress-lab-native";

const PROFILE: &str = "quick";

const FEATURES: &str = "dev";

/// Run folders kept.
const KEEP_RUNS: usize = 6;

/// One side of the comparison, built and ready to run.
pub(super) struct Side {
    /// The commit, or the working tree, as reports name it.
    pub(super) label: String,
    /// The commit, for a build kept by commit.
    pub(super) sha: Option<String>,
    pub(super) program: PathBuf,
    /// Where its bench files and fixtures are read.
    pub(super) tree: PathBuf,
    /// The generated files its workloads read (`--assets`).
    pub(super) assets: PathBuf,
}

/// Builds the program with `cargo` in `tree` (into `target`, or the tree's
/// own target folder) and copies it, with DXC's library where there is
/// one, into `into`: the copy's path.
fn build_program(
    tree: &Path,
    target: Option<&Path>,
    commit: &str,
    into: &Path,
) -> Result<PathBuf, String> {
    let mut command = Command::new("cargo");
    command
        .args([
            "build",
            "-p",
            PACKAGE,
            "--bin",
            PROGRAM,
            "--profile",
            PROFILE,
        ])
        .args(["--features", FEATURES, "--locked"])
        .env("ENGINE_COMMIT", commit)
        .current_dir(tree);
    if let Some(target) = target {
        command.env("CARGO_TARGET_DIR", target);
    }
    let status = command
        .status()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    if !status.success() {
        return Err(format!(
            "building {PROGRAM} in {} failed ({status})",
            tree.display()
        ));
    }
    let built = target
        .map_or_else(|| tree.join("target"), Path::to_path_buf)
        .join(PROFILE)
        .join(measure::program_name());
    fs::create_dir_all(into).map_err(|error| error.to_string())?;
    let program = into.join(measure::program_name());
    fs::copy(&built, &program)
        .map_err(|error| format!("cannot copy {}: {error}", built.display()))?;
    // D3D12 compiles shaders with DXC beside the program: both sides draw
    // through the same interface.
    if let Some(library) = measure::dxc() {
        let beside = into.join("dxcompiler.dll");
        if measure::file_hash(&beside).ok() != measure::file_hash(&library).ok() {
            fs::copy(&library, &beside)
                .map_err(|error| format!("cannot copy {}: {error}", library.display()))?;
        }
    }
    Ok(program)
}

/// The working tree as the candidate: built in place, uncommitted changes
/// included.
pub(super) fn working_tree(root: &Path) -> Result<Side, String> {
    let (_, commit) = crate::process::engine_commit();
    eprintln!("Building the working tree ({commit}).");
    let program = build_program(root, None, &commit, &root.join(FOLDER).join("working-tree"))?;
    let assets = bench::assets(root)?.folder;
    Ok(Side {
        label: format!("the working tree ({commit})"),
        sha: None,
        program,
        tree: root.to_path_buf(),
        assets,
    })
}

/// The commit `rev` names as `role`: built in the reused checkout
/// `target/equivalence/tree`, its program, bench files and fixtures kept in
/// `target/equivalence/builds/SHA/` and reused while its program is
/// unchanged.
pub(super) fn commit_side(root: &Path, role: &str, rev: &str) -> Result<Side, String> {
    let sha = git(
        root,
        &["rev-parse", "--verify", &format!("{rev}^{{commit}}")],
    )
    .map_err(|_| format!("{rev} names no commit"))?;
    let cache = root.join(FOLDER).join("builds").join(&sha);
    let record = cache.join("build.json");
    let program = cache.join(measure::program_name());
    let kept = read_json(&record).ok().filter(|build| {
        cache.join("tree").is_dir()
            && build["assets"]
                .as_str()
                .is_some_and(|assets| Path::new(assets).join("complete").is_file())
            && measure::file_hash(&program).ok().as_deref() == build["artifact_hash"].as_str()
    });
    let label = sha[..12].to_owned();
    if let Some(build) = kept {
        eprintln!("Reusing the build of {label} ({role}).");
        // The last use orders what the prune keeps.
        fs::write(cache.join("used"), measure::iso(SystemTime::now()))
            .map_err(|error| error.to_string())?;
        return Ok(Side {
            label,
            program,
            tree: cache.join("tree"),
            assets: PathBuf::from(build["assets"].as_str().unwrap_or_default()),
            sha: Some(sha),
        });
    }
    let tree = checkout_tree(root, "tree", &sha)?;
    eprintln!("Building {label} ({role}).");
    let program = build_program(
        &tree,
        Some(&root.join(FOLDER).join("target")),
        &label,
        &cache,
    )?;
    let assets = bench::assets(&tree)?.folder;
    let kept_tree = cache.join("tree");
    if kept_tree.exists() {
        fs::remove_dir_all(&kept_tree).map_err(|error| error.to_string())?;
    }
    for folder in ["bench", "web/scripts/fixtures"] {
        copy_folder(&tree.join(folder), &kept_tree.join(folder))?;
    }
    let build = json!({
        "sha": sha,
        "profile": PROFILE,
        "features": [FEATURES],
        "artifact_hash": measure::file_hash(&program)?,
        "assets": assets.display().to_string(),
    });
    fs::write(
        &record,
        serde_json::to_string_pretty(&build).expect("builds serialize"),
    )
    .map_err(|error| error.to_string())?;
    fs::write(cache.join("used"), measure::iso(SystemTime::now()))
        .map_err(|error| error.to_string())?;
    Ok(Side {
        label,
        sha: Some(sha),
        program,
        tree: kept_tree,
        assets,
    })
}

/// The reused checkout `target/equivalence/NAME` at commit `sha`, made the
/// first time.
pub(super) fn checkout_tree(root: &Path, name: &str, sha: &str) -> Result<PathBuf, String> {
    measure::ensure_commit(root, sha, None)?;
    let tree = root.join(FOLDER).join(name);
    if !tree.join(".git").exists() {
        let path = tree.display().to_string();
        git(
            root,
            &["worktree", "add", "--detach", "--force", &path, sha],
        )?;
    } else if git(&tree, &["rev-parse", "HEAD"])? != sha {
        git(&tree, &["checkout", "--detach", "--force", sha])?;
    }
    Ok(tree)
}

/// Copies the folder `from` into `into`, every file under it.
fn copy_folder(from: &Path, into: &Path) -> Result<(), String> {
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

/// Removes kept builds past the `keep` used last (but those of `using`),
/// and run folders past the last few.
pub(super) fn prune(root: &Path, keep: usize, using: &[&str]) {
    let used = |folder: &Path| {
        fs::metadata(folder.join("used"))
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    };
    let listed = |folder: PathBuf| -> Vec<PathBuf> {
        let mut entries: Vec<PathBuf> = fs::read_dir(folder)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        entries.sort_by_key(|path| std::cmp::Reverse(used(path)));
        entries
    };
    let builds = listed(root.join(FOLDER).join("builds"));
    let mut kept = 0;
    for build in builds {
        let sha = build.file_name().unwrap_or_default().to_string_lossy();
        if using.contains(&sha.as_ref()) {
            continue;
        }
        kept += 1;
        if kept + using.len() > keep {
            let _ = fs::remove_dir_all(&build);
        }
    }
    let mut runs: Vec<PathBuf> = fs::read_dir(root.join(FOLDER).join("runs"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .collect();
    runs.sort();
    let surplus = runs.len().saturating_sub(KEEP_RUNS);
    for run in &runs[..surplus] {
        let _ = fs::remove_dir_all(run);
    }
}
