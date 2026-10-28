use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use serde_json::json;

use super::FOLDER;
use crate::bench;
use crate::measure::{self, git, read_json};

const PACKAGE: &str = "stress-lab";

const PROGRAM: &str = "stress-lab-native";

const PROFILE: &str = "quick";

const FEATURES: &str = "dev";

const KEEP_RUNS: usize = 6;

pub(super) struct Side {
    pub(super) label: String,
    pub(super) sha: Option<String>,
    pub(super) program: PathBuf,
    pub(super) tree: PathBuf,
    pub(super) assets: PathBuf,
}

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
    if let Some(library) = measure::dxc() {
        let beside = into.join("dxcompiler.dll");
        if measure::file_hash(&beside).ok() != measure::file_hash(&library).ok() {
            fs::copy(&library, &beside)
                .map_err(|error| format!("cannot copy {}: {error}", library.display()))?;
        }
    }
    Ok(program)
}

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
    for folder in crate::checkouts::RUN_FILES {
        crate::checkouts::copy_folder(&tree.join(folder), &kept_tree.join(folder))?;
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

pub(super) fn checkout_tree(root: &Path, name: &str, sha: &str) -> Result<PathBuf, String> {
    measure::ensure_commit(root, sha, None)?;
    let tree = root.join(FOLDER).join(name);
    crate::checkouts::check_out(root, &tree, sha)?;
    Ok(tree)
}

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
