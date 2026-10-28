use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use serde_json::{Map, Value, json};

use super::FOLDER;
use crate::measure::{self, git, read_json};
use crate::process::Project;

const PROFILE: &str = "quick";

const FEATURES: &str = "dev";

const KEEP_RUNS: usize = 6;

pub(super) struct Side {
    pub(super) label: String,
    pub(super) sha: Option<String>,
    pub(super) programs: BTreeMap<String, PathBuf>,
    pub(super) tree: PathBuf,
}

impl Side {
    pub(super) fn program(&self, project: &str) -> Result<&Path, String> {
        self.programs
            .get(project)
            .map(PathBuf::as_path)
            .ok_or_else(|| format!("{} built no program of project {project}", self.label))
    }
}

fn build_program(
    tree: &Path,
    target: Option<&Path>,
    commit: &str,
    project: &Project,
    into: &Path,
) -> Result<PathBuf, String> {
    let mut command = Command::new("cargo");
    command
        .args(["build", "-p", &project.package, "--bin", &project.native])
        .args(["--profile", PROFILE, "--features", FEATURES, "--locked"])
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
            "building {} in {} failed ({status})",
            project.native,
            tree.display()
        ));
    }
    let built = target
        .map_or_else(|| tree.join("target"), Path::to_path_buf)
        .join(PROFILE)
        .join(project.program());
    fs::create_dir_all(into).map_err(|error| error.to_string())?;
    let program = into.join(project.program());
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

pub(super) fn working_tree(root: &Path, projects: &[&str]) -> Result<Side, String> {
    let (_, commit) = crate::process::engine_commit();
    eprintln!("Building the working tree ({commit}).");
    let into = root.join(FOLDER).join("working-tree");
    let mut programs = BTreeMap::new();
    for folder in projects {
        let project = Project::read(root, folder)?;
        let program = build_program(root, None, &commit, &project, &into)?;
        programs.insert((*folder).to_owned(), program);
    }
    Ok(Side {
        label: format!("the working tree ({commit})"),
        sha: None,
        programs,
        tree: root.to_path_buf(),
    })
}

pub(super) fn commit_side(
    root: &Path,
    role: &str,
    rev: &str,
    projects: &[&str],
) -> Result<Side, String> {
    let sha = git(
        root,
        &["rev-parse", "--verify", &format!("{rev}^{{commit}}")],
    )
    .map_err(|_| format!("{rev} names no commit"))?;
    let cache = root.join(FOLDER).join("builds").join(&sha);
    let record = cache.join("build.json");
    let label = sha[..12].to_owned();
    let mut wanted = BTreeMap::new();
    for folder in projects {
        let project = Project::at(root, &sha, folder)?;
        wanted.insert((*folder).to_owned(), project);
    }
    let built = |build: &Value, project: &Project| {
        let program = cache.join(project.program());
        measure::file_hash(&program).ok().as_deref()
            == build["programs"][project.program()].as_str()
    };
    let kept = read_json(&record).ok().filter(|build| {
        cache.join("tree").is_dir() && wanted.values().all(|project| built(build, project))
    });
    let programs: BTreeMap<String, PathBuf> = wanted
        .iter()
        .map(|(folder, project)| (folder.clone(), cache.join(project.program())))
        .collect();
    if kept.is_some() {
        eprintln!("Reusing the build of {label} ({role}).");
        fs::write(cache.join("used"), measure::iso(SystemTime::now()))
            .map_err(|error| error.to_string())?;
        return Ok(Side {
            label,
            programs,
            tree: cache.join("tree"),
            sha: Some(sha),
        });
    }
    let tree = checkout_tree(root, "tree", &sha)?;
    eprintln!("Building {label} ({role}).");
    let target = root.join(FOLDER).join("target");
    let mut hashes = Map::new();
    for project in wanted.values() {
        let program = build_program(&tree, Some(&target), &label, project, &cache)?;
        hashes.insert(project.program(), measure::file_hash(&program)?.into());
    }
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
        "programs": hashes,
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
        programs,
        tree: kept_tree,
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
