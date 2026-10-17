//! The two sides compared: each built, a commit in a reused checkout with
//! its build kept by commit, the working tree in place.

use std::path::{Path, PathBuf};

use crate::bench;
use crate::measure::{self, git};

/// What runs: the Stress Lab natively, with its development tools, built
/// as builds nothing is timed on are.
const PACKAGE: &str = "stress-lab";

const PROGRAM: &str = "stress-lab-native";

const PROFILE: &str = "quick";

const FEATURES: &str = "dev";

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

/// Cargo validates the requested build; the immutable copy stays leased for
/// this comparison, independently of later builds in the same context.
fn build_program(tree: &Path, commit: &str) -> Result<PathBuf, String> {
    let build = xtask::builds::cargo_with(
        tree,
        &[
            "-p",
            PACKAGE,
            "--bin",
            PROGRAM,
            "--profile",
            PROFILE,
            "--features",
            FEATURES,
            "--locked",
        ],
        "native",
        &[("ENGINE_COMMIT", commit)],
    )?;
    let libraries: Vec<_> = measure::dxc().into_iter().collect();
    let bundle = xtask::builds::bundle(tree, &build, PROGRAM, &libraries)?;
    Ok(xtask::cache::retain(bundle)?.join(measure::program_name()))
}

/// The working tree as the candidate: built in place, uncommitted changes
/// included.
pub(super) fn working_tree(root: &Path) -> Result<Side, String> {
    let (_, commit) = crate::process::engine_commit();
    eprintln!("Building the working tree ({commit}).");
    let program = build_program(root, &commit)?;
    let assets = bench::assets(root)?.folder;
    Ok(Side {
        label: format!("the working tree ({commit})"),
        sha: None,
        program,
        tree: root.to_path_buf(),
        assets,
    })
}

/// The requested revision, validated before every build and pinned while used.
pub(super) fn commit_side(root: &Path, role: &str, rev: &str) -> Result<Side, String> {
    let sha = git(
        root,
        &["rev-parse", "--verify", &format!("{rev}^{{commit}}")],
    )?;
    let tree = crate::checkouts::checkout(root, &sha, None)?;
    let label = sha[..12].to_owned();
    eprintln!("Preparing {label} ({role}).");
    let program = build_program(&tree, &label)?;
    let assets = bench::assets(&tree)?.folder;
    Ok(Side {
        label,
        sha: Some(sha),
        program,
        tree,
        assets,
    })
}

/// Names remain accepted for browser callers; source identity determines reuse.
pub(super) fn checkout_tree(root: &Path, _name: &str, sha: &str) -> Result<PathBuf, String> {
    crate::checkouts::checkout(root, sha, None)
}

/// Source/compiler/artifact retention follows the shared byte budget. Result
/// folders are caller-owned and survive cache collection.
pub(super) fn prune(root: &Path) {
    crate::checkouts::prune(root);
}
