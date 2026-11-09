use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use engine_checks::stages::{Changes, Stage};
use engine_checks::tools::clippy_free;

use crate::{
    pipeline::{build_wasm, install_web_dependencies},
    process::{NPM, WASM_TARGET, clients, run},
    pure,
    rules::{RULES, repository, warnings_allowed},
};

const TOOLS: &str = "xtask_engine";

const KINDS: &str = "kinds";

struct Lint {
    root: PathBuf,
    changes: Changes,
}

const LINT: &[Stage<Lint>] = &[
    Stage {
        name: "kinds",
        needs: &[],
        covers: |lint| {
            lint.changes.affects(KINDS) || lint.changes.any(|path| path.ends_with("Cargo.toml"))
        },
        run: |lint| run(&lint.root, "cargo", &["run", "-q", "--locked", "-p", KINDS]),
    },
    Stage {
        name: "policy",
        needs: &[],
        covers: |lint| lint.changes.any(|_| true),
        run: |lint| engine_checks::policy::check(repository(&lint.root)?, &RULES),
    },
    Stage {
        name: "comments",
        needs: &[],
        covers: |lint| lint.changes.any(engine_checks::comments::covers),
        run: |lint| engine_checks::comments::check(repository(&lint.root)?),
    },
    Stage {
        name: "structure",
        needs: &[],
        covers: |lint| lint.changes.any(engine_checks::files::source),
        run: |lint| engine_checks::structure::check(repository(&lint.root)?, &RULES.exceptions),
    },
    Stage {
        name: "shader link",
        needs: &[],
        covers: |lint| lint.changes.affects(TOOLS),
        run: shader_link,
    },
    Stage {
        name: "clippy",
        needs: &[],
        covers: |lint| !lint.changes.affected.is_empty(),
        run: clippy_changed,
    },
    Stage {
        name: "clippy wasm",
        needs: &[],
        covers: clients_changed,
        run: clippy_wasm,
    },
    Stage {
        name: "wasm",
        needs: &[],
        covers: clients_changed,
        run: |lint| build_wasm(&lint.root, false),
    },
    Stage {
        name: "web dependencies",
        needs: &[],
        covers: clients_changed,
        run: |lint| install_web_dependencies(&lint.root),
    },
    Stage {
        name: "tsc",
        needs: &["wasm", "web dependencies"],
        covers: clients_changed,
        run: |lint| run(&lint.root, NPM, &["run", "typecheck", "--prefix", "web"]),
    },
    Stage {
        name: "pure",
        needs: &[],
        covers: replay_changed,
        run: |lint| {
            pure::pure(&lint.root, &["--wasm".to_owned()]).map_err(|failure| failure.message)
        },
    },
];

pub fn lint(root: &Path, arguments: &[String]) -> Result<(), String> {
    let lint = Lint {
        root: root.to_path_buf(),
        changes: Changes::read(repository(root)?, arguments)?,
    };
    engine_checks::stages::run(&lint, LINT)
}

fn clients_changed(lint: &Lint) -> bool {
    clients()
        .iter()
        .any(|client| lint.changes.affects(&client.package))
}

fn replay_changed(lint: &Lint) -> bool {
    let (Ok(repository), Ok(corpora)) = (repository(&lint.root), pure::listed(&lint.root)) else {
        return true;
    };
    let found = |package: Option<&str>| {
        repository
            .own()
            .find(|own| Some(own.name.as_str()) == package)
    };
    let mut keys = BTreeSet::new();
    let mut folders = vec![
        "crates/engine_worker/".to_owned(),
        "tools/xtask_engine/src/pure".to_owned(),
        "tools/xtask_engine/pure/".to_owned(),
    ];
    for corpus in &corpora {
        let Some(owner) = found(corpus["owner"].as_str()) else {
            return true;
        };
        folders.extend(owner.file("src/pure/"));
        for package in corpus["replays"].as_array().into_iter().flatten() {
            let Some(replayed) = found(package.as_str()) else {
                return true;
            };
            keys.extend(engine_sources::job_key_files(&replayed.folder));
        }
    }
    lint.changes.any(|path| {
        keys.contains(path)
            || folders
                .iter()
                .any(|folder| path.starts_with(folder.as_str()))
            || path == pure::LISTED
            || path == "tools/xtask_engine/src/bricks.rs"
    })
}

fn shader_link(lint: &Lint) -> Result<(), String> {
    let folder = lint.root.join("target").join("lint").join("shaders");
    crate::shader_report::run(
        &lint.root,
        &[
            "--out".to_owned(),
            folder.display().to_string(),
            "--no-rga".to_owned(),
        ],
    )
}

fn clippy(root: &Path, selection: &[&str]) -> Result<(), String> {
    let mut arguments = vec!["clippy"];
    arguments.extend(selection);
    arguments.extend(["--locked", "--", "-D", "warnings"]);
    run(root, "cargo", &arguments)
}

fn clippy_changed(lint: &Lint) -> Result<(), String> {
    let changes = &lint.changes;
    let free: Vec<&str> = warnings_allowed(&lint.root)?
        .into_iter()
        .filter(|package| changes.affects(package))
        .collect();
    let strict: Vec<&str> = changes
        .affected
        .iter()
        .map(String::as_str)
        .filter(|package| !free.contains(package))
        .collect();
    let (features, free_features) = (
        required_features(&lint.root, &strict)?,
        required_features(&lint.root, &free)?,
    );
    let mut selection = vec!["--all-targets"];
    if !features.is_empty() {
        selection.extend(["--features", features.as_str()]);
    }
    if changes.every {
        selection.push("--workspace");
        for package in &free {
            selection.extend(["--exclude", *package]);
        }
    } else {
        for package in &changes.affected {
            if !free.contains(&package.as_str()) {
                selection.extend(["-p", package.as_str()]);
            }
        }
    }
    if changes.affected.len() > free.len() {
        clippy(&lint.root, &selection)?;
    }
    let mut options = vec!["--all-targets"];
    if !free_features.is_empty() {
        options.extend(["--features", free_features.as_str()]);
    }
    clippy_free(&lint.root, &free, &options)
}

fn required_features(root: &Path, packages: &[&str]) -> Result<String, String> {
    let repository = repository(root)?;
    Ok(packages
        .iter()
        .filter_map(|name| repository.package(name))
        .flat_map(|package| {
            package
                .required_features
                .iter()
                .map(|feature| format!("{}/{feature}", package.name))
        })
        .collect::<Vec<_>>()
        .join(","))
}

fn clippy_wasm(lint: &Lint) -> Result<(), String> {
    let free = warnings_allowed(&lint.root)?;
    let (free_clients, strict): (Vec<&str>, Vec<&str>) = clients()
        .iter()
        .map(|client| client.package.as_str())
        .filter(|package| lint.changes.affects(package))
        .partition(|package| free.contains(package));
    let options = ["--target", WASM_TARGET];
    if !strict.is_empty() {
        let mut selection: Vec<&str> = Vec::new();
        for package in &strict {
            selection.extend(["-p", *package]);
        }
        selection.extend(options);
        clippy(&lint.root, &selection)?;
    }
    clippy_free(&lint.root, &free_clients, &options)
}
