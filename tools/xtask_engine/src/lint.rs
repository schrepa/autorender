use std::path::Path;

use engine_checks::stages::Stage;
use engine_checks::tools::clippy_free;

use crate::{
    pipeline::{QUICK, build_wasm, install_web_dependencies},
    process::{NPM, WASM_TARGET, clients, run},
    rules::{RULES, repository, warnings_allowed},
};

pub(crate) const LINT: &[Stage<Path>] = &[
    Stage {
        name: "kinds",
        needs: &[],
        run: kinds,
    },
    Stage {
        name: "policy",
        needs: &[],
        run: |root| engine_checks::policy::check(repository(root)?, &RULES),
    },
    Stage {
        name: "comments",
        needs: &[],
        run: |root| engine_checks::comments::check(repository(root)?),
    },
    Stage {
        name: "structure",
        needs: &[],
        run: |root| engine_checks::structure::check(repository(root)?, &RULES.exceptions),
    },
    Stage {
        name: "shader link",
        needs: &[],
        run: shader_link,
    },
    Stage {
        name: "clippy",
        needs: &[],
        run: clippy_workspace,
    },
    Stage {
        name: "clippy wasm",
        needs: &[],
        run: clippy_wasm,
    },
    Stage {
        name: "wasm",
        needs: &[],
        run: |root| build_wasm(root, false, QUICK),
    },
    Stage {
        name: "web dependencies",
        needs: &[],
        run: install_web_dependencies,
    },
    Stage {
        name: "tsc",
        needs: &["wasm", "web dependencies"],
        run: |root| run(root, NPM, &["run", "typecheck", "--prefix", "web"]),
    },
];

fn kinds(root: &Path) -> Result<(), String> {
    run(root, "cargo", &["run", "-q", "--locked", "-p", "kinds"])
}

fn shader_link(root: &Path) -> Result<(), String> {
    let folder = root.join("target").join("lint").join("shaders");
    crate::shader_report::run(
        root,
        &[
            "--report".to_owned(),
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

fn clippy_workspace(root: &Path) -> Result<(), String> {
    let free = warnings_allowed(root)?;
    let mut selection = vec!["--workspace", "--all-targets"];
    for package in &free {
        selection.extend(["--exclude", *package]);
    }
    clippy(root, &selection)?;
    clippy_free(root, &free, &["--all-targets"])
}

fn clippy_wasm(root: &Path) -> Result<(), String> {
    let free = warnings_allowed(root)?;
    let (free_clients, strict): (Vec<&str>, Vec<&str>) = clients()
        .iter()
        .map(|client| client.package.as_str())
        .partition(|package| free.contains(package));
    let options = ["--target", WASM_TARGET];
    let mut selection: Vec<&str> = Vec::new();
    for package in &strict {
        selection.extend(["-p", *package]);
    }
    selection.extend(options);
    clippy(root, &selection)?;
    clippy_free(root, &free_clients, &options)
}
