use std::path::Path;

use engine_checks::stages::Stage;
use engine_checks::tools::clippy_free;

use crate::{
    authoring,
    pipeline::{QUICK, build_wasm, install_web_dependencies},
    process::{DEV_FEATURES, NPM, WASM_TARGET, clients, run},
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
        name: "fmt",
        needs: &[],
        run: |root| engine_checks::tools::fmt(repository(root)?),
    },
    Stage {
        name: "web layers",
        needs: &[],
        run: |root| web_check(root, "check-layers.mjs"),
    },
    Stage {
        name: "web errors",
        needs: &[],
        run: |root| web_check(root, "check-errors.mjs"),
    },
    Stage {
        name: "web parameters",
        needs: &[],
        run: |root| web_check(root, "check-parameters.mjs"),
    },
    Stage {
        name: "scene sources",
        needs: &[],
        run: authoring::validate_sources,
    },
    Stage {
        name: "bench",
        needs: &[],
        run: |root| engine_checks::bench::check(repository(root)?),
    },
    Stage {
        name: "equivalence matrix",
        needs: &[],
        run: crate::equivalence::check,
    },
    Stage {
        name: "structure",
        needs: &[],
        run: |root| engine_checks::structure::check(repository(root)?, &RULES.exceptions),
    },
    Stage {
        name: "install",
        needs: &[],
        run: install,
    },
    Stage {
        name: "pure",
        needs: &[],
        run: |root| crate::pure::pure(root, &[]).map_err(|failure| failure.message),
    },
    Stage {
        name: "clippy",
        needs: &[],
        run: clippy_workspace,
    },
    Stage {
        name: "clippy native dev",
        needs: &[],
        run: |root| clippy_native(root, &["dev"]),
    },
    Stage {
        name: "clippy native profiling",
        needs: &[],
        run: |root| clippy_native(root, &["dev", "profiling"]),
    },
    Stage {
        name: "clippy wasm",
        needs: &[],
        run: |root| clippy_wasm(root, false),
    },
    Stage {
        name: "clippy wasm dev",
        needs: &[],
        run: |root| clippy_wasm(root, true),
    },
    Stage {
        name: "clippy app template",
        needs: &[],
        run: |root| {
            clippy(
                root,
                &[
                    "-p",
                    "engine",
                    "--examples",
                    "--features",
                    "dev",
                    "--target",
                    WASM_TARGET,
                ],
            )
        },
    },
    Stage {
        name: "wasm",
        needs: &[],
        run: |root| build_wasm(root, false, QUICK),
    },
    Stage {
        name: "wasm dev",
        needs: &[],
        run: |root| build_wasm(root, true, QUICK),
    },
    Stage {
        name: "web dependencies",
        needs: &[],
        run: install_web_dependencies,
    },
    Stage {
        name: "tsc",
        needs: &["wasm", "wasm dev", "web dependencies"],
        run: |root| run(root, NPM, &["run", "typecheck", "--prefix", "web"]),
    },
];

fn kinds(root: &Path) -> Result<(), String> {
    run(root, "cargo", &["run", "-q", "--locked", "-p", "kinds"])
}

fn install(_: &Path) -> Result<(), String> {
    let mut engine = crate::bricks::standard_engine().map_err(|error| error.to_string())?;
    engine_checks::install::check(&mut engine)
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

fn features_of(packages: &[&str], wanted: &[&str]) -> String {
    packages
        .iter()
        .flat_map(|package| {
            wanted
                .iter()
                .map(move |feature| format!("{package}/{feature}"))
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn clippy_native(root: &Path, wanted: &[&str]) -> Result<(), String> {
    let free = warnings_allowed(root)?;
    let (free_games, games): (Vec<&str>, Vec<&str>) = clients()
        .iter()
        .filter(|client| client.native.is_some())
        .map(|client| client.package.as_str())
        .partition(|game| free.contains(game));
    let mut featured = vec!["engine_native"];
    featured.extend(&games);
    let features = features_of(&featured, wanted);
    let mut selection = vec!["-p", "engine_native", "-p", "engine_profiler"];
    for game in &games {
        selection.extend(["-p", *game]);
    }
    selection.extend(["--all-targets", "--features", features.as_str()]);
    clippy(root, &selection)?;
    let features = features_of(&free_games, wanted);
    clippy_free(
        root,
        &free_games,
        &["--all-targets", "--features", features.as_str()],
    )
}

fn clippy_wasm(root: &Path, development: bool) -> Result<(), String> {
    let free = warnings_allowed(root)?;
    let (free_clients, strict): (Vec<&str>, Vec<&str>) = clients()
        .iter()
        .map(|client| client.package.as_str())
        .partition(|package| free.contains(package));
    let mut options = vec!["--target", WASM_TARGET];
    if development {
        options.extend(["--features", DEV_FEATURES]);
    }
    let mut selection: Vec<&str> = Vec::new();
    for package in &strict {
        selection.extend(["-p", *package]);
    }
    selection.extend(&options);
    clippy(root, &selection)?;
    clippy_free(root, &free_clients, &options)
}

fn web_check(root: &Path, script: &str) -> Result<(), String> {
    run(&root.join("web"), "node", &[&format!("scripts/{script}")])
}
