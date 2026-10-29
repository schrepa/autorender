use std::fmt::Display;
use std::path::Path;
use std::process::ExitCode;

use engine::Engine;

use crate::stages::{self, Stage};
use crate::{Exceptions, Repository, Rules};

const USAGE: &str = "usage: cargo xtask verify [--level quick|pr|merge] | schema";

const SCHEMA: &str = "cargo xtask schema";

const WASM_TARGET: &str = "wasm32-unknown-unknown";

type Build = Box<dyn Fn() -> Result<Engine, String>>;

pub struct App {
    pub repository: Repository,
    build: Build,
}

impl App {
    pub fn engine(&self) -> Result<Engine, String> {
        (self.build)()
    }

    pub fn generated(&self) -> Result<Vec<(String, String)>, String> {
        let mut engine = self.engine()?;
        let registered = crate::bricks::registered_crates(&engine);
        let mut files = vec![(
            crate::bricks::FILE.to_owned(),
            crate::bricks::bricks_json(&self.repository, &mut engine)?,
        )];
        files.extend(crate::faces::files(&self.repository, &registered)?);
        Ok(files)
    }
}

const QUICK: usize = 8;

const STAGES: &[Stage<App>] = &[
    Stage {
        name: "policy",
        needs: &[],
        run: |app| crate::policy::check(&app.repository, &Rules::NONE),
    },
    Stage {
        name: "comments",
        needs: &[],
        run: |app| crate::comments::check(&app.repository),
    },
    Stage {
        name: "fmt",
        needs: &[],
        run: |app| crate::tools::fmt(&app.repository),
    },
    Stage {
        name: "structure",
        needs: &[],
        run: |app| crate::structure::check(&app.repository, &Exceptions::NONE),
    },
    Stage {
        name: "generated files",
        needs: &[],
        run: |app| crate::generated::check(&app.repository.root, &app.generated()?, SCHEMA),
    },
    Stage {
        name: "dependencies",
        needs: &["generated files"],
        run: |app| crate::dependencies::check(&app.repository, &Exceptions::NONE),
    },
    Stage {
        name: "faces",
        needs: &[],
        run: |app| crate::faces::check(&app.repository),
    },
    Stage {
        name: "schedule ambiguities",
        needs: &[],
        run: |app| crate::ambiguities::check(app.engine()?),
    },
    Stage {
        name: "clippy",
        needs: &[],
        run: |app| {
            let mut selection = Vec::new();
            for package in app.repository.own() {
                selection.extend(["-p", package.name.as_str()]);
            }
            selection.push("--all-targets");
            crate::tools::clippy(&app.repository.root, &selection)
        },
    },
    Stage {
        name: "clippy wasm",
        needs: &[],
        run: |app| {
            let mut selection = Vec::new();
            for project in app.repository.projects() {
                selection.extend(["-p", project.name.as_str()]);
            }
            selection.extend(["--target", WASM_TARGET]);
            crate::tools::clippy(&app.repository.root, &selection)
        },
    },
];

pub fn main<G: 'static, E: Display + 'static>(
    folder: &str,
    build: fn() -> Result<(Engine, G), E>,
) -> ExitCode {
    match run(folder, build) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run<G: 'static, E: Display + 'static>(
    folder: &str,
    build: fn() -> Result<(Engine, G), E>,
) -> Result<(), String> {
    let root = Path::new(folder)
        .parent()
        .ok_or_else(|| format!("{folder} has no parent folder"))?;
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let app = App {
        repository: Repository::read(root)?,
        build: Box::new(move || {
            build()
                .map(|(engine, _)| engine)
                .map_err(|error| error.to_string())
        }),
    };
    match arguments.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["verify"] | ["verify", "--level", "pr" | "merge"] => stages::run(&app, STAGES),
        ["verify", "--level", "quick"] => stages::run(&app, &STAGES[..QUICK]),
        ["schema"] => crate::generated::write(&app.repository.root, &app.generated()?),
        _ => Err(USAGE.to_owned()),
    }
}
