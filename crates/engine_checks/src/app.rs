use std::fmt::Display;
use std::path::Path;
use std::process::ExitCode;

use engine::Engine;
use engine_command_line::{self as command_line, Command, CommandOption};

use crate::install::List;
use crate::stages::{self, Changes, Stage};
use crate::{Exceptions, Repository, Rules};

const USAGE: &str = "usage: cargo xtask lint [--base REV] [--all] | generate [--check]";

const WASM_TARGET: &str = "wasm32-unknown-unknown";

type Build = Box<dyn Fn(&mut Engine) -> Result<(), String>>;

pub struct App {
    pub repository: Repository,
    list: List,
    build: Build,
}

impl App {
    pub fn engine(&self) -> Result<Engine, String> {
        let mut engine = crate::install::engine(self.list)?;
        (self.build)(&mut engine)?;
        Ok(engine)
    }

    pub fn generated(&self) -> Result<Vec<(String, String)>, String> {
        let mut engine = self.engine()?;
        Ok(vec![(
            crate::bricks::FILE.to_owned(),
            crate::bricks::bricks_json(&self.repository, self.list, &mut engine)?,
        )])
    }
}

struct Lint {
    app: App,
    changes: Changes,
}

impl Lint {
    fn packages(&self) -> Vec<&str> {
        self.changes.affected.iter().map(String::as_str).collect()
    }

    fn projects(&self) -> Vec<&str> {
        self.app
            .repository
            .projects()
            .map(|project| project.name.as_str())
            .filter(|name| self.changes.affects(name))
            .collect()
    }
}

const STAGES: &[Stage<Lint>] = &[
    Stage {
        name: "kinds",
        needs: &[],
        covers: |lint| lint.changes.any(|path| path.ends_with("Cargo.toml")),
        run: |lint| {
            crate::tools::run(
                &lint.app.repository.root,
                "cargo",
                &["run", "-q", "--locked", "-p", "kinds"],
            )
        },
    },
    Stage {
        name: "policy",
        needs: &[],
        covers: |lint| lint.changes.any(|_| true),
        run: |lint| crate::policy::check(&lint.app.repository, &Rules::NONE),
    },
    Stage {
        name: "comments",
        needs: &[],
        covers: |lint| lint.changes.any(crate::comments::covers),
        run: |lint| crate::comments::check(&lint.app.repository),
    },
    Stage {
        name: "fmt",
        needs: &[],
        covers: |lint| lint.changes.any(|path| path.ends_with(".rs")),
        run: |lint| crate::tools::fmt(&lint.app.repository),
    },
    Stage {
        name: "structure",
        needs: &[],
        covers: |lint| lint.changes.any(crate::files::source),
        run: |lint| crate::structure::check(&lint.app.repository, &Exceptions::NONE),
    },
    Stage {
        name: "install",
        needs: &[],
        covers: |lint| !lint.packages().is_empty(),
        run: |lint| crate::install::check(&mut lint.app.engine()?),
    },
    Stage {
        name: "generated",
        needs: &[],
        covers: |lint| !lint.packages().is_empty(),
        run: |lint| crate::generated::check(&lint.app.repository.root, &lint.app.generated()?),
    },
    Stage {
        name: "clippy",
        needs: &[],
        covers: |lint| !lint.packages().is_empty(),
        run: |lint| {
            crate::tools::clippy(&lint.app.repository, &lint.packages(), &["--all-targets"])
        },
    },
    Stage {
        name: "clippy wasm",
        needs: &[],
        covers: |lint| !lint.projects().is_empty(),
        run: |lint| {
            crate::tools::clippy(
                &lint.app.repository,
                &lint.projects(),
                &["--target", WASM_TARGET],
            )
        },
    },
];

pub fn main<G: 'static, E: Display + 'static>(
    folder: &str,
    list: List,
    build: fn(&mut Engine) -> Result<G, E>,
) -> ExitCode {
    match run(folder, list, build) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run<G: 'static, E: Display + 'static>(
    folder: &str,
    list: List,
    build: fn(&mut Engine) -> Result<G, E>,
) -> Result<(), String> {
    let root = Path::new(folder)
        .parent()
        .ok_or_else(|| format!("{folder} has no parent folder"))?;
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if let Some(json) = command_line::asks_help(&arguments) {
        let commands = [
            stages::command(),
            Command::new("generate", [CommandOption::flag("--check")]),
        ];
        if json {
            println!("{}", command_line::commands_json(&commands));
        } else {
            println!("{}", command_line::commands_text(&commands));
        }
        return Ok(());
    }
    let app = App {
        repository: Repository::read(root)?,
        list,
        build: Box::new(move |engine| build(engine).map(drop).map_err(|error| error.to_string())),
    };
    match arguments.split_first() {
        Some((task, options)) if task == "lint" => {
            let changes = Changes::read(&app.repository, options)?;
            stages::run(&Lint { app, changes }, STAGES)
        }
        Some((task, [])) if task == "generate" => {
            crate::generated::write(&app.repository.root, &app.generated()?)
        }
        Some((task, [check])) if task == "generate" && check == "--check" => {
            crate::generated::check(&app.repository.root, &app.generated()?)
        }
        _ => Err(USAGE.to_owned()),
    }
}
