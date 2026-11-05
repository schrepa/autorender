use std::fmt::Display;
use std::path::Path;
use std::process::ExitCode;

use engine::Engine;
use engine_core::command_line::{self, Command, CommandOption};

use crate::install::List;
use crate::stages::{self, Stage};
use crate::{Exceptions, Repository, Rules};

const USAGE: &str = "usage: cargo xtask verify [--level quick|pr|merge] | schema";

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

const QUICK: usize = 6;

const STAGES: &[Stage<App>] = &[
    Stage {
        name: "kinds",
        needs: &[],
        run: |app| {
            crate::tools::run(
                &app.repository.root,
                "cargo",
                &["run", "-q", "--locked", "-p", "kinds"],
            )
        },
    },
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
        name: "install",
        needs: &[],
        run: |app| crate::install::check(&mut app.engine()?),
    },
    Stage {
        name: "clippy",
        needs: &[],
        run: |app| {
            let packages: Vec<&str> = app
                .repository
                .own()
                .map(|package| package.name.as_str())
                .collect();
            crate::tools::clippy(&app.repository, &packages, &["--all-targets"])
        },
    },
    Stage {
        name: "clippy wasm",
        needs: &[],
        run: |app| {
            let projects: Vec<&str> = app
                .repository
                .projects()
                .map(|project| project.name.as_str())
                .collect();
            crate::tools::clippy(&app.repository, &projects, &["--target", WASM_TARGET])
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
            Command::new(
                "verify",
                [CommandOption::taking("--level", "quick|pr|merge").defaults_to("pr")],
            ),
            Command::new("schema", []),
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
    match arguments.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["verify"] | ["verify", "--level", "pr" | "merge"] => stages::run(&app, STAGES),
        ["verify", "--level", "quick"] => stages::run(&app, &STAGES[..QUICK]),
        ["schema"] => crate::generated::write(&app.repository.root, &app.generated()?),
        _ => Err(USAGE.to_owned()),
    }
}
