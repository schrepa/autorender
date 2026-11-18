mod authoring;
mod bricks;
mod development;
mod lint;
mod pure;
mod rules;
mod scaffold;
mod shader_report;
mod ship;

use std::{env, process::ExitCode};

use engine_command_line::{self as command_line, Command, CommandOption};
use xtask::{Failure, cook, pipeline, process};

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("xtask: {}", failure.message);
            ExitCode::from(failure.code)
        }
    }
}

fn execute() -> Result<(), Failure> {
    let mut arguments = env::args().skip(1);
    let task = arguments.next().unwrap_or_else(|| "help".to_owned());
    let options: Vec<_> = arguments.collect();
    if matches!(task.as_str(), "help" | "--help" | "-h")
        || options.iter().any(|option| option == "--help")
    {
        return Ok(help(&options)?);
    }
    let root = process::repository_root()?;
    if task == "scene" {
        return Ok(development::scene(&options)?);
    }
    if task == "build" {
        let profiling = match options.as_slice() {
            [] => false,
            [option] if option == "--profiling" => true,
            _ => return Err("usage: build [--profiling]".into()),
        };
        authoring::validate_sources(&root)?;
        return Ok(pipeline::build(&root, profiling)?);
    }
    if task == "pure" {
        return pure::pure(&root, &options);
    }
    if task == "ship" {
        return ship::ship(&root, &options);
    }
    if task == "lint" {
        return Ok(lint::lint(&root, &options)?);
    }
    if task == "new" {
        return Ok(scaffold::new(&root, &options)?);
    }
    if task == "shaders" {
        return Ok(shader_report::run(&root, &options)?);
    }
    if task == "generate" {
        return Ok(authoring::generate(&root, &options)?);
    }
    if let Some(argument) = options.first() {
        return Err(
            format!("unexpected argument {argument:?}; run `cargo xtask help` for usage").into(),
        );
    }
    Err(format!("unknown command {task:?}; run `cargo xtask help` for usage").into())
}

fn commands() -> Vec<Command> {
    let mut commands = authoring_commands();
    commands.extend(xtask::agent::commands());
    commands.push(xtask::compare::command());
    commands.extend(xtask::runs::commands());
    commands.extend(checking_commands());
    commands
}

fn authoring_commands() -> Vec<Command> {
    let corpora = process::repository_root()
        .map(|root| pure::names(&root))
        .unwrap_or_default()
        .join("|");
    vec![
        Command::new("build", [CommandOption::flag("--profiling")]),
        Command::new(
            "dev",
            [
                CommandOption::taking("--game", process::game_names()),
                CommandOption::taking("--port", "N").defaults_to(xtask::dev::PORT),
            ],
        ),
        Command::new(
            "scene validate",
            [CommandOption::taking("--file", "PATH").required()],
        ),
        Command::new(
            "scene format",
            [CommandOption::taking("--file", "PATH").required()],
        ),
        Command::new(
            "scene save",
            [
                CommandOption::taking("--file", "PATH").required(),
                CommandOption::taking("--candidate", "PATH").required(),
                CommandOption::taking("--expected-revision", "N").required(),
            ],
        ),
        Command::new("generate", [CommandOption::flag("--check")]),
        Command::new(
            "new app",
            [CommandOption::taking("--project", "DIR").required()],
        ),
        Command::new(
            "new brick",
            [
                CommandOption::taking("NAME", "NAME").required(),
                CommandOption::taking("--project", "DIR").required(),
            ],
        ),
        Command::new(
            "pure",
            [
                CommandOption::flag("--bless"),
                CommandOption::flag("--wasm"),
                CommandOption::taking("CORPUS", format!("{corpora}...")),
            ],
        ),
        Command::new(
            "shaders",
            [
                CommandOption::taking("--out", "DIR").required(),
                CommandOption::flag("--no-rga"),
            ],
        ),
    ]
}

fn checking_commands() -> Vec<Command> {
    vec![
        Command::new(
            "cook",
            [
                CommandOption::taking("--project", "DIR").required(),
                CommandOption::taking("SCENE", "PATH.scene.json..."),
                CommandOption::taking("--limit-ms", "MS").in_unit("ms"),
                CommandOption::flag("--check"),
            ],
        ),
        Command::new(
            "ship",
            [CommandOption::taking("--project", "DIR").required()],
        ),
        engine_checks::stages::command(),
        Command::new("help", [CommandOption::flag("--json")]),
    ]
}

fn help(options: &[String]) -> Result<(), String> {
    let json = options.iter().any(|option| option == "--json");
    if json {
        println!("{}", command_line::commands_json(&commands()));
    } else {
        println!("{}", command_line::commands_text(&commands()));
    }
    Ok(())
}
