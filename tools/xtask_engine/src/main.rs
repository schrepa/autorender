mod authoring;
mod bricks;
mod development;
mod importing;
mod lint;
mod pure;
mod rules;
mod scaffold;
mod shader_report;
mod ship;
mod smoke;

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
    if task == "asset" {
        return Ok(importing::asset(&options)?);
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
    if task == "smoke" {
        return smoke::smoke(&root, &options);
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
    commands.extend(checking_commands());
    commands
}

fn authoring_commands() -> Vec<Command> {
    let verbs = engine_dev::Verb::names().collect::<Vec<_>>().join("|");
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
            "agent",
            [
                CommandOption::taking("VERB", verbs).required(),
                CommandOption::taking("--session", "S"),
                CommandOption::taking("--json", "JSON"),
                CommandOption::taking("--file", "PATH"),
                CommandOption::taking("--timeout-ms", "MS").in_unit("ms"),
                CommandOption::flag("--brief"),
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
        Command::new(
            "asset import",
            [
                CommandOption::taking("--file", "PATH").required(),
                CommandOption::taking("--id", "ID").required(),
                CommandOption::taking("--output", "PATH"),
                CommandOption::taking("--into", "PATH"),
                CommandOption::taking("--expected-revision", "N"),
                CommandOption::flag("--replace"),
                CommandOption::flag("--as-prefab"),
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
                CommandOption::taking("--place", "PLACE"),
            ],
        ),
        Command::new(
            "smoke",
            [
                CommandOption::taking("NAME", "NAME...|all"),
                CommandOption::taking("--limit-ms", "MS").in_unit("ms"),
                CommandOption::taking("--project", "DIR"),
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
                CommandOption::taking("--project", "DIR"),
                CommandOption::taking("SCENE", "PATH.scene.json..."),
                CommandOption::taking("--limit-ms", "MS").in_unit("ms"),
                CommandOption::flag("--check"),
            ],
        ),
        Command::new(
            "ship",
            [CommandOption::taking("--project", "DIR").required()],
        ),
        Command::new(
            "look",
            [
                CommandOption::taking("PROJECT", "DIR").required(),
                CommandOption::taking("--limit-ms", "MS")
                    .in_unit("ms")
                    .required(),
                CommandOption::flag("--browser"),
                CommandOption::taking("--out", "DIR"),
            ],
        ),
        engine_checks::stages::command(),
        Command::new("help", [CommandOption::flag("--json")]),
    ]
}

fn help(options: &[String]) -> Result<(), String> {
    let json = options.iter().any(|option| option == "--json");
    if json {
        let mut help = command_line::commands_json(&commands());
        help["agent"] = serde_json::json!({
            "verbs": development::verbs()?,
            "calls": development::calls()?,
        });
        println!("{help}");
    } else {
        println!("{}", command_line::commands_text(&commands()));
    }
    Ok(())
}
