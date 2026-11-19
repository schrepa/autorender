mod authoring;
mod bricks;
mod development;
mod lint;
mod rules;
mod shader_report;

use std::{env, process::ExitCode};

use engine_command_line::{self as command_line, Command, CommandOption};
use xtask::{pipeline, process};

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("xtask: {failure}");
            ExitCode::FAILURE
        }
    }
}

fn execute() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let task = arguments.next().unwrap_or_else(|| "help".to_owned());
    let options: Vec<_> = arguments.collect();
    if matches!(task.as_str(), "help" | "--help" | "-h")
        || options.iter().any(|option| option == "--help")
    {
        return help(&options);
    }
    let root = process::repository_root()?;
    if task == "scene" {
        return development::scene(&options);
    }
    if task == "build" {
        let profiling = match options.as_slice() {
            [] => false,
            [option] if option == "--profiling" => true,
            _ => return Err("usage: build [--profiling]".into()),
        };
        authoring::validate_sources(&root)?;
        return pipeline::build(&root, profiling);
    }
    if task == "lint" {
        return lint::lint(&root, &options);
    }
    if task == "shaders" {
        return shader_report::run(&root, &options);
    }
    if task == "generate" {
        return authoring::generate(&root, &options);
    }
    if let Some(argument) = options.first() {
        return Err(
            format!("unexpected argument {argument:?}; run `cargo xtask help` for usage"),
        );
    }
    Err(format!("unknown command {task:?}; run `cargo xtask help` for usage"))
}

fn commands() -> Vec<Command> {
    let mut commands = vec![
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
            "shaders",
            [
                CommandOption::taking("--out", "DIR").required(),
                CommandOption::flag("--no-rga"),
            ],
        ),
    ];
    commands.extend(xtask::agent::commands());
    commands.extend([
        engine_checks::stages::command(),
        Command::new("help", [CommandOption::flag("--json")]),
    ]);
    commands
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
