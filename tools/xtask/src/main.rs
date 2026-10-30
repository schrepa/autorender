mod authoring;
mod bench;
mod bricks;
mod checkouts;
mod checks;
mod cook;
mod coverage;
mod development;
mod equivalence;
mod heavy;
mod importing;
mod iteration_build;
mod ladder;
mod lint;
mod measure;
mod pipeline;
mod process;
mod profiling;
mod pure;
mod rules;
mod scaffold;
mod selfcheck;
mod serve;
mod shader_report;
mod smoke;
mod verify;

use std::{env, process::ExitCode};

use engine_core::command_line::{self, Command, CommandOption};

pub struct Failure {
    pub message: String,
    pub code: u8,
}

impl Failure {
    pub fn found(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 3,
        }
    }
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self { message, code: 1 }
    }
}

impl From<&str> for Failure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

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
    engine_worker::register_standard_layers();
    bricks::register_standard_kinds();
    let mut arguments = env::args().skip(1);
    let task = arguments.next().unwrap_or_else(|| "help".to_owned());
    let options: Vec<_> = arguments.collect();
    if matches!(task.as_str(), "help" | "--help" | "-h")
        || options.iter().any(|option| option == "--help")
    {
        return Ok(help(&options)?);
    }
    let root = process::repository_root()?;
    if task == "dev" {
        return Ok(pipeline::dev(&root, dev_port(&options)?)?);
    }
    if task == "agent" {
        return development::agent(&root, &options);
    }
    if task == "scene" {
        return Ok(development::scene(&options)?);
    }
    if task == "asset" {
        return Ok(importing::asset(&options)?);
    }
    if task == "profile" {
        return Ok(match options.as_slice() {
            [] => profiling::build(&root, false),
            [option] if option == "--attribution" => profiling::build(&root, true),
            _ => Err("usage: profile [--attribution]".into()),
        }?);
    }
    if task == "smoke" {
        return smoke::smoke(&root, &options);
    }
    if task == "pure" {
        return pure::pure(&root, &options);
    }
    if task == "core-smoke" {
        let mut corpus = vec!["caches".to_owned()];
        corpus.extend(options);
        return pure::pure(&root, &corpus);
    }
    if task == "bench" {
        return Ok(bench::bench(&root, &options)?);
    }
    if task == "measure" {
        return Ok(measure::measure(&root, &options)?);
    }
    if task == "ladder" {
        return Ok(ladder::ladder(&root, &options)?);
    }
    if task == "cook" {
        return cook::cook(&root, &options);
    }
    if task == "equivalence" {
        return equivalence::equivalence(&root, &options);
    }
    if task == "selfcheck" {
        return selfcheck::selfcheck(&root, &options);
    }
    if task == "verify" {
        return verify::verify(&root, &options);
    }
    if task == "checks" {
        return Ok(checks::check_task(&root, &options)?);
    }
    if task == "new" {
        return Ok(scaffold::new(&root, &options)?);
    }
    if task == "gpu" && !options.is_empty() {
        return Ok(shader_report::run(&root, &options)?);
    }
    if task == "build-dev" {
        let [option, output] = options.as_slice() else {
            return Err("usage: build-dev --output web/pkg-iterations/GENERATION".into());
        };
        if option != "--output" {
            return Err("build-dev requires --output".into());
        }
        let result = iteration_build::build(&root, std::path::Path::new(output));
        println!("{result}");
        return if result["ok"] == true {
            Ok(())
        } else {
            Err("development build failed; see JSON diagnostics".into())
        };
    }
    if let Some(argument) = options.first() {
        return Err(
            format!("unexpected argument {argument:?}; run `cargo xtask help` for usage").into(),
        );
    }
    Ok(match task.as_str() {
        "build" => pipeline::build(&root),
        "lint" => engine_checks::stages::run(root.as_path(), lint::LINT),
        "assets" => authoring::validate_sources(&root),
        "schema" => authoring::write_generated(&root),
        "gpu" => rules::repository(&root).and_then(engine_checks::gpu::check),
        _ => Err(format!(
            "unknown command {task:?}; run `cargo xtask help` for usage"
        )),
    }?)
}

const DEV_PORT: u16 = 5173;

fn dev_port(arguments: &[String]) -> Result<u16, String> {
    match arguments {
        [] => Ok(DEV_PORT),
        [option, value] if option == "--port" => value
            .parse::<u16>()
            .ok()
            .filter(|port| *port != 0)
            .ok_or_else(|| "dev --port requires an integer from 1 to 65535".to_owned()),
        _ => Err("usage: cargo xtask dev [--port N]".to_owned()),
    }
}

fn commands() -> Vec<Command> {
    let mut commands = authoring_commands();
    commands.extend(checking_commands());
    commands
}

fn authoring_commands() -> Vec<Command> {
    let verbs = engine_dev::Verb::names().collect::<Vec<_>>().join("|");
    let corpora = pure::CORPORA
        .iter()
        .map(|corpus| corpus.name)
        .collect::<Vec<_>>()
        .join("|");
    vec![
        Command::new("build", []),
        Command::new("profile", [CommandOption::flag("--attribution")]),
        Command::new(
            "dev",
            [CommandOption::taking("--port", "N").defaults_to(DEV_PORT)],
        ),
        Command::new(
            "agent iterate",
            [
                CommandOption::taking("--game", process::game_names()).required(),
                CommandOption::taking("--web-port", "N"),
                CommandOption::taking("--port", "N"),
            ],
        ),
        Command::new("agent help", []),
        Command::new(
            "agent",
            [
                CommandOption::taking("VERB", verbs).required(),
                CommandOption::taking("--session", "S"),
                CommandOption::taking("--json", "JSON"),
                CommandOption::taking("--file", "PATH"),
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
        Command::new("scene schema", []),
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
        Command::new("assets", []),
        Command::new("schema", []),
        Command::new(
            "new app",
            [CommandOption::taking("--project", "DIR").required()],
        ),
        Command::new(
            "new brick",
            [
                CommandOption::taking("PART", "NAME").required(),
                CommandOption::taking("NAME", "NAME").required(),
            ],
        ),
        Command::new(
            "smoke",
            [
                CommandOption::taking("NAME", "NAME...|all"),
                CommandOption::taking("--project", "DIR"),
                CommandOption::flag("--browser"),
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
        Command::new("core-smoke", [CommandOption::flag("--bless")]),
        Command::new(
            "gpu",
            [
                CommandOption::taking("--report", "DIR"),
                CommandOption::flag("--no-rga"),
            ],
        ),
        Command::new("bench check", []),
    ]
}

fn checking_commands() -> Vec<Command> {
    vec![
        Command::new(
            "cook",
            [
                CommandOption::taking("--project", "DIR"),
                CommandOption::taking("SCENE", "PATH.scene.json..."),
                CommandOption::flag("--check"),
            ],
        ),
        Command::new(
            "measure",
            [
                CommandOption::taking("--task", "FILE").required(),
                CommandOption::taking("--profile", "FILE").required(),
                CommandOption::taking("--out", "FILE").required(),
                CommandOption::taking("--extend", "FILE"),
                CommandOption::taking("--formats", "FILE"),
            ],
        ),
        Command::new(
            "ladder",
            [
                CommandOption::taking("LADDER", "FILE").required(),
                CommandOption::taking("--out", "DIR").required(),
                CommandOption::taking("--program", "PATH"),
                CommandOption::taking("--", "ARGUMENTS..."),
            ],
        ),
        Command::new(
            "equivalence",
            [
                CommandOption::taking("--base", "REV").defaults_to("origin/main"),
                CommandOption::taking("--candidate", "REV"),
                CommandOption::taking("--tier", "smoke|full").defaults_to("smoke"),
                CommandOption::taking("--cells", "NAME,..."),
                CommandOption::flag("--aa"),
                CommandOption::taking("--repeat", "N"),
                CommandOption::taking("--keep", "N").defaults_to(equivalence::KEEP),
                CommandOption::taking("--gpu", "discrete|integrated").defaults_to("discrete"),
                CommandOption::flag("--browser"),
                CommandOption::flag("--both"),
                CommandOption::flag("--hosts"),
            ],
        ),
        Command::new(
            "checks",
            [
                CommandOption::taking("--task", "FILE").required(),
                CommandOption::taking("--profile", "FILE").required(),
                CommandOption::taking("--out", "FILE").required(),
                CommandOption::taking("--formats", "FILE"),
            ],
        ),
        Command::new(
            "selfcheck",
            [CommandOption::taking(
                "KIND",
                format!("{}...", selfcheck::KINDS.join("|")),
            )],
        ),
        Command::new(
            "verify",
            [
                CommandOption::taking("--level", "quick|pr|merge").defaults_to("pr"),
                CommandOption::taking("--base", "REV").defaults_to("origin/main"),
                CommandOption::flag("--plan"),
                CommandOption::flag("--json"),
            ],
        ),
        Command::new("lint", []),
        Command::new("help", [CommandOption::flag("--json")]),
    ]
}

fn help(options: &[String]) -> Result<(), String> {
    let json = options.iter().any(|option| option == "--json");
    if json {
        let mut help = command_line::commands_json(&commands());
        help["agent"] = serde_json::json!({ "verbs": development::verbs()? });
        println!("{help}");
    } else {
        println!("{}", command_line::commands_text(&commands()));
    }
    Ok(())
}
