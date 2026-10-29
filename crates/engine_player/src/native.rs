use std::path::PathBuf;
use std::process::ExitCode;

use engine::command_line::CommandOption;
use engine::{Engine, Error, ErrorCode};
use engine_native::{Arguments, HostOptions, files};

use crate::{ScenePlayer, bricks};

fn player_options() -> [CommandOption; 2] {
    [
        CommandOption::taking("--source", "PATH").required(),
        CommandOption::taking("--without", "BRICK,..."),
    ]
}

fn options() -> Vec<CommandOption> {
    let mut options = player_options().to_vec();
    options.extend(engine_native::options());
    options
}

#[cfg(feature = "dev")]
fn parameters() -> Vec<(String, String)> {
    player_options()
        .into_iter()
        .filter_map(|option| {
            let name = option.name.strip_prefix("--")?.to_owned();
            Some((name, option.value?))
        })
        .collect()
}

struct Options {
    source: Option<PathBuf>,
    without: String,
    host: HostOptions,
}

pub fn main(arguments: impl Iterator<Item = String>) -> ExitCode {
    let arguments: Vec<String> = arguments.collect();
    if let Some(help) = engine::command_line::help(&arguments, &options()) {
        println!("{help}");
        return ExitCode::SUCCESS;
    }
    match Options::parse(arguments) {
        Ok(options) => engine_native::exit(options.run()),
        Err(error) => {
            let record = serde_json::json!({"event": "arguments_rejected", "error": error});
            eprintln!("{record}");
            ExitCode::from(2)
        }
    }
}

impl Options {
    fn parse(arguments: Vec<String>) -> Result<Self, String> {
        let mut options = Self {
            source: None,
            without: String::new(),
            host: HostOptions::default(),
        };
        let mut arguments = Arguments::new(arguments);
        while let Some(argument) = arguments.next() {
            if options.host.take(&argument, &mut arguments)? {
                continue;
            }
            match argument.name.as_str() {
                "--source" => options.source = Some(arguments.value(&argument)?.into()),
                "--without" => options.without = arguments.value(&argument)?,
                _ => return Err(argument.unknown()),
            }
        }
        options.host.check()?;
        Ok(options)
    }

    fn run(&self) -> engine::Result<()> {
        #[cfg(feature = "dev")]
        if self.host.dev {
            let mut session = self
                .host
                .dev_session(|| self.build())?
                .parameters(parameters());
            if let Some(path) = &self.source {
                session = session.source(path);
            }
            return session.run();
        }
        self.host.run("Engine Player", || self.build())
    }

    fn build(&self) -> engine::Result<(Engine, ScenePlayer)> {
        let path = self.source.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArguments,
                "$.source",
                "required: --source PATH",
            )
        })?;
        let source = std::fs::read_to_string(path).map_err(|error| {
            Error::new(
                ErrorCode::MissingFile,
                "$.source",
                format!("{}: {error}", path.display()),
            )
        })?;
        let files = files::scene_files(path, &source)?;
        let without = bricks(&self.without, "$.without")?;
        ScenePlayer::new(&source, &files, &without).map_err(Error::from)
    }
}
