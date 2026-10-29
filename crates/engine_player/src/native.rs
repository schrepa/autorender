use std::path::PathBuf;
use std::process::ExitCode;

use engine::{Engine, Error, ErrorCode, Expected};
use engine_native::{Arguments, HostOptions, file_error, files};

use crate::{ScenePlayer, bricks};

const USAGE: &str = "Usage: engine-player-native --source PATH [--without BRICK,...] [OPTIONS]";

#[cfg(feature = "dev")]
const PARAMETERS: [(&str, &str); 2] = [
    ("source", "scene file"),
    ("without", "comma-separated features the engine leaves out"),
];

struct Options {
    source: Option<PathBuf>,
    without: String,
    host: HostOptions,
}

pub fn main(arguments: impl Iterator<Item = String>) -> ExitCode {
    let arguments: Vec<String> = arguments.collect();
    if arguments.iter().any(|argument| argument == "--help") {
        println!("{USAGE}\n\n{}", engine_native::usage());
        return ExitCode::SUCCESS;
    }
    match Options::parse(arguments) {
        Ok(options) => engine_native::exit(options.run()),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

impl Options {
    fn parse(arguments: Vec<String>) -> Result<Self, Error> {
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
                .parameters(PARAMETERS);
            if let Some(path) = &self.source {
                session = session.source(path);
            }
            return session.run();
        }
        self.host.run("Engine Player", || self.build())
    }

    fn build(&self) -> engine::Result<(Engine, ScenePlayer)> {
        let path = self.source.as_ref().ok_or_else(|| {
            Error::new(ErrorCode::InvalidArguments, "--source").expected(Expected::Present)
        })?;
        let source =
            std::fs::read_to_string(path).map_err(|error| file_error("--source", path, &error))?;
        let files = files::scene_files(path, &source)?;
        let without = bricks(&self.without, "--without")?;
        ScenePlayer::new(&source, &files, &without)
    }
}
