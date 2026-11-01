use std::path::PathBuf;
use std::process::ExitCode;

use engine::command_line::CommandOption;
use engine::scene::SceneKinds;
use engine::{Engine, Error, ErrorCode, Expected};
use engine_native::{Arguments, HostOptions, file_error, files};

use crate::{ScenePlayer, listed};

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
        Ok(options) => options.start(),
        Err(error) => {
            let record = serde_json::json!({"event": "arguments_rejected", "error": error});
            eprintln!("{record}");
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

    fn start(&self) -> ExitCode {
        #[cfg(feature = "dev")]
        if let Some(names) = &self.host.smoke {
            let list = crate::bricks(engine::BrickList::default());
            let entries = match crate::smokes(&list) {
                Ok(entries) => entries,
                Err(error) => return engine_native::exit(Err(error)),
            };
            return engine_native::smoke::run::<engine::smokes::Played<ScenePlayer>>(
                &self.host,
                names,
                &entries,
                &list,
                &|source, engine| {
                    let world = engine.app.world_mut();
                    if source.is_none() && engine::host::has_setting(world, crate::player::FLIES) {
                        engine::host::apply_setting(
                            world,
                            crate::player::FLIES,
                            &serde_json::json!(false),
                        )?;
                    }
                    engine::smokes::played(source, engine, ScenePlayer::new)
                },
                &|session| session,
            );
        }
        engine_native::exit(self.run())
    }

    fn run(&self) -> engine::Result<()> {
        let bricks = listed(&self.without, "--without")?;
        #[cfg(feature = "dev")]
        if self.host.dev {
            let mut session = self
                .host
                .dev_session(bricks, |engine: &mut Engine| self.build(engine))?
                .parameters(parameters());
            if let Some(path) = &self.source {
                session = session.source(path);
            }
            return session.run();
        }
        self.host.run("Engine Player", bricks, |engine: &mut Engine| self.build(engine))
    }

    fn build(&self, engine: &mut Engine) -> engine::Result<ScenePlayer> {
        let path = self.source.as_ref().ok_or_else(|| {
            Error::new(ErrorCode::InvalidArguments, "--source").expected(Expected::Present)
        })?;
        let source =
            std::fs::read_to_string(path).map_err(|error| file_error("--source", path, &error))?;
        let kinds = SceneKinds::of(engine.app.world());
        let files = files::scene_files(path, &source, &kinds)?;
        ScenePlayer::new(&source, &files, engine)
    }
}
