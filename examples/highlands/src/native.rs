use std::path::{Path, PathBuf};
use std::process::ExitCode;

use engine::Engine;
use engine::command_line::CommandOption;
use engine_native::{Arguments, HostOptions, files};

use crate::land::ROCK_FILES;
use crate::{Highlands, Options, RockMaps, parameters};

struct Native {
    options: Options,
    assets: PathBuf,
    host: HostOptions,
}

pub fn main(arguments: impl Iterator<Item = String>) -> ExitCode {
    let arguments: Vec<String> = arguments.collect();
    if let Some(help) = engine::command_line::help(&arguments, &command_options()) {
        println!("{help}");
        return ExitCode::SUCCESS;
    }
    match Native::parse(arguments) {
        Ok(native) => native.start(),
        Err(error) => {
            let record = serde_json::json!({"event": "arguments_rejected", "error": error});
            eprintln!("{record}");
            ExitCode::from(2)
        }
    }
}

fn options() -> Vec<(&'static str, String)> {
    let mut options = parameters();
    options.push(("assets", "DIR".to_owned()));
    options
}

fn default_assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/public")
}

fn command_options() -> Vec<CommandOption> {
    let defaults = Options::default();
    let mut options: Vec<CommandOption> = options()
        .into_iter()
        .map(|(name, values)| {
            let option = CommandOption::taking(&format!("--{name}"), values);
            match defaults.value(name) {
                Some(default) => option.defaults_to(default),
                None if name == "assets" => option.defaults_to(default_assets().display()),
                None => option,
            }
        })
        .collect();
    options.extend(engine_native::options());
    options
}

impl Native {
    fn parse(arguments: Vec<String>) -> engine::Result<Self> {
        let mut native = Self {
            options: Options::default(),
            assets: default_assets(),
            host: HostOptions::default(),
        };
        let mut arguments = Arguments::new(arguments);
        while let Some(argument) = arguments.next() {
            if native.host.take(&argument, &mut arguments)? {
                continue;
            }
            let name = argument.name.strip_prefix("--").unwrap_or_default();
            if name == "assets" {
                native.assets = arguments.value(&argument)?.into();
            } else if parameters().iter().any(|(known, _)| *known == name) {
                let value = arguments.value(&argument)?;
                native.options.set(name, &value)?;
            } else {
                return Err(argument.unknown());
            }
        }
        if native.host.jobs.packs.is_none() {
            native.host.jobs.packs = Some(native.assets.join("cooked"));
        }
        native.host.check()?;
        Ok(native)
    }

    fn start(&self) -> ExitCode {
        #[cfg(feature = "dev")]
        if let Some(names) = &self.host.smoke {
            let build = |source: Option<&str>, without: &[&str]| match source {
                None => Ok((
                    engine::smokes::standard_engine(without)?,
                    Highlands::new(self.options.clone()),
                )),
                Some(_) => Err(
                    engine::Error::new(engine::ErrorCode::UnsupportedCommand, "$.source")
                        .expected(engine::Expected::Absent),
                ),
            };
            return engine_native::smoke::run::<Highlands>(
                &self.host,
                names,
                &crate::smokes(),
                &build,
                &controlled,
            );
        }
        engine_native::exit(self.run())
    }

    fn run(&self) -> engine::Result<()> {
        let build = || self.build();
        #[cfg(feature = "dev")]
        if self.host.dev {
            return self.run_dev(build);
        }
        self.host.run("Highlands", build)
    }

    fn build(&self) -> engine::Result<(Engine, Highlands)> {
        let rocks = if self.options.rocks {
            rocks(&self.assets.join("highlands/rocks"))
        } else {
            RockMaps::default()
        };
        Highlands::build(self.options.clone(), &rocks)
    }

    #[cfg(feature = "dev")]
    fn run_dev(
        &self,
        build: impl FnOnce() -> engine::Result<(Engine, Highlands)>,
    ) -> engine::Result<()> {
        let session = self.host.dev_session(build)?.parameters(options());
        controlled(session).run()
    }
}

#[cfg(feature = "dev")]
fn controlled(
    mut session: engine_native::dev::DevSession<Highlands>,
) -> engine_native::dev::DevSession<Highlands> {
    for (name, values) in crate::controls() {
        session = session.control(name, &values, move |host, args| {
            let value = args
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let (engine, highlands) = host.parts_mut();
            highlands
                .control(engine, name, value)
                .map_err(|error| error.to_string())?;
            let mut reply = serde_json::Map::new();
            reply.insert(name.to_owned(), value.into());
            Ok(reply.into())
        });
    }
    session
}

fn rocks(folder: &Path) -> RockMaps {
    ROCK_FILES
        .iter()
        .filter_map(|name| {
            let bytes = std::fs::read(folder.join(name)).ok()?;
            if files::image_format(&bytes).map(|format| format.name) != Some("ktx2-rgba8") {
                return None;
            }
            let image = files::decode_image(name, &bytes).ok()?;
            Some((format!("rocks/{name}"), image))
        })
        .collect()
}
