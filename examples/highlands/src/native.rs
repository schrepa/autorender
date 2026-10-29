use std::path::{Path, PathBuf};
use std::process::ExitCode;

use engine::Engine;
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
    if arguments.iter().any(|argument| argument == "--help") {
        println!("{}\n{}", usage(), engine_native::usage());
        return ExitCode::SUCCESS;
    }
    match Native::parse(arguments) {
        Ok(native) => engine_native::exit(native.run()),
        Err(message) => {
            let record = serde_json::json!({"event": "arguments_rejected", "error": message});
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

fn usage() -> String {
    let lines: String = options()
        .into_iter()
        .map(|(name, values)| format!("  --{name} {values}\n"))
        .collect();
    format!("Usage: highlands-native [OPTIONS]\n\n{lines}")
}

impl Native {
    fn parse(arguments: Vec<String>) -> Result<Self, String> {
        let mut native = Self {
            options: Options::default(),
            assets: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/public"),
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
                native
                    .options
                    .set(name, &value)
                    .map_err(|error| error.to_string())?;
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
        let mut session = self.host.dev_session(build)?.parameters(options());
        for (name, description) in crate::controls() {
            session = session.control(name, &description, move |host, args| {
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
        session.run()
    }
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
