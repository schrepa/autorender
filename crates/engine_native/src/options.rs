use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use engine::command_line::CommandOption;
use engine::host::{HostSettings, OVERLAY, SceneLoad};
use engine::scene::SceneKinds;
use engine::{
    BrickList, DefaultScene, Engine, Error, ErrorCode, Expected, Game, GameEntry, GameHost,
    GpuPreference,
};
use serde_json::json;

use crate::{Offscreen, Window};

pub fn options() -> Vec<CommandOption> {
    let defaults = HostOptions::default();
    let gpu = match defaults.gpu {
        GpuPreference::HighPerformance => "discrete",
        GpuPreference::LowPower => "integrated",
    };
    vec![
        CommandOption::taking("--scene", "PATH"),
        CommandOption::taking("--size", "WxH").in_unit("px"),
        CommandOption::taking("--gpu", "discrete|integrated").defaults_to(gpu),
        CommandOption::taking("--vsync", "on|off").defaults_to(switch(defaults.vsync)),
        CommandOption::taking(&format!("--{OVERLAY}"), "KEY=JSON"),
        CommandOption::taking("--offscreen", "[N]"),
        CommandOption::flag("--dev"),
        CommandOption::taking("--run", "NAME"),
        CommandOption::flag("--schemas"),
        CommandOption::flag("--shaders"),
        CommandOption::flag("--help"),
        CommandOption::flag("--json"),
    ]
}

fn switch(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub name: String,
    inline: Option<String>,
}

impl Argument {
    pub fn unknown(&self) -> Error {
        Error::new(ErrorCode::UnknownField, &self.name).expected(Expected::Absent)
    }

    pub fn invalid(&self, value: &str) -> Error {
        Error::new(ErrorCode::InvalidValue, &self.name).got(value)
    }
}

pub struct Arguments {
    rest: VecDeque<String>,
}

impl Arguments {
    pub fn new(arguments: impl IntoIterator<Item = String>) -> Self {
        Self {
            rest: arguments.into_iter().collect(),
        }
    }

    pub fn value(&mut self, argument: &Argument) -> Result<String, Error> {
        argument
            .inline
            .clone()
            .or_else(|| self.rest.pop_front())
            .ok_or_else(|| {
                Error::new(ErrorCode::MissingField, &argument.name).expected(Expected::Present)
            })
    }

    pub fn switch(&mut self, argument: &Argument) -> Result<bool, Error> {
        match self.value(argument)?.as_str() {
            "on" => Ok(true),
            "off" => Ok(false),
            value => Err(argument.invalid(value).variants(["on", "off"])),
        }
    }

    pub fn optional_count(&mut self, argument: &Argument) -> Result<Option<u32>, Error> {
        let value = match (&argument.inline, self.rest.front()) {
            (Some(value), _) => value.clone(),
            (None, Some(next)) if !next.starts_with("--") => {
                self.rest.pop_front().unwrap_or_default()
            }
            (None, Some(_) | None) => return Ok(None),
        };
        value
            .parse()
            .ok()
            .filter(|&count: &u32| count > 0)
            .map(Some)
            .ok_or_else(|| {
                argument
                    .invalid(&value)
                    .expected(Expected::Integer)
                    .range(1.0, f64::from(u32::MAX))
            })
    }

    pub fn flag(&self, argument: &Argument) -> Result<(), Error> {
        match &argument.inline {
            Some(value) => Err(argument.invalid(value).expected(Expected::Absent)),
            None => Ok(()),
        }
    }
}

impl Iterator for Arguments {
    type Item = Argument;

    fn next(&mut self) -> Option<Argument> {
        let argument = self.rest.pop_front()?;
        Some(match argument.split_once('=') {
            Some((name, value)) => Argument {
                name: name.to_owned(),
                inline: Some(value.to_owned()),
            },
            None => Argument {
                name: argument,
                inline: None,
            },
        })
    }
}

#[derive(Clone, Debug)]
pub enum StartScene {
    File(PathBuf),
    Declared { scene: DefaultScene, folder: String },
}

pub fn open_scene<G: Game>(host: &mut GameHost<G>, scene: &StartScene) -> engine::Result<()> {
    let kinds = SceneKinds::of(host.engine().app.world());
    match scene {
        StartScene::Declared { scene, folder } => {
            let (document, files) =
                SceneLoad::new(scene.text, &kinds).and_then(SceneLoad::opened)?;
            host.open_document(&document, &files)?;
            host.set_file_root(folder);
        }
        StartScene::File(path) => {
            let source = std::fs::read_to_string(path)
                .map_err(|error| crate::file_error("--scene", path, &error))?;
            let (document, files) = crate::files::scene_opened(path, &source, &kinds)?;
            host.open_document(&document, &files)?;
            host.set_file_root(&folder_of(path));
            #[cfg(feature = "dev")]
            host.watch_source(path);
        }
    }
    Ok(())
}

fn keep_folder(name: &str) -> Option<String> {
    let absolute = |variable: &str| {
        std::env::var_os(variable)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    let data = if cfg!(target_os = "windows") {
        absolute("APPDATA")
    } else if cfg!(target_os = "macos") {
        absolute("HOME").map(|home| home.join("Library").join("Application Support"))
    } else {
        absolute("XDG_DATA_HOME")
            .or_else(|| absolute("HOME").map(|home| home.join(".local").join("share")))
    };
    data.map(|data| data.join(name).to_string_lossy().into_owned())
}

fn folder_of(path: &Path) -> String {
    path.parent()
        .map(|folder| folder.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[derive(Clone, Debug)]
pub struct HostOptions {
    pub scene: Option<PathBuf>,
    pub assets: Option<PathBuf>,
    pub size: Option<(u32, u32)>,
    pub gpu: GpuPreference,
    pub vsync: bool,
    pub settings: HostSettings,
    pub headless: bool,
    pub frames: Option<u32>,
    pub dev: bool,
    pub run: Option<String>,
    pub schemas: bool,
    pub shaders: bool,
}

impl Default for HostOptions {
    fn default() -> Self {
        Self {
            scene: None,
            assets: None,
            size: None,
            gpu: GpuPreference::HighPerformance,
            vsync: true,
            settings: HostSettings::default(),
            headless: false,
            frames: None,
            dev: false,
            run: None,
            schemas: false,
            shaders: false,
        }
    }
}

impl HostOptions {
    pub fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, Error> {
        let mut options = Self::default();
        let mut arguments = Arguments::new(arguments);
        while let Some(argument) = arguments.next() {
            if !options.take(&argument, &mut arguments)? {
                return Err(argument.unknown());
            }
        }
        options.check()?;
        Ok(options)
    }

    pub fn take(&mut self, argument: &Argument, arguments: &mut Arguments) -> Result<bool, Error> {
        let name = argument.name.as_str();
        match name {
            "--scene" => self.scene = Some(arguments.value(argument)?.into()),
            "--size" => {
                let size = arguments.value(argument)?;
                let parsed = size
                    .split_once('x')
                    .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
                    .filter(|&(width, height): &(u32, u32)| width > 0 && height > 0);
                self.size = Some(parsed.ok_or_else(|| argument.invalid(&size))?);
            }
            "--gpu" => {
                self.gpu = match arguments.value(argument)?.as_str() {
                    "discrete" => GpuPreference::HighPerformance,
                    "integrated" => GpuPreference::LowPower,
                    other => {
                        return Err(argument.invalid(other).variants(["discrete", "integrated"]));
                    }
                }
            }
            "--vsync" => self.vsync = arguments.switch(argument)?,
            switch if switch.strip_prefix("--").is_some_and(HostSettings::takes) => {
                let value = arguments.value(argument)?;
                self.settings
                    .set(&value)
                    .map_err(|error| error.within(switch))?;
            }
            "--offscreen" => {
                self.headless = true;
                self.frames = arguments.optional_count(argument)?;
            }
            "--dev" => {
                arguments.flag(argument)?;
                self.dev = true;
            }
            "--run" => self.run = Some(arguments.value(argument)?),
            "--schemas" => {
                arguments.flag(argument)?;
                self.schemas = true;
            }
            "--shaders" => {
                arguments.flag(argument)?;
                self.shaders = true;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub fn check(&self) -> Result<(), Error> {
        let excludes = |option: &str| {
            Error::new(ErrorCode::InvalidArguments, option).expected(Expected::Absent)
        };
        let dev_only = [
            ("--dev", self.dev),
            ("--run", self.run.is_some()),
            ("--schemas", self.schemas),
            ("--shaders", self.shaders),
        ];
        if let Some((option, _)) = dev_only.iter().find(|(_, given)| *given)
            && !cfg!(feature = "dev")
        {
            return Err(
                Error::new(ErrorCode::InvalidArguments, *option).expected(Expected::Supported)
            );
        }
        if self.headless && self.frames.is_none() && !self.dev {
            return Err(
                Error::new(ErrorCode::MissingField, "--offscreen").expected(Expected::Present)
            );
        }
        let other = [
            ("--offscreen", self.headless),
            ("--dev", self.dev),
            ("--run", self.run.is_some()),
        ]
        .iter()
        .find(|(_, given)| *given)
        .map(|(option, _)| *option);
        if self.schemas
            && let Some(other) = other
        {
            return Err(excludes(other));
        }
        if self.shaders
            && let Some(other) = other.or(self.schemas.then_some("--schemas"))
        {
            return Err(excludes(other));
        }
        Ok(())
    }

    pub fn run<G: Game>(&self, game: GameEntry<G>) -> engine::Result<()> {
        let options = self.for_game(&game);
        let bricks = (game.bricks)(BrickList::default());
        #[cfg(feature = "dev")]
        if options.schemas {
            return engine::tooling::print_schemas(move |_| bricks).map_err(|problems| {
                Error::new(ErrorCode::InvalidArguments, "--schemas").got(problems)
            });
        }
        #[cfg(feature = "dev")]
        if options.shaders {
            return engine::tooling::print_shaders(move |_| bricks).map_err(|problems| {
                Error::new(ErrorCode::InvalidArguments, "--shaders").got(problems)
            });
        }
        let scene = options.start_scene(&game);
        let build = game.build;
        crate::keep_pipeline_cache();
        if options.headless {
            return options.run_offscreen((move |_| bricks, build), &scene);
        }
        crate::run(
            Window {
                title: game.title.to_owned(),
                keep: keep_folder(game.name),
                options: options.clone(),
                scene,
            },
            move |_| bricks,
            build,
        )
    }

    pub(crate) fn for_game<G>(&self, game: &GameEntry<G>) -> Self {
        let mut options = self.clone();
        options.assets = Some(default_assets(game.folder));
        options
    }

    fn start_scene<G>(&self, game: &GameEntry<G>) -> StartScene {
        if let Some(path) = &self.scene {
            return StartScene::File(path.clone());
        }
        let scene = game.scene;
        if cfg!(feature = "dev") && Path::new(scene.path).is_file() {
            return StartScene::File(scene.path.into());
        }
        StartScene::Declared {
            scene,
            folder: folder_of(&self.assets.clone().unwrap_or_default().join(scene.name)),
        }
    }

    fn size(&self) -> engine::Result<(u32, u32)> {
        self.size.ok_or_else(|| {
            Error::new(ErrorCode::MissingField, "--size").expected(Expected::Present)
        })
    }

    #[cfg(feature = "dev")]
    pub(crate) fn session<G: Game>(
        &self,
        host: &mut GameHost<G>,
        size: (u32, u32),
    ) -> engine::Result<crate::dev::Session> {
        crate::dev::Session::open(
            host,
            crate::dev::Start {
                requests: self.dev,
                size,
                run: self.run.as_deref(),
            },
        )
    }

    fn run_offscreen<G: Game, E: Into<engine::Error>>(
        &self,
        (bricks, build): (
            impl FnOnce(BrickList) -> BrickList,
            impl FnOnce(&mut Engine) -> Result<G, E>,
        ),
        scene: &StartScene,
    ) -> engine::Result<()> {
        let size = self.size()?;
        let mut game = Offscreen::launch(size, self.gpu, bricks, build)?;
        open_scene(game.host_mut(), scene)?;
        self.settings.apply(game.host_mut())?;
        #[cfg(feature = "dev")]
        let played = {
            let mut session = self.session(game.host_mut(), size)?;
            game.play(self.frames, |host, phase| {
                crate::dev::follow_source(host);
                session.after(host)?;
                let wait = if phase == engine::host::FramePhase::Paused {
                    std::time::Duration::try_from_secs_f64(host.refresh_interval() / 1000.0)
                        .unwrap_or_default()
                } else {
                    std::time::Duration::ZERO
                };
                Ok(session.receive(host, wait))
            })?
        };
        #[cfg(not(feature = "dev"))]
        let played = game.play(self.frames, |_, _| Ok(true))?;
        if self.frames.is_none() {
            return Ok(());
        }
        let report = json!({
            "size": [size.0, size.1],
            "frames": played,
            "stats": game.host().frame_times(),
        });
        println!("{report:#}");
        Ok(())
    }
}

fn default_assets(folder: &str) -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|program| Some(program.parent()?.join("assets")))
        .filter(|assets| assets.is_dir())
        .or_else(|| {
            Path::new(folder)
                .ancestors()
                .map(|parent| parent.join("web/public"))
                .find(|assets| assets.is_dir())
        })
        .unwrap_or_else(|| Path::new(folder).join("assets"))
}
