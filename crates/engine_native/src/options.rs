use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine::command_line::CommandOption;
use engine::host::{HostSettings, JobRef, OVERLAY, SceneLoad};
use engine::scene::SceneKinds;
use engine::{
    BrickList, DefaultScene, Engine, Error, ErrorCode, Expected, Game, GameEntry, GameHost,
    GpuPreference,
};
use serde_json::json;

use crate::{CacheOptions, JobOptions, Offscreen, Window};

const MIB: u64 = 1024 * 1024;

pub fn options() -> Vec<CommandOption> {
    let defaults = HostOptions::default();
    let gpu = match defaults.gpu {
        GpuPreference::HighPerformance => "discrete",
        GpuPreference::LowPower => "integrated",
    };
    let jobcache = match defaults.jobs.cache {
        CacheOptions::Off => "off".to_owned(),
        CacheOptions::Verify { sample: None } => "verify".to_owned(),
        CacheOptions::Verify {
            sample: Some(count),
        } => format!("verify:{count}"),
        CacheOptions::Keep { budget } => (budget / MIB).to_string(),
    };
    vec![
        CommandOption::taking("--scene", "PATH"),
        CommandOption::taking("--size", "WxH").in_unit("px"),
        CommandOption::taking("--gpu", "discrete|integrated").defaults_to(gpu),
        CommandOption::taking("--workers", "N|off").defaults_to(defaults.jobs.workers),
        CommandOption::taking("--jobcache", "off|verify|verify:N|N")
            .defaults_to(jobcache)
            .in_unit("MiB"),
        CommandOption::taking("--vsync", "on|off").defaults_to(switch(defaults.vsync)),
        CommandOption::taking(&format!("--{OVERLAY}"), "KEY=JSON"),
        CommandOption::taking("--limit-ms", "MS").in_unit("ms"),
        CommandOption::taking("--offscreen", "[N]"),
        CommandOption::taking("--packs", "DIR"),
        CommandOption::taking("--cook", "FILE"),
        CommandOption::flag("--dev"),
        CommandOption::taking("--run", "NAME"),
        CommandOption::taking("--record", "JSON"),
        CommandOption::flag("--schemas"),
        CommandOption::flag("--shaders"),
        CommandOption::taking("--pure", "NAME,...|all|list"),
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

#[derive(Clone, Debug, Default)]
pub enum StartScene {
    #[default]
    Empty,
    File(PathBuf),
    Declared {
        scene: DefaultScene,
        folder: String,
    },
}

pub fn open_scene<G: Game>(host: &mut GameHost<G>, scene: &StartScene) -> engine::Result<()> {
    let kinds = SceneKinds::of(host.engine().app.world());
    match scene {
        StartScene::Empty => {}
        StartScene::Declared { scene, folder } => {
            let (document, files) = SceneLoad::new(scene.text, &kinds).and_then(SceneLoad::opened)?;
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
    pub jobs: JobOptions,
    pub vsync: bool,
    pub settings: HostSettings,
    pub limit_ms: Option<u64>,
    pub headless: bool,
    pub frames: Option<u32>,
    pub dev: bool,
    pub run: Option<String>,
    pub record: Option<String>,
    pub schemas: bool,
    pub shaders: bool,
    pub pure: Option<String>,
    pub cook: Option<PathBuf>,
    pub workers_given: bool,
}

impl Default for HostOptions {
    fn default() -> Self {
        Self {
            scene: None,
            assets: None,
            size: None,
            gpu: GpuPreference::HighPerformance,
            jobs: JobOptions::default(),
            vsync: true,
            settings: HostSettings::default(),
            limit_ms: None,
            headless: false,
            frames: None,
            dev: false,
            run: None,
            record: None,
            schemas: false,
            shaders: false,
            pure: None,
            cook: None,
            workers_given: false,
        }
    }
}

fn job_cache(argument: &Argument, arguments: &mut Arguments) -> Result<CacheOptions, Error> {
    let cache = arguments.value(argument)?;
    if let Some(count) = cache.strip_prefix("verify:") {
        let sample = count
            .parse()
            .ok()
            .filter(|&count: &u32| count > 0)
            .ok_or_else(|| {
                argument
                    .invalid(&cache)
                    .expected(Expected::Integer)
                    .range(1.0, f64::from(u32::MAX))
            })?;
        return Ok(CacheOptions::Verify {
            sample: Some(sample),
        });
    }
    Ok(match cache.as_str() {
        "off" => CacheOptions::Off,
        "verify" => CacheOptions::Verify { sample: None },
        mebibytes => CacheOptions::Keep {
            budget: mebibytes
                .parse::<u64>()
                .ok()
                .and_then(|mebibytes| mebibytes.checked_mul(MIB))
                .ok_or_else(|| argument.invalid(&cache).range(0.0, (u64::MAX / MIB) as f64))?,
        },
    })
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
            "--workers" => {
                let workers = arguments.value(argument)?;
                self.workers_given = true;
                self.jobs.workers = match workers.as_str() {
                    "off" => 0,
                    count => count
                        .parse::<usize>()
                        .map_err(|_| argument.invalid(&workers).at_least(0))?,
                };
            }
            "--jobcache" => self.jobs.cache = job_cache(argument, arguments)?,
            "--vsync" => self.vsync = arguments.switch(argument)?,
            switch if switch.strip_prefix("--").is_some_and(HostSettings::takes) => {
                let value = arguments.value(argument)?;
                self.settings
                    .set(&value)
                    .map_err(|error| error.within(switch))?;
            }
            "--packs" => self.jobs.packs = Some(arguments.value(argument)?.into()),
            "--cook" => self.cook = Some(arguments.value(argument)?.into()),
            "--limit-ms" => {
                let limit = arguments.value(argument)?;
                self.limit_ms = Some(
                    limit
                        .parse()
                        .ok()
                        .filter(|&limit: &u64| limit > 0)
                        .ok_or_else(|| argument.invalid(&limit).expected(Expected::Integer))?,
                );
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
            "--record" => self.record = Some(arguments.value(argument)?),
            "--schemas" => {
                arguments.flag(argument)?;
                self.schemas = true;
            }
            "--shaders" => {
                arguments.flag(argument)?;
                self.shaders = true;
            }
            "--pure" => self.pure = Some(arguments.value(argument)?),
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
            ("--record", self.record.is_some()),
            ("--schemas", self.schemas),
            ("--shaders", self.shaders),
            ("--pure", self.pure.is_some()),
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
        let others = [
            ("--offscreen", self.headless),
            ("--dev", self.dev),
            ("--run", self.run.is_some()),
            ("--record", self.record.is_some()),
        ];
        let other = others
            .iter()
            .find(|(_, given)| *given)
            .map(|(option, _)| *option);
        if self.cook.is_some()
            && let Some(other) = others
                .iter()
                .find(|(option, given)| *given && *option != "--offscreen")
                .map(|(option, _)| *option)
        {
            return Err(excludes(other));
        }
        if self.schemas
            && let Some(other) = other.or(self.cook.is_some().then_some("--cook"))
        {
            return Err(excludes(other));
        }
        if self.shaders
            && let Some(other) = other
                .or(self.cook.is_some().then_some("--cook"))
                .or(self.schemas.then_some("--schemas"))
        {
            return Err(excludes(other));
        }
        if self.pure.is_some()
            && let Some(other) = other
                .or(self.cook.is_some().then_some("--cook"))
                .or(self.schemas.then_some("--schemas"))
                .or(self.shaders.then_some("--shaders"))
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
        #[cfg(feature = "dev")]
        if let Some(names) = &options.pure {
            return crate::pure::run(names, move |_| bricks);
        }
        let scene = options.start_scene(&game);
        let build = game.build;
        #[cfg(feature = "dev")]
        if options.records("causes") {
            engine_profile::trace(true);
        }
        crate::job_cache::keep_pipeline_cache();
        if let Some(pack) = &options.cook {
            return options.run_cook(pack, move |_| bricks, build, &scene);
        }
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
        let assets = default_assets(game.folder);
        if options.jobs.packs.is_none() {
            options.jobs.packs = Some(assets.join("cooked"));
        }
        options.assets = Some(assets);
        options
    }

    fn start_scene<G>(&self, game: &GameEntry<G>) -> StartScene {
        if let Some(path) = &self.scene {
            return StartScene::File(path.clone());
        }
        match game.scene {
            Some(scene) if cfg!(feature = "dev") && Path::new(scene.path).is_file() => {
                StartScene::File(scene.path.into())
            }
            Some(scene) => StartScene::Declared {
                scene,
                folder: folder_of(&self.assets.clone().unwrap_or_default().join(scene.name)),
            },
            None => StartScene::Empty,
        }
    }

    #[cfg(feature = "dev")]
    fn records(&self, instrument: &str) -> bool {
        self.record
            .as_deref()
            .and_then(|args| serde_json::from_str::<serde_json::Value>(args).ok())
            .is_some_and(|args| {
                args["instruments"]
                    .as_array()
                    .is_some_and(|names| names.iter().any(|name| name == instrument))
            })
    }

    fn size(&self) -> engine::Result<(u32, u32)> {
        self.size
            .ok_or_else(|| Error::new(ErrorCode::MissingField, "--size").expected(Expected::Present))
    }

    #[cfg(feature = "dev")]
    pub(crate) fn session<G: Game>(
        &self,
        host: &mut GameHost<G>,
        size: (u32, u32),
        scene: &StartScene,
        (shell, pack): (&str, serde_json::Value),
    ) -> engine::Result<crate::dev::Session> {
        let features: &[&str] = if cfg!(feature = "profiling") {
            &["dev", "profiling"]
        } else {
            &["dev"]
        };
        let scene = match scene {
            StartScene::Empty => json!({"unavailable": "no_scene_opened"}),
            StartScene::File(path) => {
                let workspace = std::env::current_dir().unwrap_or_default();
                let path = path.strip_prefix(&workspace).unwrap_or(path);
                json!(
                    path.to_string_lossy()
                        .replace(std::path::MAIN_SEPARATOR, "/")
                )
            }
            StartScene::Declared { scene, .. } => json!(scene.name),
        };
        let about = json!({
            "machine": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "adapter": engine::host::AdapterStatus::of(host.adapter()),
            },
            "host": "native",
            "shell": shell,
            "build": {"version": env!("CARGO_PKG_VERSION"), "features": features},
            "scene": scene,
            "picture": [size.0, size.1],
            "pack": pack,
        });
        let serde_json::Value::Object(about) = about else {
            unreachable!("about_is_an_object");
        };
        crate::dev::Session::open(
            host,
            crate::dev::Start {
                requests: self.dev,
                size,
                run: self.run.as_deref(),
                record: self.record.as_deref(),
                about,
            },
        )
    }

    fn run_cook<G: Game, E: Into<engine::Error>>(
        &self,
        path: &Path,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<G, E>,
        scene: &StartScene,
    ) -> engine::Result<()> {
        let limit = self.limit_ms.ok_or_else(|| {
            Error::new(ErrorCode::InvalidArguments, "--limit-ms").expected(Expected::Present)
        })?;
        let size = self.size()?;
        let started = Instant::now();
        let mut jobs = self.jobs.clone();
        if !self.workers_given {
            jobs.workers = std::thread::available_parallelism().map_or(1, usize::from);
        }
        let mut game = Offscreen::launch(size, self.gpu, jobs.pool(true), bricks, build)?;
        open_scene(game.host_mut(), scene)?;
        self.settings.apply(game.host_mut())?;
        let read = game.jobs_mut().packed();
        let (mut frame, mut said) = (0_u32, Instant::now());
        let (after, mut played) = (self.frames.unwrap_or(0), 0_u32);
        game.host_mut().set_preparation(true);
        loop {
            game.host_mut().cook();
            let at = f64::from(frame) * 1000.0 / 60.0;
            game.turn(at)?;
            frame += 1;
            let readiness = game.host().frame_times().readiness.clone();
            if !readiness.failed.is_empty() {
                return Err(Error::new(ErrorCode::RenderFailed, "--cook").got(json!({
                    "failed": readiness.failed,
                })));
            }
            let playing = readiness.settled() && !game.host().preparing();
            if playing && played >= after {
                break;
            }
            if playing {
                played += 1;
            }
            if started.elapsed() >= Duration::from_millis(limit) {
                return Err(Error::new(ErrorCode::CommandTimeout, "--limit-ms")
                    .at_most(limit)
                    .got(json!({
                        "readiness": readiness,
                        "jobs_held": game.host().jobs_held(),
                    })));
            }
            if said.elapsed() > Duration::from_secs(5) {
                said = Instant::now();
                let record = json!({
                    "event": "cooking",
                    "readiness": readiness,
                    "jobs_held": game.host().jobs_held(),
                });
                eprintln!("{record}");
            }
        }
        let written = game
            .jobs_mut()
            .write_pack(path)
            .map_err(|error| error.within("--cook"))?;
        let jobs = game.host().progress().1;
        println!(
            "{:#}",
            json!({
                "pack": path.display().to_string(),
                "records": written.records,
                "kinds": written.kinds,
                "bytes": written.bytes,
                "packed_before": read,
                "frames": frame,
                "seconds": started.elapsed().as_secs_f64(),
                "jobs": jobs,
            })
        );
        Ok(())
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
        let mut game = Offscreen::launch(size, self.gpu, self.jobs.pool(false), bricks, build)?;
        open_scene(game.host_mut(), scene)?;
        self.settings.apply(game.host_mut())?;
        #[cfg(feature = "dev")]
        let played = {
            let pack = pack_header(game.jobs_mut());
            let mut session = self.session(game.host_mut(), size, scene, ("offscreen", pack))?;
            game.play(self.frames, |host, phase| {
                crate::dev::follow_source(host);
                session.after(host)?;
                let wait = if phase == engine::host::FramePhase::Paused {
                    Duration::try_from_secs_f64(host.refresh_interval() / 1000.0)
                        .unwrap_or_default()
                } else {
                    Duration::ZERO
                };
                Ok(session.receive(host, wait))
            })?
        };
        #[cfg(not(feature = "dev"))]
        let played = game.play(self.frames, |_, _| Ok(true))?;
        if self.frames.is_none() {
            return Ok(());
        }
        let mut stats = serde_json::to_value(game.host().frame_times()).expect("stats_serialize");
        stats["jobs"] = json!(game.host().progress().1);
        let jobs = game.jobs_mut();
        let report = json!({
            "size": [size.0, size.1],
            "frames": played,
            "mismatched": named(jobs.mismatched()),
            "sampled": named(jobs.sampled()),
            "stats": stats,
        });
        println!("{report:#}");
        Ok(())
    }
}

#[cfg(feature = "dev")]
pub(crate) fn pack_header(jobs: &crate::JobPool) -> serde_json::Value {
    let mut packs: Vec<(String, u64)> = jobs
        .packs()
        .filter_map(|(path, bytes)| {
            Some((path.file_stem()?.to_string_lossy().into_owned(), bytes))
        })
        .collect();
    if packs.is_empty() {
        return json!({"unavailable": "no_pack_loaded"});
    }
    packs.sort();
    packs
        .into_iter()
        .map(|(name, bytes)| json!({"name": name, "bytes": bytes}))
        .collect()
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

fn named(records: &[(Option<JobRef>, String)]) -> Vec<serde_json::Value> {
    records
        .iter()
        .map(|(kind, key)| json!({ "kind": kind.as_ref().map(JobRef::name), "key": key }))
        .collect()
}
