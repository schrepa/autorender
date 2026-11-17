use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine::command_line::CommandOption;
use engine::host::{HostSettings, JobRef, OVERLAY, SETTINGS, SceneLoad};
use engine::scene::SceneKinds;
use engine::{
    BrickList, DefaultScene, Engine, Error, ErrorCode, Expected, Game, GameEntry, GameHost,
    GpuPreference,
};
use serde_json::json;

use crate::{CacheOptions, Frames, JobOptions, Offscreen, Record, Window};

const MIB: u64 = 1024 * 1024;

const DEV_PARAMETERS: [&str; 5] = ["size", "gpu", "workers", "jobcache", "scene"];

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
    let mut options = vec![
        CommandOption::taking("--scene", "PATH"),
        CommandOption::taking("--size", "WxH").in_unit("px"),
        CommandOption::taking("--gpu", "discrete|integrated").defaults_to(gpu),
        CommandOption::taking("--workers", "N|off").defaults_to(defaults.jobs.workers),
        CommandOption::taking("--jobcache", "off|verify|verify:N|N")
            .defaults_to(jobcache)
            .in_unit("MiB"),
        CommandOption::taking("--vsync", "on|off").defaults_to(switch(defaults.vsync)),
    ];
    options.extend(SETTINGS.iter().map(|setting| {
        CommandOption::taking(&format!("--{}", setting.name), setting.accepted())
            .defaults_to(setting.default_value())
    }));
    options.extend([
        CommandOption::taking(&format!("--{OVERLAY}"), "KEY=JSON"),
        CommandOption::taking("--limit-ms", "MS").in_unit("ms"),
        CommandOption::taking("--offscreen", "N"),
        CommandOption::taking("--packs", "DIR"),
        CommandOption::taking("--cook", "FILE"),
        CommandOption::flag("--dev"),
        CommandOption::flag("--schemas"),
        CommandOption::taking("--pure", "NAME,...|all|list"),
        CommandOption::taking("--dev-artifacts", "DIR").defaults_to(default_artifacts().display()),
        CommandOption::flag("--help"),
        CommandOption::flag("--json"),
    ]);
    options
}

pub fn parameters() -> Vec<(String, String)> {
    options()
        .into_iter()
        .filter_map(|option| {
            let name = option.name.strip_prefix("--")?;
            let known = DEV_PARAMETERS.contains(&name) || HostSettings::takes(name);
            known.then(|| (name.to_owned(), option.value.clone().unwrap_or_default()))
        })
        .collect()
}

fn switch(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

fn default_artifacts() -> PathBuf {
    std::env::temp_dir().join("engine-captures")
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

    pub fn count(&mut self, argument: &Argument) -> Result<u32, Error> {
        let value = self.value(argument)?;
        value
            .parse()
            .ok()
            .filter(|&count: &u32| count > 0)
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
    pub offscreen: Option<u32>,
    pub dev: bool,
    pub dev_artifacts: Option<PathBuf>,
    pub schemas: bool,
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
            offscreen: None,
            dev: false,
            dev_artifacts: None,
            schemas: false,
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
                    .set(&switch[2..], &value)
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
            "--offscreen" => self.offscreen = Some(arguments.count(argument)?),
            "--dev" => {
                arguments.flag(argument)?;
                self.dev = true;
            }
            "--dev-artifacts" => self.dev_artifacts = Some(arguments.value(argument)?.into()),
            "--schemas" => {
                arguments.flag(argument)?;
                self.schemas = true;
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
        if self.dev && self.offscreen.is_some() {
            return Err(excludes("--offscreen"));
        }
        if self.dev && !cfg!(feature = "dev") {
            return Err(
                Error::new(ErrorCode::InvalidArguments, "--dev").expected(Expected::Supported)
            );
        }
        if self.schemas && !cfg!(feature = "dev") {
            return Err(
                Error::new(ErrorCode::InvalidArguments, "--schemas").expected(Expected::Supported)
            );
        }
        if self.pure.is_some() && !cfg!(feature = "dev") {
            return Err(
                Error::new(ErrorCode::InvalidArguments, "--pure").expected(Expected::Supported)
            );
        }
        let others = [
            ("--offscreen", self.offscreen.is_some()),
            ("--dev", self.dev),
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
        if self.pure.is_some()
            && let Some(other) = other
                .or(self.cook.is_some().then_some("--cook"))
                .or(self.schemas.then_some("--schemas"))
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
        if let Some(names) = &options.pure {
            return crate::pure::run(names, move |_| bricks);
        }
        let scene = options.start_scene(&game);
        let build = game.build;
        crate::job_cache::keep_pipeline_cache();
        if options.dev {
            #[cfg(feature = "dev")]
            return options.session(move |_| bricks, build, &scene)?.run();
        }
        if let Some(pack) = &options.cook {
            return options.run_cook(pack, move |_| bricks, build, &scene);
        }
        match options.offscreen {
            Some(frames) => options.run_offscreen(frames, (move |_| bricks, build), &scene),
            None => crate::run(
                Window {
                    size: options.size,
                    gpu: options.gpu,
                    jobs: options.jobs.clone(),
                    vsync: options.vsync,
                    settings: options.settings.clone(),
                    scene,
                    ..Window::new(game.title)
                },
                move |_| bricks,
                build,
            ),
        }
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
    fn session<G: Game, E: Into<engine::Error>>(
        &self,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<G, E>,
        scene: &StartScene,
    ) -> engine::Result<crate::dev::DevSession<G>> {
        let size = self.size.unwrap_or((1280, 720));
        let mut game = Offscreen::launch(size, self.gpu, &self.jobs, bricks, build)?;
        open_scene(game.host_mut(), scene)?;
        self.settings.apply(game.host_mut())?;
        let artifacts = self.dev_artifacts.clone().unwrap_or_else(default_artifacts);
        Ok(crate::dev::DevSession::new(game, artifacts).parameters(parameters()))
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
        let size = self.size.unwrap_or((320, 180));
        let started = Instant::now();
        let mut jobs = self.jobs.clone();
        if !self.workers_given {
            jobs.workers = std::thread::available_parallelism().map_or(1, usize::from);
        }
        let mut game = Offscreen::launch_with(size, self.gpu, jobs.pool(true), bricks, build)?;
        open_scene(game.host_mut(), scene)?;
        self.settings.apply(game.host_mut())?;
        let read = game.jobs_mut().packed();
        let (mut frame, mut said) = (0_u32, Instant::now());
        let (after, mut played) = (self.offscreen.unwrap_or(0), 0_u32);
        game.host_mut().set_preparation(true);
        loop {
            game.host_mut().cook();
            let at = f64::from(frame) * 1000.0 / 60.0;
            if game.host().preparing() {
                game.starting(at)?;
            } else {
                game.frame(at)?;
            }
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
        let stats: serde_json::Value =
            serde_json::from_str(&game.host().stats()).expect("stats_json");
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
                "jobs": stats["jobs"],
            })
        );
        Ok(())
    }

    fn run_offscreen<G: Game, E: Into<engine::Error>>(
        &self,
        frames: u32,
        (bricks, build): (
            impl FnOnce(BrickList) -> BrickList,
            impl FnOnce(&mut Engine) -> Result<G, E>,
        ),
        scene: &StartScene,
    ) -> engine::Result<()> {
        let size = self.size.unwrap_or((1280, 720));
        let started = Instant::now();
        let jobs = self.jobs.pool(false);
        let mut game = Offscreen::launch_with(size, self.gpu, jobs, bricks, build)?;
        open_scene(game.host_mut(), scene)?;
        self.settings.apply(game.host_mut())?;
        game.prepare(0.0)?;
        let prepared_ms = started.elapsed().as_secs_f64() * 1000.0;
        let at = |frame: u32| f64::from(frame) * 1000.0 / 60.0;
        let recorded = game.record(Frames::Count(frames), at, &Record::default())?;
        if let Some(error) = recorded.failed {
            return Err(error);
        }
        let stats: serde_json::Value =
            serde_json::from_str(&game.host().stats()).expect("stats_json");
        let jobs = game.jobs_mut();
        let (mismatched, sampled) = (named(jobs.mismatched()), named(jobs.sampled()));
        let report = json!({
            "size": [size.0, size.1],
            "frames": recorded.frames,
            "prepare_ms": prepared_ms,
            "frame_ms": recorded.elapsed_ms / f64::from(recorded.frames.max(1)),
            "cpu_ms": distribution(&recorded.cpu_ms),
            "main_stack_peak_bytes": engine_profiler::stack::peak(),
            "mismatched": mismatched,
            "sampled": sampled,
            "rows_frames": recorded.frames - recorded.frames / 2,
            "rows": rows(engine::profiling::take_rows()),
            "waits": rows(engine::profiling::take_waits()),
            "speed": speed(),
            "stats": stats,
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

fn named(records: &[(Option<JobRef>, String)]) -> Vec<serde_json::Value> {
    records
        .iter()
        .map(|(kind, key)| json!({ "kind": kind.as_ref().map(JobRef::name), "key": key }))
        .collect()
}

fn distribution(values: &[f64]) -> serde_json::Value {
    json!(engine::host::Distribution::of(values))
}

fn rows(rows: Vec<engine::profiling::Row>) -> serde_json::Value {
    rows.iter()
        .map(|row| {
            json!({
                "name": row.name,
                "ms": row.ms,
                "worst_ms": row.worst_ms,
                "runs": row.runs,
                "allocations": row.allocations,
            })
        })
        .collect()
}

fn speed() -> serde_json::Value {
    engine::profiling::take_speed().map_or(
        serde_json::Value::Null,
        |(fast, slow)| json!({ "fast_ms": fast, "slow_ms": slow }),
    )
}

