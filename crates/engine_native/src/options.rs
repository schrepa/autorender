use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine::host::{HostSettings, SETTINGS, policy};
use engine::{Engine, Game, GpuPreference};
use serde_json::json;

use crate::{
    CacheOptions, Frames, JobOptions, Measure, Method, Offscreen, Pix, PixRecord, ProfileRecord,
    Record, Window, WindowReport, host_error,
};

const USAGE_HOST: &str = "\
Native options:
  --size WxH           window or offscreen size in physical pixels
  --gpu discrete|integrated
                       the GPU asked for where there are two (default discrete)
  --workers N|off      job worker threads (default every logical processor
                       but two, every one while cooking); off runs jobs in
                       the engine
  --jobcache off|verify|MiB
                       keep job outputs beside the program (default 1024
                       MiB); verify runs every job anyway and compares
  --vsync on|off       whether window frames wait for the display's refresh
                       (default on); off measures as browsers' frame rates do";

const USAGE_TOOLS: &str = "\
  --frames N           measure N window frames after --settle-ms, then close
                       and print their rate and CPU times as JSON
  --settle-ms MS       with --frames: play before measuring (default 20000)
  --offscreen FRAMES   draw FRAMES frames at 60 Hz offscreen and print their
                       statistics as JSON, without a window
  --capture PATH       with --offscreen: the last frame as a PNG file
  --gpu-capture PATH   with --offscreen: the last frame's GPU work, frame-exact:
                       a PIX capture (.wpix) where PIX is installed and the
                       GPU interface is D3D12; else a RenderDoc capture
                       (PATH_capture.rdc) under `renderdoccmd capture -w`;
                       else Xcode's, with MTL_CAPTURE_ENABLED=1
  --profile DIR        with --offscreen: a deep profile of the frames, on any
                       machine without setup: CPU samples of every thread
                       (functions, lines, instructions), the engine's spans,
                       each frame's GPU passes and draws, and the deeper
                       layers recommended here; files in DIR
  --counters on|LIST   with --profile, or --dev's profiles: each pass's GPU
                       hardware counters, where the driver offers Vulkan's
                       performance queries (Intel's; Mesa's on Linux), which
                       need no privileges: the first set one run counts, or
                       the comma-separated counters LIST names; the game then
                       draws on Vulkan, and counting perturbs its timings
  --vendor pix|nsight  a deep dive through the GPU vendor's or platform's
                       tool, set up once: pix (with --offscreen, into
                       --vendor-dir, or --dev's profiles asking for it, on
                       D3D12) captures the last frame and replays each draw's
                       hardware counters, summed per render-graph node; nsight
                       (in a window, into --vendor-dir, on NVIDIA's GPUs) runs
                       the program again under Nsight's GPU Trace, --settle-ms
                       after it starts, for 5 frames, with its shader profiler
  --vendor-dir DIR     with --vendor: where its files go
  --vendor-counters LIST
                       with --vendor pix: the comma-separated counters (PIX's
                       names) to replay in place of the summary's own
  --packs DIR          the folder of the cook's packs records are read from
                       before the store and before any job runs
  --cook FILE          cook offscreen: every record the game's content needs,
                       made or read, written into the pack FILE once none is
                       left
  --measure FILE       a measured run offscreen, as the method in FILE says
                       (`cargo xtask measure` writes it): settled by its rule,
                       then timed one frame at a time and with frames in
                       flight, with marks to time them by and a JSON report on
                       standard output; profiling builds add each GPU pass's
                       and CPU span's time
  --shader-hash        print the SHA-256 of the renderer's composed shader
                       modules (each one's name and source) as JSON and exit
  --settings FILE      the options in FILE, a JSON object whose `options` holds
                       them by name ({\"options\": {\"size\": \"1400x900\"}}), the
                       game's too; options after it override its own
  --dev                serve the development protocol over standard input
                       and output, drawing offscreen and starting paused
  --dev-artifacts DIR  with --dev: where captures and profiles go (default
                       the system's temporary folder)
  --help               this text

--profile, --counters and --vendor need a build with the profiling feature,
--dev one with the dev feature. WGPU_BACKEND=dx12|vulkan|metal chooses the
GPU interface; by default the one browsers use on the platform: on Windows
D3D12 where DXC (dxcompiler.dll, which Chrome and Firefox ship) is beside
the program or on the path, else Vulkan.";

pub fn usage() -> String {
    let mut text = format!("{USAGE_HOST}\n");
    for setting in SETTINGS {
        let head = format!("  --{} {}", setting.name, setting.values.join("|"));
        let what = format!("{} (default {})", setting.what, setting.values[0]);
        text.push_str(&option_lines(&head, &what));
    }
    text.push_str(USAGE_TOOLS);
    text
}

fn option_lines(head: &str, what: &str) -> String {
    const COLUMN: usize = 23;
    const WIDTH: usize = 78;
    let mut text = String::new();
    let mut line = if head.len() < COLUMN - 1 {
        format!("{head:COLUMN$}")
    } else {
        text.push_str(head);
        text.push('\n');
        " ".repeat(COLUMN)
    };
    let mut words = 0;
    for word in what.split_whitespace() {
        if words > 0 && line.len() + 1 + word.len() > WIDTH {
            text.push_str(&line);
            text.push('\n');
            line = " ".repeat(COLUMN);
            words = 0;
        }
        if words > 0 {
            line.push(' ');
        }
        line.push_str(word);
        words += 1;
    }
    text.push_str(&line);
    text.push('\n');
    text
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub name: String,
    inline: Option<String>,
}

impl Argument {
    pub fn unknown(&self) -> String {
        format!("unknown option {}", self.name)
    }
}

pub struct Arguments {
    rest: VecDeque<String>,
    included: usize,
}

const MAX_OPTION_FILES: usize = 16;

impl Arguments {
    pub fn new(arguments: impl IntoIterator<Item = String>) -> Self {
        Self {
            rest: arguments.into_iter().collect(),
            included: 0,
        }
    }

    pub fn include(&mut self, argument: &Argument) -> Result<(), String> {
        let path = self.value(argument)?;
        self.included += 1;
        if self.included > MAX_OPTION_FILES {
            return Err(format!(
                "{} {path}: more than {MAX_OPTION_FILES} option files",
                argument.name
            ));
        }
        let failed = |what: String| format!("{} {path}: {what}", argument.name);
        let text = std::fs::read_to_string(&path).map_err(|error| failed(error.to_string()))?;
        let file: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| failed(error.to_string()))?;
        let fields = file
            .as_object()
            .ok_or_else(|| failed("an option file is a JSON object".into()))?;
        if let Some(field) = fields
            .keys()
            .find(|field| !["note", "options"].contains(&field.as_str()))
        {
            return Err(failed(format!(
                "{field} is not a field; option files hold `options` and a `note`"
            )));
        }
        let options = fields
            .get("options")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| failed("`options` is an object of options by name".into()))?;
        let mut expanded = Vec::with_capacity(options.len() * 2);
        for (name, value) in options {
            let value = match value {
                serde_json::Value::String(text) => text.clone(),
                serde_json::Value::Bool(true) => "on".into(),
                serde_json::Value::Bool(false) => "off".into(),
                serde_json::Value::Number(number) => number.to_string(),
                serde_json::Value::Null => {
                    return Err(failed(format!("option {name} has no value")));
                }
                json @ (serde_json::Value::Array(_) | serde_json::Value::Object(_)) => {
                    json.to_string()
                }
            };
            expanded.extend([format!("--{name}"), value]);
        }
        for argument in expanded.into_iter().rev() {
            self.rest.push_front(argument);
        }
        Ok(())
    }

    pub fn value(&mut self, argument: &Argument) -> Result<String, String> {
        argument
            .inline
            .clone()
            .or_else(|| self.rest.pop_front())
            .ok_or_else(|| format!("{} takes a value", argument.name))
    }

    pub fn switch(&mut self, argument: &Argument) -> Result<bool, String> {
        match self.value(argument)?.as_str() {
            "on" => Ok(true),
            "off" => Ok(false),
            value => Err(format!("{} is on or off, not {value}", argument.name)),
        }
    }

    pub fn count(&mut self, argument: &Argument) -> Result<u32, String> {
        let value = self.value(argument)?;
        value
            .parse()
            .ok()
            .filter(|&count: &u32| count > 0)
            .ok_or_else(|| format!("{} takes a count, not {value}", argument.name))
    }

    pub fn flag(&self, argument: &Argument) -> Result<(), String> {
        match argument.inline {
            Some(_) => Err(format!("{} takes no value", argument.name)),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    Pix,
    Nsight,
}

#[derive(Clone, Debug)]
pub struct HostOptions {
    pub size: Option<(u32, u32)>,
    pub gpu: GpuPreference,
    pub jobs: JobOptions,
    pub vsync: bool,
    pub settings: HostSettings,
    pub frames: Option<u32>,
    pub settle_ms: u64,
    pub offscreen: Option<u32>,
    pub capture: Option<PathBuf>,
    pub gpu_capture: Option<PathBuf>,
    pub profile: Option<PathBuf>,
    pub counters: Option<String>,
    pub vendor: Option<Vendor>,
    pub vendor_dir: Option<PathBuf>,
    pub vendor_counters: Vec<String>,
    pub dev: bool,
    pub dev_artifacts: Option<PathBuf>,
    pub measure: Option<Method>,
    pub shader_hash: bool,
    pub cook: Option<PathBuf>,
    pub workers_given: bool,
}

impl Default for HostOptions {
    fn default() -> Self {
        Self {
            size: None,
            gpu: GpuPreference::HighPerformance,
            jobs: JobOptions::default(),
            vsync: true,
            settings: HostSettings::default(),
            frames: None,
            settle_ms: 20_000,
            offscreen: None,
            capture: None,
            gpu_capture: None,
            profile: None,
            counters: None,
            vendor: None,
            vendor_dir: None,
            vendor_counters: Vec::new(),
            dev: false,
            dev_artifacts: None,
            measure: None,
            shader_hash: false,
            cook: None,
            workers_given: false,
        }
    }
}

pub const PARAMETERS: &[(&str, &str)] = &[
    ("size", "WIDTHxHEIGHT of the offscreen texture"),
    ("gpu", "discrete or integrated"),
    ("workers", "job worker threads, or off"),
    ("jobcache", "off, verify, or the kept outputs' MiB"),
    (
        "counters",
        "on, or counter names: each profile's per-pass hardware counters (profiling builds; Vulkan)",
    ),
    (
        "vendor",
        "pix: profiles may ask for the last frame's per-draw counters replayed through PIX (profiling builds; D3D12)",
    ),
];

impl HostOptions {
    pub fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
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

    pub fn take(&mut self, argument: &Argument, arguments: &mut Arguments) -> Result<bool, String> {
        let name = argument.name.as_str();
        match name {
            "--size" => {
                let size = arguments.value(argument)?;
                let parsed = size
                    .split_once('x')
                    .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
                    .filter(|&(width, height): &(u32, u32)| width > 0 && height > 0);
                self.size =
                    Some(parsed.ok_or_else(|| format!("--size is WIDTHxHEIGHT, not {size}"))?);
            }
            "--gpu" => {
                self.gpu = match arguments.value(argument)?.as_str() {
                    "discrete" => GpuPreference::HighPerformance,
                    "integrated" => GpuPreference::LowPower,
                    other => return Err(format!("--gpu is discrete or integrated, not {other}")),
                }
            }
            "--workers" => {
                let workers = arguments.value(argument)?;
                self.workers_given = true;
                self.jobs.workers = match workers.as_str() {
                    "off" => 0,
                    count => count
                        .parse()
                        .ok()
                        .filter(|&count: &u32| count <= policy::MAX_WORKERS)
                        .ok_or_else(|| {
                            format!(
                                "--workers is off or 0 to {}, not {workers}",
                                policy::MAX_WORKERS
                            )
                        })? as usize,
                };
            }
            "--jobcache" => {
                let cache = arguments.value(argument)?;
                self.jobs.cache = match cache.as_str() {
                    "off" => CacheOptions::Off,
                    "verify" => CacheOptions::Verify,
                    mebibytes => CacheOptions::Keep {
                        budget: mebibytes
                            .parse::<u64>()
                            .ok()
                            .filter(|&mebibytes| mebibytes <= policy::MAX_JOB_CACHE_MIB)
                            .ok_or_else(|| {
                                format!("--jobcache is off, verify or MiB, not {cache}")
                            })?
                            * 1024
                            * 1024,
                    },
                };
            }
            "--vsync" => self.vsync = arguments.switch(argument)?,
            switch if switch.strip_prefix("--").is_some_and(HostSettings::takes) => {
                let value = arguments.value(argument)?;
                self.settings
                    .set(&switch[2..], &value)
                    .map_err(|error| format!("--{error}"))?;
            }
            "--packs" => self.jobs.packs = Some(arguments.value(argument)?.into()),
            "--cook" => self.cook = Some(arguments.value(argument)?.into()),
            "--frames" => self.frames = Some(arguments.count(argument)?),
            "--settle-ms" => {
                let settle = arguments.value(argument)?;
                self.settle_ms = settle
                    .parse()
                    .map_err(|_| format!("--settle-ms takes milliseconds, not {settle}"))?;
            }
            "--offscreen" => self.offscreen = Some(arguments.count(argument)?),
            "--capture" => self.capture = Some(arguments.value(argument)?.into()),
            "--gpu-capture" => self.gpu_capture = Some(arguments.value(argument)?.into()),
            "--profile" => self.profile = Some(arguments.value(argument)?.into()),
            "--counters" => self.counters = Some(arguments.value(argument)?),
            "--vendor" => {
                self.vendor = Some(match arguments.value(argument)?.as_str() {
                    "pix" => Vendor::Pix,
                    "nsight" => Vendor::Nsight,
                    other => return Err(format!("--vendor is pix or nsight, not {other}")),
                });
            }
            "--vendor-dir" => self.vendor_dir = Some(arguments.value(argument)?.into()),
            "--vendor-counters" => {
                self.vendor_counters = arguments
                    .value(argument)?
                    .split(',')
                    .map(str::to_owned)
                    .collect();
            }
            "--dev" => {
                arguments.flag(argument)?;
                self.dev = true;
            }
            "--dev-artifacts" => self.dev_artifacts = Some(arguments.value(argument)?.into()),
            "--measure" => {
                let path = arguments.value(argument)?;
                let text = std::fs::read_to_string(&path)
                    .map_err(|error| format!("--measure {path}: {error}"))?;
                self.measure = Some(Method::parse(&text)?);
            }
            "--settings" => arguments.include(argument)?,
            "--shader-hash" => {
                arguments.flag(argument)?;
                self.shader_hash = true;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub fn check(&self) -> Result<(), String> {
        if (self.capture.is_some() || self.gpu_capture.is_some() || self.profile.is_some())
            && self.offscreen.is_none()
        {
            return Err("--capture, --gpu-capture and --profile need --offscreen".into());
        }
        let profiling = self.profile.is_some() || self.counters.is_some() || self.vendor.is_some();
        if profiling && !cfg!(feature = "profiling") {
            return Err(
                "--profile, --counters and --vendor need a build with the profiling feature".into(),
            );
        }
        if self.dev && !cfg!(feature = "dev") {
            return Err("--dev needs a build with the dev feature".into());
        }
        let others = self.frames.is_some()
            || self.offscreen.is_some()
            || self.dev
            || self.vendor.is_some()
            || self.counters.is_some();
        if self.cook.is_some() && (others || self.measure.is_some()) {
            return Err(
                "--cook runs alone: without --frames, --offscreen, --measure, --dev, --counters and --vendor"
                    .into(),
            );
        }
        if self.measure.is_some() && others {
            return Err(
                "--measure runs alone: without --frames, --offscreen, --dev, --counters and --vendor"
                    .into(),
            );
        }
        if self.counters.is_some() && self.profile.is_none() && !self.dev {
            return Err("--counters counts profiles: --profile or --dev".into());
        }
        match self.vendor {
            Some(Vendor::Pix) if self.dev => Ok(()),
            Some(Vendor::Pix) if self.offscreen.is_none() => {
                Err("--vendor pix captures an --offscreen frame, or --dev's profiles".into())
            }
            Some(Vendor::Nsight) if self.offscreen.is_some() || self.dev => {
                Err("--vendor nsight traces a window's frames".into())
            }
            Some(_) if self.vendor_dir.is_none() => Err("--vendor needs --vendor-dir".into()),
            _ => Ok(()),
        }
    }

    pub fn run<G: Game, E: Into<engine::Error>>(
        &self,
        title: &str,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> engine::Result<()> {
        if self.shader_hash {
            println!("{}", json!({ "shader_hash": shader_hash() }));
            return Ok(());
        }
        self.start_tools()?;
        #[cfg(feature = "profiling")]
        if let (Some(Vendor::Nsight), Some(folder)) = (self.vendor, &self.vendor_dir) {
            return self.run_nsight(folder);
        }
        if self.dev {
            #[cfg(feature = "dev")]
            return self.session(build)?.run();
        }
        if let Some(pack) = &self.cook {
            return self.run_cook(pack, build);
        }
        if let Some(method) = &self.measure {
            let size = self.size.unwrap_or((1280, 720));
            return crate::measure::run(
                method,
                size,
                self.gpu,
                self.jobs.clone(),
                self.settings,
                build,
            );
        }
        match self.offscreen {
            Some(frames) => self.run_offscreen(frames, build),
            None => {
                let report = crate::run(
                    Window {
                        size: self.size,
                        gpu: self.gpu,
                        jobs: self.jobs.clone(),
                        vsync: self.vsync,
                        settings: self.settings,
                        measure: self.frames.map(|frames| Measure {
                            settle: Duration::from_millis(self.settle_ms),
                            frames,
                        }),
                        ..Window::new(title)
                    },
                    build,
                )?;
                if let Some(report) = report {
                    println!("{:#}", window_report(&report));
                }
                Ok(())
            }
        }
    }

    #[cfg(feature = "dev")]
    pub fn dev_session<G: Game, E: Into<engine::Error>>(
        &self,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> engine::Result<crate::dev::DevSession<G>> {
        self.start_tools()?;
        self.session(build)
    }

    #[cfg(feature = "dev")]
    fn session<G: Game, E: Into<engine::Error>>(
        &self,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> engine::Result<crate::dev::DevSession<G>> {
        let size = self.size.unwrap_or((1280, 720));
        let pix = self.loads_pix().then(Pix::load).flatten();
        let mut game = Offscreen::launch(size, self.gpu, &self.jobs, build)?;
        self.settings.apply(game.host_mut());
        let artifacts = self
            .dev_artifacts
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("engine-captures"));
        Ok(crate::dev::DevSession::new(game, artifacts)
            .pix(pix)
            .parameters(PARAMETERS.iter().copied())
            .parameters(
                SETTINGS
                    .iter()
                    .map(|setting| (setting.name, setting.described())),
            ))
    }

    fn loads_pix(&self) -> bool {
        self.gpu_capture.is_some() || self.vendor == Some(Vendor::Pix)
    }

    fn start_tools(&self) -> engine::Result<()> {
        #[cfg(feature = "profiling")]
        if let Some(counters) = &self.counters {
            use engine_profiler::counters::Selection;
            let selection = match counters.as_str() {
                "on" => Selection::First,
                names => Selection::Named(names.split(',').map(str::to_owned).collect()),
            };
            engine_profiler::counters::enable(selection).map_err(|reason| {
                engine::Error::new(engine::ErrorCode::ProfilingUnavailable, "$", reason)
            })?;
        }
        Ok(())
    }

    fn run_cook<G: Game, E: Into<engine::Error>>(
        &self,
        path: &Path,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> engine::Result<()> {
        let size = self.size.unwrap_or((320, 180));
        let started = Instant::now();
        let mut jobs = self.jobs.clone();
        if !self.workers_given {
            let cores = std::thread::available_parallelism().map_or(1, usize::from);
            jobs.workers = policy::cook_workers(u32::try_from(cores).unwrap_or(u32::MAX)) as usize;
        }
        let mut game = Offscreen::launch_with(size, self.gpu, jobs.pool(true), build)?;
        self.settings.apply(game.host_mut());
        game.host_mut().cook();
        let read = game.jobs_mut().packed();
        let (mut frame, mut settled, mut said) = (0_u32, 0, Instant::now());
        let (mut last, mut moved) = ((u64::MAX, u32::MAX), Instant::now());
        loop {
            game.frame(f64::from(frame) * 1000.0 / 60.0)?;
            frame += 1;
            let left = game
                .host()
                .cooking()
                .map_or(0, engine::host::Cooking::remaining);
            let held = game.host().jobs_held();
            if (left, held) != last {
                (last, moved) = ((left, held), Instant::now());
            }
            settled = if left == 0 && held == 0 {
                settled + 1
            } else {
                0
            };
            if settled >= COOK_SETTLE {
                break;
            }
            if moved.elapsed() > COOK_STALL {
                return Err(engine::Error::new(
                    engine::ErrorCode::RenderFailed,
                    "$",
                    format!(
                        "the cook stalled: {left} records left and {held} jobs held, \
                         unchanged for {} s",
                        COOK_STALL.as_secs()
                    ),
                ));
            }
            if said.elapsed() > Duration::from_secs(5) {
                said = Instant::now();
                let features = game
                    .host()
                    .cooking()
                    .map(|cooking| cooking.features().to_vec());
                eprintln!("Cooking: {left} records left, {held} jobs held: {features:?}");
            }
        }
        let written = game.jobs_mut().write_pack(path).map_err(|reason| {
            host_error(
                &format!("The pack {} was not written", path.display()),
                reason,
            )
        })?;
        let stats: serde_json::Value =
            serde_json::from_str(&game.host().stats()).expect("frame statistics are JSON");
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
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> engine::Result<()> {
        let size = self.size.unwrap_or((1280, 720));
        let pix = self.loads_pix().then(Pix::load).flatten();
        let started = Instant::now();
        let mut game = Offscreen::launch(size, self.gpu, &self.jobs, build)?;
        self.settings.apply(game.host_mut());
        game.prepare(0.0)?;
        let prepared_ms = started.elapsed().as_secs_f64() * 1000.0;
        let record = Record {
            gpu_capture: self.gpu_capture.clone(),
            profile: self.profile.as_ref().map(|folder| ProfileRecord {
                folder: folder.clone(),
                stem: "profile".into(),
                max_events: None,
                cpu_samples: true,
                context: json!({
                    "program": program(),
                    "arguments": std::env::args().skip(1).collect::<Vec<_>>(),
                    "size": [size.0, size.1],
                }),
            }),
            pix_counters: self
                .vendor_dir
                .clone()
                .filter(|_| self.vendor == Some(Vendor::Pix))
                .map(|folder| PixRecord {
                    folder,
                    counters: self.vendor_counters.clone(),
                }),
        };
        let at = |frame: u32| f64::from(frame) * 1000.0 / 60.0;
        let recorded = game.record(Frames::Count(frames), at, &record, pix.as_ref())?;
        if let Some(error) = recorded.failed {
            return Err(error);
        }
        if let Some(path) = &self.capture {
            game.save_png(path)?;
        }
        let stats: serde_json::Value =
            serde_json::from_str(&game.host().stats()).expect("frame statistics are JSON");
        let mismatched: Vec<serde_json::Value> = game
            .jobs_mut()
            .mismatched()
            .iter()
            .map(|(kind, key)| json!({ "kind": kind, "key": key }))
            .collect();
        let adapter = game.host().adapter();
        let report = json!({
            "size": [size.0, size.1],
            "frames": recorded.frames,
            "prepare_ms": prepared_ms,
            "frame_ms": recorded.elapsed_ms / f64::from(recorded.frames.max(1)),
            "cpu_ms": distribution(&recorded.cpu_ms),
            "capture": self.capture.as_ref().map(|path| path.display().to_string()),
            "profile": recorded.profile,
            "vendor": recorded.pix_counters.map(|summary| json!({
                "summary": record.pix_counters.as_ref().map(|dive| dive.folder.join("pix.summary.json").display().to_string()),
                "passes": summary["passes"].as_array().map(|passes| passes.iter().take(8).cloned().collect::<Vec<_>>()),
            })),
            "gpu_capture": recorded.capture.as_ref().map(|path| json!({
                "path": path.display().to_string(),
                "by": recorded.captured,
                "adapter": adapter.name,
                "backend": adapter.backend,
            })),
            "main_stack_peak_bytes": engine_profiler::stack::peak(),
            "mismatched": mismatched,
            "stats": stats,
        });
        println!("{report:#}");
        Ok(())
    }

    #[cfg(feature = "profiling")]
    fn run_nsight(&self, folder: &std::path::Path) -> engine::Result<()> {
        let mut arguments = Vec::new();
        let mut given = std::env::args().skip(1);
        while let Some(argument) = given.next() {
            let name = argument.split('=').next().unwrap_or_default();
            if ["--vendor", "--vendor-dir", "--vendor-counters"].contains(&name) {
                if !argument.contains('=') {
                    given.next();
                }
                continue;
            }
            arguments.push(argument);
        }
        if self.frames.is_none() {
            arguments.extend(["--frames".to_owned(), "120".to_owned()]);
        }
        let trace = engine_profiler::vendor::nsight_trace(&arguments, self.settle_ms, 5, folder)
            .map_err(|reason| {
                engine::Error::new(engine::ErrorCode::ProfilingUnavailable, "$", reason)
            })?;
        println!("{trace:#}");
        Ok(())
    }
}

const COOK_SETTLE: u32 = 8;

const COOK_STALL: Duration = Duration::from_secs(600);

fn shader_hash() -> String {
    use sha2::Digest;
    let mut hash = sha2::Sha256::new();
    for (name, source) in engine::SHADER_MODULES {
        for part in [name, source] {
            hash.update(part.as_bytes());
            hash.update([0]);
        }
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn program() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .unwrap_or_default()
}

fn distribution(values: &[f64]) -> serde_json::Value {
    json!(engine::host::Distribution::of(values))
}

fn window_report(report: &WindowReport) -> serde_json::Value {
    let stats: serde_json::Value =
        serde_json::from_str(&report.stats).expect("frame statistics are JSON");
    json!({
        "size": [report.size.0, report.size.1],
        "window": [report.window.0, report.window.1],
        "frames": report.frames,
        "fps": f64::from(report.frames) * 1000.0 / report.elapsed_ms,
        "frame_ms": report.elapsed_ms / f64::from(report.frames),
        "cpu_ms": distribution(&report.cpu_ms),
        "adapter": {"name": report.adapter.name, "backend": report.adapter.backend},
        "main_stack_peak_bytes": engine_profiler::stack::peak(),
        "stats": stats,
    })
}
