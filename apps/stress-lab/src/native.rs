//! The Stress Lab natively: a workload or an authored scene in a window, or
//! drawn offscreen for a number of frames, with the page's parameters as
//! options (`--preset metro`, `--source PATH`...) and the files the page
//! fetches read from disk.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use engine::{Engine, Error, ErrorCode, Feature};
use engine_native::{CacheOptions, GpuPreference, JobOptions, Offscreen, Pix, Window, files};

use crate::{DENSE_FILES, HIGHLANDS_FILES, ROCK_FILES, StressConfig, StressFile, StressLab};
use crate::{StressScene, preset, presets, set_view};

const USAGE: &str = "\
Usage: stress-lab-native [OPTIONS]

The page's parameters:
  --preset NAME        start from that workload preset (default visual)
  --config JSON        workload fields over the preset's
  --source PATH        run that authored scene file in place of a workload
  --without LIST       with --source: comma-separated features the engine
                       leaves out (terrain, navigation...)
  --look NAME          lighting preset over the scene's own
  --effects NAME       all, none, or comma-separated no-<effect> exclusions
  --visibility on|off  shade virtual geometry through the visibility buffer
  --lod on|off         off: draw every instance as its mesh
  --impostors on|off   off: draw no impostors

Native options:
  --view NAME          camera view
  --size WxH           window or offscreen size in physical pixels
  --vsync on|off       whether window frames wait for the display's refresh
                       (default on); off measures as browsers' frame rates do
  --gpu discrete|integrated
                       the GPU asked for where there are two (default discrete)
  --workers N|off      job worker threads (default half the logical
                       processors, at most 8); off runs jobs in the engine
  --jobcache off|verify|MiB
                       keep job outputs beside the program (default 1024
                       MiB); verify runs every job anyway and compares
  --assets DIR         the folder holding stress/ (default web/public)
  --offscreen FRAMES   draw FRAMES frames at 60 Hz offscreen and print their
                       statistics as JSON, without a window
  --capture PATH       with --offscreen: the last frame as a PNG file
  --gpu-capture PATH   with --offscreen: the last frame's GPU work, frame-exact:
                       a PIX capture (.wpix) where PIX is installed and the
                       GPU interface is D3D12; else a RenderDoc capture
                       (PATH_capture.rdc) under `renderdoccmd capture -w`;
                       else Xcode's, with MTL_CAPTURE_ENABLED=1
  --dev                serve the development protocol over standard input
                       and output, drawing offscreen and starting paused
                       (builds with --features dev)
  --dev-artifacts DIR  with --dev: where captures go (default the system's
                       temporary folder)
  --help               this text

WGPU_BACKEND=dx12|vulkan|metal chooses the GPU interface; by default the one
browsers use on the platform: on Windows D3D12 where DXC (dxcompiler.dll,
which Chrome and Firefox ship) is beside the program or on the path, else
Vulkan.";

/// The options development sessions name as their parameters, with what
/// each does: the page's parameters and the native ones.
#[cfg(feature = "dev")]
const PARAMETERS: &[(&str, &str)] = &[
    ("preset", "start from that workload preset"),
    ("config", "workload fields over the preset's (JSON)"),
    (
        "source",
        "run that authored scene file in place of a workload",
    ),
    (
        "without",
        "with source: comma-separated features the engine leaves out",
    ),
    ("look", "lighting preset over the scene's own"),
    (
        "effects",
        "all, none, or comma-separated no-<effect> exclusions",
    ),
    (
        "visibility",
        "on: shade virtual geometry through the visibility buffer",
    ),
    ("lod", "off: draw every instance as its mesh"),
    ("impostors", "off: draw no impostors"),
    ("view", "camera view"),
    ("size", "WIDTHxHEIGHT of the offscreen texture"),
    ("gpu", "discrete or integrated"),
    ("workers", "job worker threads, or off"),
    ("jobcache", "off, verify, or the kept outputs' MiB"),
    ("assets", "the folder holding stress/"),
];

/// What the command line asks for.
struct Options {
    preset: String,
    config: Option<String>,
    source: Option<PathBuf>,
    without: String,
    look: String,
    effects: String,
    visibility: bool,
    lod: bool,
    impostors: bool,
    view: Option<String>,
    size: Option<(u32, u32)>,
    gpu: GpuPreference,
    jobs: JobOptions,
    vsync: bool,
    assets: PathBuf,
    offscreen: Option<u32>,
    capture: Option<PathBuf>,
    gpu_capture: Option<PathBuf>,
    dev: bool,
    dev_artifacts: Option<PathBuf>,
}

/// Runs the Stress Lab as `arguments` ask: the program's `main`.
pub fn main(arguments: impl Iterator<Item = String>) -> ExitCode {
    let arguments: Vec<String> = arguments.collect();
    if arguments.iter().any(|argument| argument == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match Options::parse(arguments) {
        Ok(options) => engine_native::exit(options.run()),
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

impl Options {
    fn parse(arguments: Vec<String>) -> Result<Self, String> {
        let mut options = Self {
            preset: "visual".into(),
            config: None,
            source: None,
            without: String::new(),
            look: "scene".into(),
            effects: "all".into(),
            visibility: false,
            lod: true,
            impostors: true,
            view: None,
            size: None,
            gpu: GpuPreference::HighPerformance,
            jobs: JobOptions::default(),
            vsync: true,
            // The page's public folder, beside this crate.
            assets: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/public"),
            offscreen: None,
            capture: None,
            gpu_capture: None,
            dev: false,
            dev_artifacts: None,
        };
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            let (name, inline) = match argument.split_once('=') {
                Some((name, value)) => (name.to_owned(), Some(value.to_owned())),
                None => (argument, None),
            };
            let mut value = || {
                inline
                    .clone()
                    .or_else(|| arguments.next())
                    .ok_or_else(|| format!("{name} takes a value"))
            };
            let switch = |value: String| match value.as_str() {
                "on" => Ok(true),
                "off" => Ok(false),
                _ => Err(format!("{name} is on or off, not {value}")),
            };
            match name.as_str() {
                "--preset" => options.preset = value()?,
                "--config" => options.config = Some(value()?),
                "--source" => options.source = Some(value()?.into()),
                "--without" => options.without = value()?,
                "--look" => options.look = value()?,
                "--effects" => options.effects = value()?,
                "--visibility" => options.visibility = switch(value()?)?,
                "--lod" => options.lod = switch(value()?)?,
                "--impostors" => options.impostors = switch(value()?)?,
                "--view" => options.view = Some(value()?),
                "--size" => {
                    let size = value()?;
                    let parsed = size
                        .split_once('x')
                        .and_then(|(width, height)| {
                            Some((width.parse().ok()?, height.parse().ok()?))
                        })
                        .filter(|&(width, height): &(u32, u32)| width > 0 && height > 0);
                    options.size =
                        Some(parsed.ok_or_else(|| format!("--size is WIDTHxHEIGHT, not {size}"))?);
                }
                "--gpu" => {
                    options.gpu = match value()?.as_str() {
                        "discrete" => GpuPreference::HighPerformance,
                        "integrated" => GpuPreference::LowPower,
                        other => {
                            return Err(format!("--gpu is discrete or integrated, not {other}"));
                        }
                    }
                }
                "--workers" => {
                    let workers = value()?;
                    options.jobs.workers = match workers.as_str() {
                        "off" => 0,
                        count => count
                            .parse()
                            .ok()
                            .filter(|&count: &usize| count <= 32)
                            .ok_or_else(|| format!("--workers is off or 0 to 32, not {workers}"))?,
                    };
                }
                "--jobcache" => {
                    let cache = value()?;
                    options.jobs.cache = match cache.as_str() {
                        "off" => CacheOptions::Off,
                        "verify" => CacheOptions::Verify,
                        mebibytes => CacheOptions::Keep {
                            budget: mebibytes
                                .parse::<u64>()
                                .ok()
                                .filter(|&mebibytes| mebibytes < 10_000_000)
                                .ok_or_else(|| {
                                    format!("--jobcache is off, verify or MiB, not {cache}")
                                })?
                                * 1024
                                * 1024,
                        },
                    };
                }
                "--vsync" => options.vsync = switch(value()?)?,
                "--assets" => options.assets = value()?.into(),
                "--offscreen" => {
                    let frames = value()?;
                    options.offscreen = Some(
                        frames
                            .parse()
                            .ok()
                            .filter(|&frames: &u32| frames > 0)
                            .ok_or_else(|| {
                                format!("--offscreen takes a frame count, not {frames}")
                            })?,
                    );
                }
                "--capture" => options.capture = Some(value()?.into()),
                "--gpu-capture" => options.gpu_capture = Some(value()?.into()),
                "--dev" => {
                    if inline.is_some() {
                        return Err("--dev takes no value".into());
                    }
                    options.dev = true;
                }
                "--dev-artifacts" => options.dev_artifacts = Some(value()?.into()),
                _ => return Err(format!("unknown option {name}")),
            }
        }
        if (options.capture.is_some() || options.gpu_capture.is_some())
            && options.offscreen.is_none()
        {
            return Err("--capture and --gpu-capture need --offscreen".into());
        }
        Ok(options)
    }

    fn run(&self) -> engine::Result<()> {
        let build = || self.build();
        if self.dev {
            return self.run_dev(build);
        }
        match self.offscreen {
            None => engine_native::run(
                Window {
                    size: self.size,
                    gpu: self.gpu,
                    jobs: self.jobs,
                    vsync: self.vsync,
                    ..Window::new("Stress Lab")
                },
                build,
            ),
            Some(frames) => self.run_offscreen(frames, build),
        }
    }

    /// The engine and Stress Lab the options make, with the page's choices
    /// applied as the page applies them.
    fn build(&self) -> engine::Result<(Engine, StressLab)> {
        if let Some(path) = &self.source {
            let source = std::fs::read_to_string(path).map_err(|error| {
                Error::new(
                    ErrorCode::MissingFile,
                    "$",
                    format!("cannot read {}: {error}", path.display()),
                )
            })?;
            let files = files::scene_files(path, &source)?;
            let without = self
                .without
                .split(',')
                .filter(|name| !name.is_empty())
                .map(|name| {
                    Feature::from_name(name)
                        .ok_or_else(|| Error::from(format!("no feature is named {name}")))
                })
                .collect::<engine::Result<Vec<_>>>()?;
            return Ok(StressLab::authored(&source, &files, &without)?);
        }
        let config = self.config()?;
        let files = stress_files(&self.assets, config.scene)?;
        let (mut engine, mut lab) = StressLab::workload(config, files)?;
        lab.set_visibility_buffer(&mut engine, self.visibility);
        lab.set_level_of_detail(&mut engine, self.lod);
        lab.set_impostors(&mut engine, self.impostors);
        lab.set_look(&mut engine, &self.look)?;
        lab.set_effects(&mut engine, &self.effects)?;
        if let Some(view) = &self.view {
            set_view(&mut engine, view)?;
        }
        Ok((engine, lab))
    }

    /// Serves the development protocol over standard input and output.
    #[cfg(feature = "dev")]
    fn run_dev(
        &self,
        build: impl FnOnce() -> engine::Result<(Engine, StressLab)>,
    ) -> engine::Result<()> {
        let size = self.size.unwrap_or((1280, 720));
        let game = Offscreen::launch(size, self.gpu, self.jobs, build)?;
        let artifacts = self
            .dev_artifacts
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("engine-captures"));
        let assets = self.assets.clone();
        let mut session = engine_native::dev::DevSession::new(game, artifacts)
            .parameters(PARAMETERS)
            .control(
                "workload",
                "Runs preset {preset: NAME} or configuration {config: {...}} (fields over the current one) in place of the workload",
                move |host, args| {
                    let (config, title) = match args.get("preset") {
                        Some(name) => {
                            let name = name.as_str().unwrap_or_default();
                            let preset = preset(name).ok_or_else(|| {
                                let names: Vec<&str> =
                                    presets().iter().map(|preset| preset.value).collect();
                                format!("no preset {name}; presets are {}", names.join(", "))
                            })?;
                            (preset.config, preset.title)
                        }
                        None => {
                            let current = host
                                .engine()
                                .app
                                .world()
                                .get_resource::<StressConfig>()
                                .cloned()
                                .unwrap_or_default();
                            let mut config = serde_json::to_value(current)
                                .expect("configurations serialize");
                            if let (Some(fields), Some(serde_json::Value::Object(over))) =
                                (config.as_object_mut(), args.get("config"))
                            {
                                fields.extend(over.clone());
                            }
                            let config = StressConfig::from_json(&config.to_string())
                                .map_err(|error| error.to_string())?;
                            (config, "AGENT WORKLOAD")
                        }
                    };
                    let files =
                        stress_files(&assets, config.scene).map_err(|error| error.to_string())?;
                    let engine = host.game_mut().configure(config.clone(), files)?;
                    host.replace_engine(engine);
                    let (engine, lab) = host.parts_mut();
                    lab.configured(engine)?;
                    Ok(serde_json::json!({"config": config, "title": title}))
                },
            )
            .control(
                "look",
                &format!(
                    "Lighting preset {{name}}: {}",
                    crate::catalog().looks.iter().map(|look| look.value.as_str()).collect::<Vec<_>>().join(", ")
                ),
                |host, args| {
                    let name = args.get("name").and_then(|name| name.as_str()).unwrap_or_default();
                    let (engine, lab) = host.parts_mut();
                    lab.set_look(engine, name)?;
                    Ok(serde_json::json!({"look": name}))
                },
            )
            .control(
                "effects",
                &format!(
                    "Rendering effects {{name}}: all, none or comma-separated {}",
                    crate::catalog()
                        .effects
                        .iter()
                        .filter(|effect| effect.value.starts_with("no-"))
                        .map(|effect| effect.value.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                |host, args| {
                    let name = args.get("name").and_then(|name| name.as_str()).unwrap_or_default();
                    let (engine, lab) = host.parts_mut();
                    lab.set_effects(engine, name)?;
                    Ok(serde_json::json!({"effects": name}))
                },
            )
            .control(
                "view",
                &format!(
                    "Camera view {{name}}: {}",
                    crate::catalog().views.iter().map(|view| view.value.as_str()).collect::<Vec<_>>().join(", ")
                ),
                |host, args| {
                    let name = args.get("name").and_then(|name| name.as_str()).unwrap_or_default();
                    set_view(host.engine_mut(), name)?;
                    Ok(serde_json::json!({"view": name}))
                },
            );
        if let Some(path) = &self.source {
            session = session.source(path);
        }
        session.run()
    }

    #[cfg(not(feature = "dev"))]
    fn run_dev(
        &self,
        _: impl FnOnce() -> engine::Result<(Engine, StressLab)>,
    ) -> engine::Result<()> {
        Err(Error::new(
            ErrorCode::UnsupportedCommand,
            "$",
            "development sessions need a build with --features dev",
        ))
    }

    /// The preset's configuration under the `--config` fields.
    fn config(&self) -> engine::Result<StressConfig> {
        let preset = preset(&self.preset).ok_or_else(|| {
            let names: Vec<&str> = presets().iter().map(|preset| preset.value).collect();
            Error::from(format!(
                "no preset {}; presets are {}",
                self.preset,
                names.join(", ")
            ))
        })?;
        let Some(fields) = &self.config else {
            return Ok(preset.config);
        };
        let mut config = serde_json::to_value(&preset.config).expect("configurations serialize");
        let fields: serde_json::Value = serde_json::from_str(fields)
            .map_err(|error| Error::new(ErrorCode::InvalidJson, "$", error.to_string()))?;
        let (Some(config_fields), serde_json::Value::Object(fields)) =
            (config.as_object_mut(), fields)
        else {
            return Err(Error::new(
                ErrorCode::InvalidType,
                "$",
                "--config is a JSON object of workload fields",
            ));
        };
        config_fields.extend(fields);
        Ok(StressConfig::from_json(&config.to_string())?)
    }

    /// Draws `frames` frames offscreen at 60 Hz after preparing, then prints
    /// their CPU times and the last frame's statistics.
    fn run_offscreen(
        &self,
        frames: u32,
        build: impl FnOnce() -> engine::Result<(Engine, StressLab)>,
    ) -> engine::Result<()> {
        let size = self.size.unwrap_or((1280, 720));
        // PIX's capturer must be in the program before its D3D12 device.
        let pix = self.gpu_capture.as_ref().and_then(|_| Pix::load());
        let started = Instant::now();
        let mut game = Offscreen::launch(size, self.gpu, self.jobs, build)?;
        game.prepare(0.0)?;
        let prepared_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut cpu_ms = Vec::with_capacity(frames as usize);
        let drawing = Instant::now();
        let mut captured = None;
        for frame in 0..frames {
            let timestamp = f64::from(frame) * 1000.0 / 60.0;
            match &self.gpu_capture {
                Some(path) if frame + 1 == frames => {
                    captured = Some(game.capture_frame(timestamp, pix.as_ref(), path)?);
                }
                _ => game.frame(timestamp)?,
            }
            let stats = game.host().frame_stats();
            cpu_ms.push(stats.simulation_ms + stats.extraction_ms + stats.render_ms);
        }
        let frame_ms = drawing.elapsed().as_secs_f64() * 1000.0 / f64::from(frames);
        if let Some(path) = &self.capture {
            game.save_png(path)?;
        }
        cpu_ms.sort_by(f64::total_cmp);
        let at = |share: f64| cpu_ms[((cpu_ms.len() - 1) as f64 * share).round() as usize];
        let stats: serde_json::Value =
            serde_json::from_str(&game.host().stats()).expect("frame statistics are JSON");
        let report = serde_json::json!({
            "size": [size.0, size.1],
            "frames": frames,
            "prepare_ms": prepared_ms,
            "frame_ms": frame_ms,
            "cpu_ms": {"median": at(0.5), "p95": at(0.95)},
            "capture": self.capture.as_ref().map(|path| path.display().to_string()),
            "gpu_capture": self.gpu_capture.as_ref().map(|path| serde_json::json!({
                "path": path.display().to_string(),
                "by": captured,
                "adapter": game.host().adapter().name,
                "backend": game.host().adapter().backend,
            })),
            "stats": stats,
        });
        println!("{report:#}");
        Ok(())
    }
}

/// The files workload `scene` reads from the folder `assets`, as the page
/// fetches them: the dense scene's virtual mesh prefixes, streamed from
/// their files, and the highlands' terrain material maps, without which
/// its ground shows plain colors, and its optional scanned rock maps.
fn stress_files(assets: &Path, scene: StressScene) -> engine::Result<HashMap<String, StressFile>> {
    let mut found = HashMap::new();
    match scene {
        StressScene::Dense => {
            for name in DENSE_FILES {
                let path = assets.join("stress/dense").join(name);
                let bytes = files::read_prefix(&path, 128, engine::VirtualMesh::prefix_bytes)
                    .map_err(|error| {
                        Error::new(
                            error.code,
                            error.path,
                            format!(
                                "{} Build the files with `cargo run -p engine_vgbuild --release --bin dense_assets`.",
                                error.message
                            ),
                        )
                    })?;
                let url = Some(path.to_string_lossy().into_owned());
                found.insert(
                    name.to_owned(),
                    StressFile {
                        bytes,
                        url,
                        size: None,
                    },
                );
            }
        }
        StressScene::Highlands => {
            let image = |path: &Path| -> engine::Result<Option<files::Image>> {
                let bytes = std::fs::read(path).map_err(|error| {
                    Error::new(
                        ErrorCode::MissingFile,
                        "$",
                        format!("cannot read {}: {error}", path.display()),
                    )
                })?;
                files::read_rgba8_ktx2(&path.display().to_string(), &bytes)
            };
            let mut add = |key: String, image: files::Image| {
                found.insert(
                    key,
                    StressFile {
                        bytes: image.pixels,
                        url: None,
                        size: Some([image.width, image.height]),
                    },
                );
            };
            let materials = HIGHLANDS_FILES.iter().try_for_each(|name| {
                let path = assets.join("stress/terrain").join(name);
                let decoded = image(&path)?.ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidTexture,
                        "$",
                        format!("{} is not an uncompressed RGBA8 KTX2 image", path.display()),
                    )
                })?;
                add((*name).to_owned(), decoded);
                Ok::<_, Error>(())
            });
            if let Err(error) = materials {
                eprintln!(
                    "The ground shows plain colors: {}. Build its materials with \
                     `cargo run -p engine_assetc --release --bin terrain_materials`.",
                    error.message
                );
                return Ok(HashMap::new());
            }
            // Scanned rock maps, kept locally and optional: without them
            // rocks wear the terrain rock's maps.
            for name in ROCK_FILES {
                if let Ok(Some(decoded)) = image(&assets.join("stress/rocks").join(name)) {
                    add(format!("rocks/{name}"), decoded);
                }
            }
        }
        _ => {}
    }
    Ok(found)
}
