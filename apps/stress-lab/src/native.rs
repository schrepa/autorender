//! The Stress Lab natively: a workload or an authored scene in a window, or
//! drawn offscreen for a number of frames, with the page's parameters as
//! options (`--preset metro`, `--source PATH`...) beside the native host's
//! (`engine_native::HostOptions`), and the files the page fetches read from
//! disk. Measurements run a bench scene (`--scene bench/scenes/NAME.json`)
//! with a settings bundle (`--settings bench/settings/NAME.json`): option
//! files, which hold these options by name.

use engine::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use engine::{Engine, Error, ErrorCode, Feature};
use engine_native::{Arguments, HostOptions, files};

use crate::{StressConfig, StressFile, StressLab};
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
  --basin-fog on|off   optional terrain basin fog (default off)
  --view NAME          camera view
  --assets DIR         the folder holding stress/ (default web/public)
  --rocks on|off       off: the highlands' rocks wear the terrain rock's maps,
                       not the scanned ones stress/rocks may hold (default on)
  --scene FILE         a bench scene: the options in FILE (its preset,
                       configuration and view), as --settings reads them";

/// The options development sessions name as their parameters, with what
/// each does: the page's and the switches' (the host names its own).
#[cfg(feature = "dev")]
fn parameters() -> Vec<(&'static str, &'static str)> {
    let mut parameters = vec![
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
        ("view", "camera view"),
        ("assets", "the folder holding stress/"),
        ("rocks", "off: no scanned rock maps"),
    ];
    parameters.extend(
        crate::switches()
            .into_iter()
            .map(|switch| (switch.name, switch.what)),
    );
    parameters
}

/// What the command line asks for: the page's parameters, and the host's
/// options.
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
    basin_fog: Option<bool>,
    view: Option<String>,
    assets: PathBuf,
    rocks: bool,
    host: HostOptions,
}

/// Runs the Stress Lab as `arguments` ask: the program's `main`.
pub fn main(arguments: impl Iterator<Item = String>) -> ExitCode {
    let arguments: Vec<String> = arguments.collect();
    if arguments.iter().any(|argument| argument == "--help") {
        println!("{USAGE}\n\n{}", engine_native::usage());
        return ExitCode::SUCCESS;
    }
    match Options::parse(arguments) {
        Ok(options) => engine_native::exit(options.run()),
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}\n\n{}", engine_native::usage());
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
            basin_fog: None,
            view: None,
            // The page's public folder, beside this crate.
            assets: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/public"),
            rocks: true,
            host: HostOptions::default(),
        };
        let mut arguments = Arguments::new(arguments);
        while let Some(argument) = arguments.next() {
            if options.host.take(&argument, &mut arguments)? {
                continue;
            }
            match argument.name.as_str() {
                "--preset" => options.preset = arguments.value(&argument)?,
                "--config" => options.config = Some(arguments.value(&argument)?),
                "--source" => options.source = Some(arguments.value(&argument)?.into()),
                "--without" => options.without = arguments.value(&argument)?,
                "--look" => options.look = arguments.value(&argument)?,
                "--effects" => options.effects = arguments.value(&argument)?,
                "--visibility" => options.visibility = arguments.switch(&argument)?,
                "--lod" => options.lod = arguments.switch(&argument)?,
                "--impostors" => options.impostors = arguments.switch(&argument)?,
                "--basin-fog" => options.basin_fog = Some(arguments.switch(&argument)?),
                "--view" => options.view = Some(arguments.value(&argument)?),
                "--assets" => options.assets = arguments.value(&argument)?.into(),
                "--rocks" => options.rocks = arguments.switch(&argument)?,
                "--scene" => arguments.include(&argument)?,
                _ => return Err(argument.unknown()),
            }
        }
        options.host.check()?;
        Ok(options)
    }

    fn run(&self) -> engine::Result<()> {
        let build = || self.build();
        if self.host.dev {
            return self.run_dev(build);
        }
        self.host.run("Stress Lab", build)
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
                    Feature::from_name(name).ok_or_else(|| {
                        Error::new(
                            ErrorCode::InvalidArguments,
                            "$.without",
                            format!("no feature is named {name}"),
                        )
                    })
                })
                .collect::<engine::Result<Vec<_>>>()?;
            let (mut engine, mut lab) = StressLab::authored(&source, &files, &without)?;
            if let Some(enabled) = self.basin_fog {
                lab.set_basin_fog(&mut engine, enabled);
            }
            return Ok((engine, lab));
        }
        let config = self.config()?;
        let files = stress_files(&self.assets, config.scene, self.rocks)?;
        let (mut engine, mut lab) = StressLab::workload(config, files)?;
        lab.set_visibility_buffer(&mut engine, self.visibility);
        lab.set_level_of_detail(&mut engine, self.lod);
        lab.set_impostors(&mut engine, self.impostors);
        if let Some(enabled) = self.basin_fog {
            lab.set_basin_fog(&mut engine, enabled);
        }
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
        let (assets, rocks) = (self.assets.clone(), self.rocks);
        let mut session = self
            .host
            .dev_session(build)?
            .parameters(parameters())
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
                            let config = StressConfig::from_json(&config.to_string()).map_err(|error| error.to_string())?;
                            (config, "AGENT WORKLOAD")
                        }
                    };
                    let files = stress_files(&assets, config.scene, rocks).map_err(|error| error.to_string())?;
                    let engine = host.game_mut().configure(config.clone(), files).map_err(|error| error.to_string())?;
                    host.replace_engine(engine);
                    let (engine, lab) = host.parts_mut();
                    lab.configured(engine).map_err(|error| error.to_string())?;
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
                    lab.set_look(engine, name).map_err(|error| error.to_string())?;
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
                    lab.set_effects(engine, name).map_err(|error| error.to_string())?;
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
                    set_view(host.engine_mut(), name).map_err(|error| error.to_string())?;
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
            Error::new(
                ErrorCode::InvalidArguments,
                "$.preset",
                format!(
                    "no preset {}; presets are {}",
                    self.preset,
                    names.join(", ")
                ),
            )
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
        StressConfig::from_json(&config.to_string())
    }
}

/// The files workload `scene` reads from the folder `assets`, as the page
/// fetches them (its world's), with `rocks` the highlands' optional
/// scanned rock maps.
pub fn stress_files(
    assets: &Path,
    scene: StressScene,
    rocks: bool,
) -> engine::Result<HashMap<String, StressFile>> {
    scene.world().files(assets, rocks)
}
