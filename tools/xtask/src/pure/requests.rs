//! The requests corpus: development requests, controls, inspector
//! registrations and scene saves that fail, each with its rule's error
//! code at its path, through the development tools as hosts run them, on
//! an engine with every feature and the scenes corpus's components
//! document. Their messages are observed too, serde's English among them,
//! which the codes of malformed requests and sources are read from. Every
//! code of the catalog is reached by a refusal here or in the scenes
//! corpus, or [`UNREACHED`] says why none can be without a GPU, a host or
//! a browser.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use engine::{Engine, EnginePlugins, ErrorCode};
use engine_dev::{
    BrowserControl, Context, Control, DevError, DevTools, InspectorDescriptor, MAX_INSPECTORS,
    MAX_REQUEST_BYTES, MAX_SCAN_ENTITIES, PROTOCOL_VERSION,
};
use engine_scene::{SceneDocument, SceneInstance};
use serde_json::{Value, json};

use super::Observations;

/// The corpus's refusals.
const FIXTURES: &str = "tools/xtask/pure/requests.json";
/// The scenes corpus, whose refusals count too and whose components
/// document requests select entities in.
const SCENES: &str = "tools/xtask/pure/scenes.json";
/// Where scene saves' refusals write their files.
const AUTHORING: &str = "target/pure/requests";

/// Who answers codes the engine never does.
const BRIDGE: &str = "the agent bridge and its sessions answer it (web/scripts/agent*.mjs)";
const WATCH: &str = "the watch tools answer it (web/scripts/iteration*.mjs and the build worker)";
const PAGE: &str = "the page answers it (web/src/engine/dev.ts and the benchmark it runs)";
const NATIVE_PROFILE: &str = "native profile captures answer it (crates/engine_native)";
const COLLISION: &str = "engine_collision's own check, after the development tools' and scenes' bounds, \
                         or its budgets, which overflow only with more colliders than the corpora load";

/// Codes of the catalog no refusal here or in the scenes corpus reaches,
/// with why. Sorted by code.
const UNREACHED: &[(&str, &str)] = &[
    ("activation_committed", WATCH),
    ("activation_expired", WATCH),
    ("activation_failed", WATCH),
    ("activation_not_committed", WATCH),
    ("benchmark_failed", PAGE),
    ("benchmark_interrupted", PAGE),
    ("benchmark_timeout", PAGE),
    ("bridge_error", BRIDGE),
    ("bridge_unreachable", BRIDGE),
    ("browser_command_failed", PAGE),
    ("build_mismatch", BRIDGE),
    ("build_result_too_large", WATCH),
    ("build_superseded", WATCH),
    ("build_worker_failed", WATCH),
    ("build_worker_unavailable", WATCH),
    ("capture_failed", BRIDGE),
    ("capture_requires_pause", BRIDGE),
    ("collision_candidate_budget", COLLISION),
    ("collision_incomplete_world", COLLISION),
    ("collision_invalid_cast", COLLISION),
    ("collision_invalid_config", COLLISION),
    ("collision_invalid_ray", COLLISION),
    ("collision_invalid_shape", COLLISION),
    ("collision_invalid_transform", COLLISION),
    ("collision_numerical_failure", COLLISION),
    ("collision_pair_budget", COLLISION),
    ("collision_result_budget", COLLISION),
    ("command_busy", PAGE),
    ("command_timeout", BRIDGE),
    ("content_type", BRIDGE),
    ("duplicate_request", BRIDGE),
    (
        "game_rule",
        "a game's own refusal (engine::Error::game_rule): the lod smoke's unknown preset reaches it",
    ),
    (
        "invalid_arguments",
        "hosts' and tools' command lines answer it (crates/engine_native/src/options.rs)",
    ),
    ("invalid_browser_response", BRIDGE),
    ("invalid_build_result", WATCH),
    ("invalid_candidate", WATCH),
    ("invalid_command", BRIDGE),
    ("invalid_fingerprint", WATCH),
    (
        "invalid_input",
        "the host checks it while it runs the control (crates/engine/src/dev_host/controls/input.rs): host-parity's invalid input case reaches it",
    ),
    ("invalid_measurement", PAGE),
    ("invalid_request_id", BRIDGE),
    ("invalid_session", BRIDGE),
    (
        "invalid_step",
        "the host checks it while it runs the control (crates/engine/src/dev_host/controls/input.rs): host-parity's invalid step case reaches it",
    ),
    ("invalid_url", BRIDGE),
    ("iteration_error", WATCH),
    ("iteration_stopped", WATCH),
    ("missing_runtime_identity", BRIDGE),
    (
        "not_generated",
        "only a terrain read from a file refuses edits so, and the pure corpora load none",
    ),
    ("origin_rejected", BRIDGE),
    ("pix_unavailable", NATIVE_PROFILE),
    ("profile_too_large", NATIVE_PROFILE),
    ("profiling_unavailable", NATIVE_PROFILE),
    ("queue_full", BRIDGE),
    ("queue_timeout", BRIDGE),
    ("render_failed", "rendering needs a GPU"),
    ("request_outcome_unknown", BRIDGE),
    (
        "response_too_large",
        "replies fit by design: the schema generator checks that every description fits one, and inspect pages bound theirs",
    ),
    ("route_not_found", BRIDGE),
    ("runtime_busy", BRIDGE),
    ("runtime_not_paused", BRIDGE),
    ("runtime_panicked", PAGE),
    ("runtime_replaced", BRIDGE),
    ("runtime_unavailable", BRIDGE),
    ("session_cancelled", BRIDGE),
    ("session_closed", BRIDGE),
    ("session_mismatch", BRIDGE),
    ("session_replacing", BRIDGE),
    ("session_request_limit", BRIDGE),
    ("session_unavailable", BRIDGE),
    ("source_budget", WATCH),
    (
        "source_changed",
        "needs another writer between a save's validation and its rename",
    ),
    ("source_symlink", WATCH),
    ("source_type", WATCH),
    ("unsafe_output_path", WATCH),
    (
        "unsupported_command",
        "the development bridge answers it while a host carries out a control (crates/engine/src/dev_host/bridge.rs): the native and surfaces smokes reach it",
    ),
    ("watch_locked", WATCH),
];

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let fixtures = read_json(root, FIXTURES)?;
    let scenes = read_json(root, SCENES)?;
    let components = scenes["documents"]["components"].to_string();
    let mut engine = engine_with_features();
    let scene = instantiate(&mut engine, &components)?;
    // An engine without features, and a scene of another engine's world.
    let mut bare = Engine::default();
    let mut other = engine_with_features();
    let foreign = instantiate(&mut other, &components)?;
    let tools = DevTools::default();
    let mut reached = BTreeSet::new();
    for refusal in fixtures["refusals"]
        .as_array()
        .ok_or_else(|| format!("{FIXTURES} has no refusals"))?
    {
        let rule = refusal["rule"].as_str().unwrap_or_default();
        let path = refusal["path"].as_str().unwrap_or_default();
        let name = format!("refusal {rule} at {path}");
        let source = request_source(refusal);
        let error = match refusal["stage"].as_str().unwrap_or("query") {
            "control" => control(
                &mut engine,
                Some(&scene),
                &source,
                refusal["paused"].as_bool().unwrap_or(true),
            ),
            "register" => register(refusal),
            "authoring" => authoring(root, refusal)?,
            _ => match (refusal["engine"].as_str(), refusal["scene"].as_str()) {
                (Some("bare"), _) => query(&tools, &mut bare, None, &source),
                // An engine whose asset library was taken away.
                (Some("without assets"), _) => {
                    let mut fresh = Engine::default();
                    fresh
                        .app
                        .world_mut()
                        .remove_resource::<engine::AssetLibrary>();
                    query(&tools, &mut fresh, None, &source)
                }
                (_, Some("none")) => query(&tools, &mut engine, None, &source),
                (_, Some("foreign")) => query(&tools, &mut engine, Some(&foreign), &source),
                // A scene of more entities than lookups scan.
                (_, Some("large")) => {
                    let mut large = engine_with_features();
                    let scene = instantiate(&mut large, &crowd())?;
                    query(&tools, &mut large, Some(&scene), &source)
                }
                // The components document with one entity despawned since.
                (_, Some(despawned)) if despawned.starts_with("without ") => {
                    let mut fresh = engine_with_features();
                    let scene = instantiate(&mut fresh, &components)?;
                    let id = &despawned["without ".len()..];
                    if let Some(entity) = scene.entity(id) {
                        fresh.app.world_mut().despawn(entity);
                    }
                    query(&tools, &mut fresh, Some(&scene), &source)
                }
                _ => query(&tools, &mut engine, Some(&scene), &source),
            },
        };
        let Some(error) = error else {
            observed.check(false, || format!("{name}: it was accepted"));
            continue;
        };
        let code = error.code.as_str();
        observed.check(code == rule && error.path == path, || {
            format!(
                "{name}: expected {rule} at {path}, found {code} at {}: {}",
                error.path, error.message
            )
        });
        reached.insert(code.to_owned());
        observed.record(
            name,
            json!({"code": code, "path": error.path, "message": error.message}),
        );
    }
    for refusal in scenes["refusals"].as_array().into_iter().flatten() {
        if let Some(rule) = refusal["rule"].as_str() {
            reached.insert(rule.to_owned());
        }
    }
    // Every code is reached, or listed as reached by none, with why.
    observed.check(
        UNREACHED.windows(2).all(|pair| pair[0].0 < pair[1].0),
        || "UNREACHED (tools/xtask/src/pure/requests.rs) must be sorted and unique".to_owned(),
    );
    for &(_, code, _) in ErrorCode::ALL {
        let unreached = UNREACHED.iter().any(|(listed, _)| *listed == code);
        observed.check(reached.contains(code) != unreached, || {
            if unreached {
                format!("{code} is in UNREACHED, but a refusal reaches it")
            } else {
                format!(
                    "no refusal in {FIXTURES} or {SCENES} answers {code}: add one, or list it in UNREACHED with why"
                )
            }
        });
    }
    for (code, _) in UNREACHED {
        observed.check(ErrorCode::named(code).is_some(), || {
            format!("UNREACHED lists {code}, which is no code")
        });
    }
    Ok(())
}

/// An engine with every feature's plugin.
fn engine_with_features() -> Engine {
    let mut engine = Engine::default();
    engine.app.add_plugins(EnginePlugins);
    engine
}

/// `document` instantiated into `engine`'s world.
fn instantiate(engine: &mut Engine, document: &str) -> Result<SceneInstance, String> {
    SceneDocument::from_json(document)
        .and_then(|document| document.instantiate(engine.app.world_mut()))
        .map_err(|error| format!("a corpus scene: {error}"))
}

/// A scene of one entity more than lookups scan.
fn crowd() -> String {
    let entities: Vec<Value> = (0..=MAX_SCAN_ENTITIES)
        .map(|index| json!({"id": format!("e{index}")}))
        .collect();
    json!({"id": "crowd", "revision": 1, "materials": [], "meshes": [], "entities": entities})
        .to_string()
}

/// The request a refusal sends: its `source` text (`OVERSIZED` for one
/// byte over the bound), or its `request` with the protocol's version.
fn request_source(refusal: &Value) -> String {
    match &refusal["source"] {
        Value::String(source) if source == "OVERSIZED" => format!(
            "{{\"op\":\"status\",\"version\":{PROTOCOL_VERSION},\"pad\":\"{}\"}}",
            " ".repeat(MAX_REQUEST_BYTES)
        ),
        Value::String(source) => source.clone(),
        _ => {
            let mut request = refusal["request"].clone();
            if request.get("version").is_none() {
                request["version"] = json!(PROTOCOL_VERSION);
            }
            request.to_string()
        }
    }
}

/// `source` as hosts hand the development tools a request: the reply's
/// error, if it failed.
fn query(
    tools: &DevTools,
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    source: &str,
) -> Option<DevError> {
    let context = Context {
        scene,
        renderer: None,
        render_slot: None,
    };
    let reply: Value = serde_json::from_str(&tools.execute(engine, context, source)).ok()?;
    if reply["ok"] == true {
        return None;
    }
    let code = ErrorCode::named(reply["error"]["code"].as_str()?)?;
    Some(DevError::new(
        code,
        reply["error"]["path"].as_str().unwrap_or_default(),
        reply["error"]["message"].as_str().unwrap_or_default(),
    ))
}

/// `source` as hosts take a control: parsed and checked on a host `paused`
/// or not, then a feature's control applied as its development file does
/// (the host's own need a renderer, which a pure check has not).
fn control(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    source: &str,
    paused: bool,
) -> Option<DevError> {
    let op = serde_json::from_str::<Value>(source)
        .ok()
        .and_then(|request| request["op"].as_str().map(str::to_owned))
        .unwrap_or_default();
    let control = match BrowserControl::parse(source, &op, paused) {
        Ok(control) => control,
        Err(error) => return Some(error),
    };
    match control.route() {
        Control::Feature(control) => control.apply(engine, scene).err(),
        Control::Host(_) => None,
    }
}

/// A registration of the inspector `refusal` describes: the error of the
/// last of `count` registered on fresh tools, 1 by default and one more
/// than games may register for `"beyond"`.
fn register(refusal: &Value) -> Option<DevError> {
    let mut tools = DevTools::default();
    let count = match &refusal["count"] {
        Value::String(beyond) if beyond == "beyond" => MAX_INSPECTORS as u64 + 1,
        count => count.as_u64().unwrap_or(1),
    };
    let repeat = refusal["repeat"].as_u64().unwrap_or(1) as usize;
    let mut last = None;
    for index in 0..count {
        let name = match refusal["name"].as_str() {
            Some(name) if count == 1 => name.to_owned(),
            Some(name) => format!("{name}{index}"),
            None => format!("Probe{index}"),
        };
        let description = refusal["description"]
            .as_str()
            .unwrap_or("a probe")
            .repeat(repeat);
        let descriptor = InspectorDescriptor::new(name, description, json!({"type": "object"}));
        last = tools
            .register_resource::<engine::FixedTime>(descriptor, |_| Value::Null)
            .err();
    }
    last
}

/// The scene save or read `refusal` describes, in a folder of its own
/// under [`AUTHORING`]: its `saved` document (and a `lock` left by
/// another save, or `bytes` instead of a document), then `candidate`
/// saved over it expecting `expected` (`"max"` for the last revision), or
/// the file read back. A path naming the file reads `FILE`.
fn authoring(root: &Path, refusal: &Value) -> Result<Option<DevError>, String> {
    let folder = root
        .join(AUTHORING)
        .join(refusal["rule"].as_str().unwrap_or("refusal"));
    if folder.exists() {
        fs::remove_dir_all(&folder).map_err(|error| error.to_string())?;
    }
    fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let file = folder.join("scene.json");
    let bytes: Option<Vec<u8>> = refusal["bytes"].as_array().map(|bytes| {
        bytes
            .iter()
            .map(|byte| byte.as_u64().unwrap_or_default() as u8)
            .collect()
    });
    match (&refusal["saved"], bytes) {
        (_, Some(bytes)) => fs::write(&file, bytes),
        (Value::Null, None) => Ok(()),
        (saved, None) => fs::write(&file, saved.to_string()),
    }
    .map_err(|error| error.to_string())?;
    if refusal["lock"] == true {
        fs::write(folder.join(".scene.json.engine-dev.lock"), b"")
            .map_err(|error| error.to_string())?;
    }
    let target = match refusal["target"].as_str() {
        // The folder's root, which names no file.
        Some("root") => folder
            .ancestors()
            .last()
            .map_or_else(|| folder.clone(), Path::to_path_buf),
        _ => file.clone(),
    };
    let error = match &refusal["candidate"] {
        Value::Null => engine_dev::authoring::read_scene_source(&target).err(),
        candidate => {
            let expected = match &refusal["expected"] {
                Value::String(max) if max == "max" => u64::MAX,
                expected => expected.as_u64().unwrap_or_default(),
            };
            engine_dev::authoring::save_scene(&target, &candidate.to_string(), expected).err()
        }
    };
    // Paths naming the files read alike on every machine, and so do the
    // operating system's messages, which differ between systems.
    Ok(error.map(|mut error| {
        if error.code == ErrorCode::SourceIo {
            "the operating system's message".clone_into(&mut error.message);
        }
        for named in [folder.join(".scene.json.engine-dev.lock"), file, target] {
            let shown = named.display().to_string();
            let canonical = fs::canonicalize(&named)
                .map(|path| path.display().to_string())
                .unwrap_or_default();
            for text in [shown, canonical] {
                if !text.is_empty() && error.path.contains(&text) {
                    error.path = "FILE".to_owned();
                }
                if !text.is_empty() {
                    error.message = error.message.replace(&text, "FILE");
                }
            }
        }
        error
    }))
}

/// The JSON file at `path` from the repository root.
fn read_json(root: &Path, path: &str) -> Result<Value, String> {
    let text = fs::read_to_string(root.join(path))
        .map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("{path} is not JSON: {error}"))
}
