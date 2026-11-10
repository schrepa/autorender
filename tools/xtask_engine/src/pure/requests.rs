use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use engine::{Engine, Error, ErrorCode};
use engine_dev::{
    Context, Control, DevTools, Envelope, InspectorDescriptor, MAX_SCAN_ENTITIES, PROTOCOL_VERSION,
};
use engine_scene::{SceneDocument, SceneInstance, SceneKinds};
use serde_json::{Value, json};

use super::Observations;

const FIXTURES: &str = "tools/xtask_engine/pure/requests.json";
const SCENES: &str = "tools/xtask_engine/pure/scenes.json";
const AUTHORING: &str = "target/pure/requests";

const BRIDGE: &str = "the agent bridge and its sessions answer it (web/scripts/agent*.mjs)";
const PAGE: &str = "the page answers it (web/src/engine/dev.ts and the benchmark it runs)";
const NATIVE_PROFILE: &str = "native profile captures answer it (crates/engine_native)";
const CHECKS: &str = "the checks report it (crates/engine_checks)";
const LINKER: &str = "the shader linker answers it at install, which the no-op install runs (crates/engine_render/src/linker.rs)";
const APP_BUILD: &str = "building the App refuses it, and the no-op install builds every listed brick (crates/engine_registry, crates/engine_render)";
const SPAWNER: &str =
    "a spawner reports it when a command reaches another brick's entity (crates/engine_core)";
const IMPORT: &str = "the glTF importer answers it (tools/assetc)";
const RECORDS: &str = "records that do not decode answer it (crates/engine_codec/src/lib.rs)";
const LAYERS: &str = "terrain layer stacks answer it (crates/engine_heightfield/src/layers.rs)";
const FONTS: &str = "registering font bytes answers it (crates/engine_text/src/typeset.rs)";
const PACKS: &str = "packs that do not read answer it (crates/engine_worker/src/pack.rs)";
const COLLISION: &str = "engine_collision's own check, after the development tools' and scenes' bounds, \
                         or its budgets, which overflow only with more colliders than the corpora load";

const UNREACHED: &[(&str, &str)] = &[
    ("ambiguous_passes", APP_BUILD),
    ("attachment_limit", APP_BUILD),
    (
        "audio_failed",
        "the mixer answers it when a sound does not decode or the audio device fails (crates/engine_audio/src/mixer.rs); the corpora play no sound",
    ),
    ("benchmark_failed", PAGE),
    ("benchmark_interrupted", PAGE),
    ("benchmark_timeout", PAGE),
    (
        "binding_conflict",
        "readiness reports it when two verbs claim one source in one active input context (crates/engine_core/src/input.rs); no request refuses with it",
    ),
    ("binding_foreign", LINKER),
    ("binding_limit", LINKER),
    ("bridge_error", BRIDGE),
    ("bridge_unreachable", BRIDGE),
    ("browser_command_failed", PAGE),
    ("build_mismatch", BRIDGE),
    ("capture_failed", BRIDGE),
    ("capture_requires_pause", BRIDGE),
    (
        "code_too_long",
        "reading an error whose code names more bytes than a brick code holds answers it (crates/engine_codec/src/codes.rs)",
    ),
    ("collision_candidate_budget", COLLISION),
    ("collision_collider_budget", COLLISION),
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
    ("comment", CHECKS),
    ("content_type", BRIDGE),
    ("duplicate_request", BRIDGE),
    ("error_type", CHECKS),
    ("exception_unused", CHECKS),
    ("file_too_long", CHECKS),
    ("foreign_access", APP_BUILD),
    ("foreign_entity", SPAWNER),
    ("game_rule", "a game's own refusal (ErrorCode::GameRule)"),
    ("import_foreign", LINKER),
    (
        "invalid_arguments",
        "hosts' and tools' command lines answer it (crates/engine_native/src/options.rs)",
    ),
    ("invalid_browser_response", BRIDGE),
    ("invalid_command", BRIDGE),
    ("invalid_font", FONTS),
    ("invalid_gltf", IMPORT),
    (
        "invalid_input",
        "the host checks it while it runs the control (crates/engine/src/dev_host/controls/input.rs): host-parity's invalid input case reaches it",
    ),
    ("invalid_measurement", PAGE),
    ("invalid_pack", PACKS),
    ("invalid_record", RECORDS),
    (
        "invalid_renderable",
        "the drawn table refuses a drawn thing whose geometry is missing or lacks its kind's attributes as frames settle (crates/engine_core/src/drawn.rs); the corpora instantiate scenes but run no frames",
    ),
    ("invalid_request_id", BRIDGE),
    (
        "invalid_step",
        "the host checks it while it runs the control (crates/engine/src/dev_host/controls/input.rs): host-parity's invalid step case reaches it",
    ),
    ("invalid_url", BRIDGE),
    (
        "job_unsupported",
        "a job of a kind the list lacks answers it, and so does the impostors brick's bake kind, which only the GPU makes (crates/engine_core/src/jobs.rs, crates/engine_impostors/src/books.rs)",
    ),
    ("lint_allowed", CHECKS),
    ("lint_missing", CHECKS),
    ("marker", CHECKS),
    ("missing_layer", LAYERS),
    (
        "missing_plugin",
        "a brick answers it when a request needs a brick or face the game left out or gated off (crates/engine_dev/src/bricks.rs, crates/engine_collision, crates/engine_navigation, crates/engine_terrain); the corpora list every brick and gate none off",
    ),
    ("missing_runtime_identity", BRIDGE),
    ("module_outside_brick", APP_BUILD),
    (
        "nav_steps_not_taken",
        "navigation records it when a walking agent's step is not applied by the next tick (crates/engine_navigation); requests run no ticks with agents, and the walker_without_grounding_is_refused smoke reaches it",
    ),
    ("not_inherited", CHECKS),
    ("origin_rejected", BRIDGE),
    (
        "pass_mismatch",
        "a part's first record of a pass it did not declare answers it, which needs a GPU",
    ),
    ("pieces_limit", APP_BUILD),
    ("pix_unavailable", NATIVE_PROFILE),
    ("place_cycle", APP_BUILD),
    ("place_not_filled", APP_BUILD),
    ("place_unlisted", LINKER),
    ("profiling_unavailable", NATIVE_PROFILE),
    ("queue_full", BRIDGE),
    ("queue_timeout", BRIDGE),
    ("read_before_write", APP_BUILD),
    ("render_failed", "rendering needs a GPU"),
    ("request_outcome_unknown", BRIDGE),
    ("request_too_large", BRIDGE),
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
    ("safety_missing", CHECKS),
    ("schedule_ambiguity", APP_BUILD),
    ("session_closed", BRIDGE),
    ("session_mismatch", BRIDGE),
    ("session_request_limit", BRIDGE),
    ("session_unavailable", BRIDGE),
    ("shader_link", LINKER),
    ("slot_filled_twice", APP_BUILD),
    (
        "source_changed",
        "needs another writer between a save's validation and its rename",
    ),
    ("struct_too_wide", CHECKS),
    ("unknown_module", LINKER),
    (
        "unknown_value",
        "the registry answers it for a settings key no listed brick declares, and the worker for a step of a job it never started (crates/engine_registry/src/registry.rs, crates/engine_worker/src/lib.rs)",
    ),
    ("unknown_variable", APP_BUILD),
    ("unowned_system", APP_BUILD),
    ("unsafe_unlisted", CHECKS),
    (
        "unsupported_command",
        "the development bridge answers it while a host carries out a control (crates/engine/src/dev_host/bridge.rs): the native and surfaces smokes reach it",
    ),
    ("unsupported_feature", IMPORT),
    ("usage_conflict", APP_BUILD),
    ("version_unpinned", CHECKS),
    ("view_writer_unplaced", APP_BUILD),
    ("wgsl_unused", LINKER),
    ("written_twice", APP_BUILD),
];

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let fixtures = read_json(root, FIXTURES)?;
    let scenes = read_json(root, SCENES)?;
    let components = scenes["documents"]["components"].to_string();
    let mut engine = engine_with_features()?;
    let codes = engine
        .app
        .world()
        .get_resource::<engine_registry::Registry>()
        .map_or_else(
            || ErrorCode::NAMES.to_vec(),
            engine_registry::Registry::error_codes,
        );
    let scene = instantiate(&mut engine, &components)?;
    let mut bare = Engine::default();
    let mut other = engine_with_features()?;
    let foreign = instantiate(&mut other, &components)?;
    let tools = DevTools::new(engine::stats::frame_stats_schema());
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
                &tools,
                &mut engine,
                Some(&scene),
                &source,
                refusal["paused"].as_bool().unwrap_or(true),
            ),
            "register" => register(refusal),
            "authoring" => authoring(root, refusal)?,
            _ => match (refusal["engine"].as_str(), refusal["scene"].as_str()) {
                (Some("bare"), _) => query(&tools, &mut bare, None, &source),
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
                (_, Some("large")) => {
                    let mut large = engine_with_features()?;
                    let scene = instantiate(&mut large, &crowd())?;
                    query(&tools, &mut large, Some(&scene), &source)
                }
                (_, Some(despawned)) if despawned.starts_with("without ") => {
                    let mut fresh = engine_with_features()?;
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
                "{name}: expected {rule} at {path}, found {code} at {}",
                error.path
            )
        });
        reached.insert(code.to_owned());
        observed.record(name, json!(error));
    }
    for refusal in scenes["refusals"].as_array().into_iter().flatten() {
        if let Some(rule) = refusal["rule"].as_str() {
            reached.insert(rule.to_owned());
        }
    }
    observed.check(
        UNREACHED.windows(2).all(|pair| pair[0].0 < pair[1].0),
        || {
            "UNREACHED (tools/xtask_engine/src/pure/requests.rs) must be sorted and unique"
                .to_owned()
        },
    );
    for &code in &codes {
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
        observed.check(codes.contains(code), || {
            format!("UNREACHED lists {code}, which is no code")
        });
    }
    Ok(())
}

fn engine_with_features() -> Result<Engine, String> {
    crate::bricks::standard_engine().map_err(|error| error.to_string())
}

fn instantiate(engine: &mut Engine, document: &str) -> Result<SceneInstance, String> {
    SceneDocument::from_json(document)
        .and_then(|document| {
            document.validate(&SceneKinds::of(engine.app.world()))?;
            document.instantiate(engine.app.world_mut())
        })
        .map_err(|error| format!("a corpus scene: {error}"))
}

fn crowd() -> String {
    let entities: Vec<Value> = (0..=MAX_SCAN_ENTITIES)
        .map(|index| json!({"id": format!("e{index}")}))
        .collect();
    json!({"id": "crowd", "revision": 1, "materials": [], "meshes": [], "entities": entities})
        .to_string()
}

fn request_source(refusal: &Value) -> String {
    match &refusal["source"] {
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

fn query(
    tools: &DevTools,
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    source: &str,
) -> Option<Error> {
    let context = Context {
        scene,
        renderer: None,
    };
    let reply: Value = serde_json::from_str(&tools.execute(engine, context, source)).ok()?;
    if reply["ok"] == true {
        return None;
    }
    serde_json::from_value(reply["error"].clone()).ok()
}

fn control(
    tools: &DevTools,
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    source: &str,
    paused: bool,
) -> Option<Error> {
    let envelope = match Envelope::parse(source) {
        Ok(envelope) => envelope,
        Err(error) => return Some(error),
    };
    let control = match tools.parse_control(engine.app.world(), &envelope, paused) {
        Ok(control) => control,
        Err(error) => return Some(error),
    };
    match control {
        Control::Operation(call) => call.apply(engine, scene).err(),
        Control::Host(_) => None,
    }
}

fn register(refusal: &Value) -> Option<Error> {
    let mut tools = DevTools::new(engine::stats::frame_stats_schema());
    let repeat = refusal["repeat"].as_u64().unwrap_or(1) as usize;
    let name = refusal["name"].as_str().unwrap_or("Probe");
    let description = refusal["description"]
        .as_str()
        .unwrap_or("a probe")
        .repeat(repeat);
    let descriptor =
        InspectorDescriptor::new(name, json!({"type": "object", "title": description}));
    tools
        .register_resource::<engine::FixedTime>(descriptor, |_| Value::Null)
        .err()
}

fn authoring(root: &Path, refusal: &Value) -> Result<Option<Error>, String> {
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
            let kinds = crate::bricks::standard_kinds().map_err(|error| error.to_string())?;
            engine_dev::authoring::save_scene(
                &target,
                &candidate.to_string(),
                expected,
                &kinds,
                crate::development::alive,
            )
            .err()
        }
    };
    Ok(error.map(|mut error| {
        if error.code == ErrorCode::SourceIo {
            error.detail.got = None;
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
                if !text.is_empty()
                    && let Some(Value::String(got)) = &mut error.detail.got
                {
                    *got = got.replace(&text, "FILE");
                }
            }
        }
        error
    }))
}

fn read_json(root: &Path, path: &str) -> Result<Value, String> {
    let text = fs::read_to_string(root.join(path))
        .map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("{path} is not JSON: {error}"))
}
