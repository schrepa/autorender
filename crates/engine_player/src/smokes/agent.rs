use engine::scene::SceneDocument;
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_standard_smokes::requests::{
    Asked, accepted, component, control, environment, inspected, named, names, number, object,
    query, refused, refused_with, seen, serve, sorted_keys, status, tick,
};
use serde_json::{Value, json};

use super::pixels::{differing, pixels};

const GARDEN: &str = include_str!("../../../../web/scripts/fixtures/garden.scene.json");
pub(crate) const PLAYER: &str = "player";
pub(crate) const MESHES: &str = "engine_meshes";
const COLLISION: &str = "engine_collision";
const UNATTRIBUTED: &str = "unattributed";
const SETTLE_RENDERS: usize = 240;
const LIGHT: [(&str, &str); 4] = [
    ("sun", "inspects_sun"),
    ("sky_light", "inspects_sky_light"),
    ("surface_light", "inspects_surface_light"),
    ("grading", "inspects_grading"),
];

pub(crate) fn garden() -> Asked {
    let parsed = serde_json::from_str::<Value>(GARDEN);
    let wanted = json!(true);
    smoke_eq!(
        seen(parsed.is_ok(), json!(false), &wanted),
        wanted,
        "garden_parses"
    );
    Ok(parsed.unwrap_or_default())
}

fn effects(rendering: &Value) -> Value {
    json!([
        rendering["shadow_stats"]["active"],
        rendering["ambient_occlusion_stats"]["active"],
        rendering["bloom_stats"]["active"],
    ])
}

fn light(environment: &Value) -> Value {
    Value::Object(
        LIGHT
            .iter()
            .map(|&(key, _)| (key.to_owned(), environment[key].clone()))
            .collect(),
    )
}

pub(crate) fn agent(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let source = garden()?;
    let revision = source["revision"].as_u64().unwrap_or_default();
    serve(session, &source)?;
    let before = discovered(session, revision)?;
    let after = walked(session, &before)?;
    let day = captured(session, &after)?;
    let look = looked(session, &after, &day)?;
    let night = darkened(session, &after, &day, &look)?;
    reloaded(session, &source, revision)?;
    saved(session, &look, &night)?;
    explained(session)
}

fn discovered(session: &mut dyn SmokeSession, revision: u64) -> Asked {
    let status = status(session, "status_answers")?;
    smoke_eq!(
        status["result"]["paused"].clone(),
        json!(true),
        "starts_paused"
    );
    smoke_eq!(
        status["observation"]["scene"]["revision"].clone(),
        json!(revision.to_string()),
        "scene_revision_observed"
    );
    let transform = accepted(
        session,
        "describe",
        json!({ "name": "Transform" }),
        "describes",
    )?;
    let wanted = json!({ "descriptor": { "schema": "object" } });
    smoke_eq!(
        seen(
            transform["result"]["descriptor"]["schema"].is_object(),
            transform["result"].clone(),
            &wanted,
        ),
        wanted,
        "description"
    );
    let scene = accepted(
        session,
        "describe",
        json!({ "name": "SceneDocument" }),
        "describes_scenes",
    )?;
    let definitions: Vec<&str> = scene["result"]["schema"]["definitions"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let wanted = json!("SceneEnvironment");
    smoke_eq!(
        seen(
            definitions.contains(&"SceneEnvironment"),
            json!(definitions),
            &wanted,
        ),
        wanted,
        "scene_definitions_listed"
    );
    let body = json!({ "name": "SceneDocument", "definition": "SceneEnvironment" });
    let defined = accepted(session, "describe", body, "describes_definition")?;
    let wanted = json!({ "terrain": "object" });
    smoke_eq!(
        seen(
            defined["result"]["schema"]["properties"]["terrain"].is_object(),
            defined["result"]["schema"]["properties"].clone(),
            &wanted,
        ),
        wanted,
        "definition_described"
    );
    let body = json!({ "entity": PLAYER, "component": "Transform" });
    let before = accepted(session, "inspect", body, "inspection")?;
    let invalid = refused(
        session,
        "inspect",
        json!({ "limit": 101 }),
        "page_limit_refused",
    )?;
    smoke_eq!(
        json!(tick(&invalid)),
        json!(tick(&before)),
        "refusal_keeps_tick"
    );
    Ok(before)
}

fn translation(reply: &Value) -> Value {
    reply["result"]["entity"]["components"]["Transform"]["translation"].clone()
}

fn walked(session: &mut dyn SmokeSession, before: &Value) -> Asked {
    let still = query(session, "digest", json!({}), "digests")?["result"].clone();
    let wanted = json!("entities, transforms and meshes counted");
    smoke_eq!(
        seen(
            number(&still["entities"]) > 0.0
                && number(&still["components"]["Transform"]["count"]) > 0.0
                && number(&still["assets"]["meshes"]["count"]) > 0.0,
            json!({ "entities": still["entities"], "assets": still["assets"] }),
            &wanted,
        ),
        wanted,
        "digest"
    );
    control(session, "input", json!({ "forward": 1 }), "input")?;
    let stepped = control(session, "step", json!({ "ticks": 30 }), "steps")?;
    smoke_eq!(
        json!(tick(&stepped).wrapping_sub(tick(before))),
        json!(30),
        "precise_step"
    );
    let body = json!({ "entity": PLAYER, "component": "Transform" });
    let after = accepted(session, "inspect", body, "inspects_player")?;
    let (from, to) = (translation(before), translation(&after));
    let wanted = json!({ "z_below": from[2] });
    smoke_eq!(
        seen(number(&to[2]) < number(&from[2]), to.clone(), &wanted),
        wanted,
        "input_walks_player"
    );
    let walked = query(session, "digest", json!({}), "digests_walk")?["result"].clone();
    let wanted = json!({ "differs_from": still["hash"] });
    smoke_eq!(
        seen(
            walked["hash"] != still["hash"]
                && walked["components"]["Transform"]["hash"]
                    != still["components"]["Transform"]["hash"],
            walked["hash"].clone(),
            &wanted,
        ),
        wanted,
        "walk_changes_digest"
    );
    smoke_eq!(
        walked["assets"].clone(),
        still["assets"].clone(),
        "walk_changes_no_asset"
    );
    let again = query(session, "digest", json!({}), "digests_again")?;
    smoke_eq!(
        again["result"]["hash"].clone(),
        walked["hash"].clone(),
        "digest_changes_nothing"
    );
    control(session, "input", json!({ "forward": 0 }), "input_released")?;
    let bad = refused(
        session,
        "control",
        named("step", json!({ "ticks": 0 })),
        "zero_step_refused",
    )?;
    smoke_eq!(
        json!(tick(&bad)),
        json!(tick(&after)),
        "invalid_step_keeps_tick"
    );
    Ok(after)
}

fn captured(session: &mut dyn SmokeSession, after: &Value) -> Asked {
    let capture = accepted(session, "capture", json!({}), "capture")?;
    smoke_eq!(
        json!(tick(&capture)),
        json!(tick(after)),
        "capture_keeps_tick"
    );
    let artifact = &capture["result"]["artifact"];
    let wanted = json!({ "bytes": "> 1000" });
    smoke_eq!(
        seen(
            number(&artifact["bytes"]) > 1000.0,
            artifact.clone(),
            &wanted,
        ),
        wanted,
        "capture_holds_image"
    );
    smoke_eq!(
        artifact["tick"].clone(),
        json!(tick(after).to_string()),
        "capture_names_tick"
    );
    pixels(&capture)?;
    smoke_eq!(
        effects(&capture["result"]["rendering"]),
        json!([true, true, true]),
        "effects_drawn"
    );
    let schema = accepted(
        session,
        "describe",
        json!({ "name": "FrameStats" }),
        "describes_stats",
    )?;
    let schema = &schema["result"]["schema"];
    let mut rendering = capture["result"]["rendering"].clone();
    if let Some(fields) = rendering.as_object_mut() {
        fields.remove("player_walk_stats");
    }
    stats_described(schema, schema, &rendering, "rendering")?;
    Ok(capture)
}

fn stats_described(
    root: &Value,
    schema: &Value,
    stats: &Value,
    path: &str,
) -> Result<(), SmokeFailure> {
    let properties = &schema["properties"];
    let (got, wanted) = (sorted_keys(stats), sorted_keys(properties));
    let expected = json!({ "path": path, "fields": wanted });
    smoke_eq!(
        seen(
            got == wanted,
            json!({ "path": path, "fields": got }),
            &expected,
        ),
        expected,
        "frame_stats_described"
    );
    for (field, property) in object(properties) {
        let Some(definition) = property["$ref"]
            .as_str()
            .and_then(|at| at.rsplit('/').next())
        else {
            continue;
        };
        let inner = format!("{path}.{field}");
        stats_described(root, &root["$defs"][definition], &stats[&field], &inner)?;
    }
    Ok(())
}

fn listed(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let listing = accepted(session, "describe", json!({}), "lists_types")?;
    let types: Vec<String> = listing["result"]["types"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|described| described["name"].as_str().map(str::to_owned))
        .collect();
    for name in [
        "sun",
        "sky_light",
        "surface_light",
        "grading",
        "shadows",
        "ambient_occlusion",
        "bloom",
        "Environment",
    ] {
        let wanted = json!(name);
        smoke_eq!(
            seen(
                types.iter().any(|listed| listed == name),
                json!(types),
                &wanted,
            ),
            wanted,
            "environment_types_listed"
        );
    }
    Ok(())
}

fn looked(session: &mut dyn SmokeSession, after: &Value, day: &Value) -> Asked {
    listed(session)?;
    let build = status(session, "status_names_build")?["result"].clone();
    let wanted = json!({ "features": ["dev"], "version": "string", "adapter": "present" });
    smoke_eq!(
        seen(
            names(&build["build"]["features"])
                .iter()
                .any(|feature| feature == "dev")
                && build["build"]["version"].is_string()
                && build.get("adapter").is_some(),
            json!({ "build": build["build"], "adapter": build["adapter"] }),
            &wanted,
        ),
        wanted,
        "status_names_build_and_adapter"
    );
    let look = inspected(session, "Environment", "environment_inspection")?;
    for (key, check) in LIGHT {
        let section = inspected(session, key, check)?;
        smoke_eq!(section, look[key].clone(), "light_inside_environment");
    }
    let described = accepted(
        session,
        "describe",
        json!({ "name": "Environment" }),
        "describes_environment",
    )?;
    smoke_eq!(
        json!(sorted_keys(&look)),
        json!(sorted_keys(
            &described["result"]["descriptor"]["schema"]["properties"]
        )),
        "environment_sections_described"
    );
    let mesh = component(session, PLAYER, "Mesh3d", "inspects_mesh")?;
    let wanted = json!({ "surface": { "roughness": "number" } });
    smoke_eq!(
        seen(
            mesh["material"]["surface"]["roughness"].is_number(),
            mesh["material"].clone(),
            &wanted,
        ),
        wanted,
        "mesh_material_inspected"
    );
    let interpolation = component(
        session,
        PLAYER,
        "InterpolateTransform",
        "inspects_interpolation",
    )?;
    smoke_eq!(
        interpolation["enabled"].clone(),
        json!(true),
        "player_interpolates"
    );
    let jobs = inspected(session, "Jobs", "inspects_jobs")?;
    let tables = inspected(session, "static_tables", "inspects_tables")?;
    let wanted = json!({ "cache_hits": "number", "failed_jobs": "number" });
    smoke_eq!(
        seen(
            jobs["cache_hits"].is_number() && tables["failed_jobs"].is_number(),
            json!({ "jobs": jobs, "tables": tables }),
            &wanted,
        ),
        wanted,
        "jobs_and_tables_inspected"
    );
    let body = named("environment", json!({ "shadows": { "tile_texels": 100 } }));
    let invalid = refused_with(
        session,
        "control",
        body,
        "invalid_environment",
        "environment_validated",
    )?;
    smoke_eq!(
        json!([invalid["error"]["path"], tick(&invalid)]),
        json!(["$.args.shadows.tile_texels", tick(after)]),
        "invalid_environment_located"
    );
    smoke_eq!(
        inspected(session, "Environment", "inspects_environment_again")?,
        look.clone(),
        "invalid_environment_changes_nothing"
    );
    let rendering = &day["result"]["rendering"];
    smoke_eq!(
        json!([
            rendering["haze_stats"]["active"],
            rendering["reflections_stats"]["active"]
        ]),
        json!([true, false]),
        "day_haze_without_reflections"
    );
    Ok(look)
}

fn darkened(session: &mut dyn SmokeSession, after: &Value, day: &Value, look: &Value) -> Asked {
    let args = json!({
        "sun": { "preset": "night" },
        "sky_light": { "preset": "night" },
        "surface_light": { "preset": "night" },
        "grading": { "preset": "night" },
        "bloom": { "intensity": 0 },
        "haze": { "reach": 0 },
        "reflections": { "enabled": true },
        "highlights": { "capped": false },
        "sky_occlusion": { "strength": 0.5, "floor": 0.1 },
    });
    let night = environment(session, args, "environment_control")?;
    let result = &night["result"];
    let mut changed = names(&result["environment_changed"]);
    changed.sort();
    smoke_eq!(
        json!([result["temporary"], changed]),
        json!([
            true,
            [
                "bloom",
                "grading",
                "haze",
                "highlights",
                "reflections",
                "sky_light",
                "sky_occlusion",
                "sun",
                "surface_light"
            ]
        ]),
        "night_changes_sections"
    );
    let rendering = &result["rendering"];
    let shadows = &rendering["shadow_stats"];
    let wanted = json!({ "sky_maps": "<= 26", "sky_drawn": "<= 1" });
    smoke_eq!(
        seen(
            number(&shadows["sky_maps"]) <= 26.0 && number(&shadows["sky_drawn"]) <= 1.0,
            shadows.clone(),
            &wanted,
        ),
        wanted,
        "sky_shadows_bounded"
    );
    smoke_eq!(
        effects(rendering),
        json!([true, true, false]),
        "night_effects"
    );
    smoke_eq!(
        json!([
            rendering["haze_stats"]["active"],
            rendering["reflections_stats"]["active"],
            tick(&night)
        ]),
        json!([false, true, tick(after)]),
        "night_reflects_without_haze"
    );
    let dark = inspected(session, "Environment", "inspects_night")?;
    let wanted = json!({ "light": "changed", "shadows": look["shadows"] });
    smoke_eq!(
        seen(
            light(&dark) != light(look) && dark["shadows"] == look["shadows"],
            dark.clone(),
            &wanted,
        ),
        wanted,
        "night_inspected"
    );
    let capture = accepted(session, "capture", json!({}), "captures_night")?;
    let wanted = json!({ "sha256_differs_from": day["result"]["artifact"]["sha256"] });
    smoke_eq!(
        seen(
            capture["result"]["artifact"]["sha256"] != day["result"]["artifact"]["sha256"],
            capture["result"]["artifact"].clone(),
            &wanted,
        ),
        wanted,
        "night_capture_differs"
    );
    if let (Some(light), Some(darker)) = (pixels(day)?, pixels(&capture)?) {
        let total = light.rgba.len() / 4;
        let changed = differing(&light, &darker, 8)?;
        let wanted = json!({ "changed": "> 1000" });
        smoke_eq!(
            seen(
                changed > 1000 && changed <= total,
                json!({ "changed": changed, "total": total }),
                &wanted,
            ),
            wanted,
            "night_pixels_differ"
        );
    }
    Ok(dark)
}

fn candidate(source: &Value) -> Value {
    let mut candidate = source.clone();
    candidate["revision"] = json!(number(&source["revision"]) as u64 + 1);
    candidate["environment"] = json!({
        "haze": { "enabled": true },
        "sun": { "preset": "dusk" },
        "sky_light": { "preset": "dusk" },
        "surface_light": { "preset": "dusk" },
        "grading": { "preset": "dusk" },
        "ambient_occlusion": { "intensity": 0 },
    });
    if let Some(entities) = candidate["entities"].as_array_mut() {
        for entity in entities.iter_mut().filter(|entity| entity["id"] == PLAYER) {
            entity["name"] = json!("Agent verified player");
        }
        entities.push(json!({
            "id": "agent-lamp", "transform": { "translation": [0, 3, 0] },
            "light": { "intensity": 20, "range": 8, "radius": 0.25, "length": 1.5, "region": "interior" },
            "particles": { "rate": 20, "lifetime": 1 },
        }));
        entities.push(json!({
            "id": "agent-glass-box", "mesh": source["meshes"][0]["id"], "material": "agent-glass",
            "transform": { "translation": [0, 1, 0] },
        }));
    }
    if let Some(materials) = candidate["materials"].as_array_mut() {
        materials.push(json!({
            "id": "agent-glass", "color": [0.6, 0.85, 1.0, 1.0],
            "alpha_mode": "mask", "alpha_cutoff": 0.3, "opacity": 0.4,
        }));
    }
    candidate
}

fn validates(document: &Value) -> bool {
    SceneDocument::from_json(&document.to_string())
        .and_then(|scene| {
            engine::scene_kinds(crate::bricks::<engine::BrickList>)
                .and_then(|kinds| scene.validate(&kinds))
        })
        .is_ok()
}

fn reloaded(
    session: &mut dyn SmokeSession,
    source: &Value,
    revision: u64,
) -> Result<(), SmokeFailure> {
    let next = (revision + 1).to_string();
    let stale = named(
        "reload",
        json!({ "expected_revision": "0", "revision": next }),
    );
    let rejected = refused(session, "control", stale, "stale_reload_refused")?;
    smoke_eq!(
        rejected["observation"]["scene"]["revision"].clone(),
        json!(revision.to_string()),
        "failed_reload"
    );
    let mut invalid = source.clone();
    invalid["revision"] = json!(revision + 1);
    invalid["entities"][0]["material"] = json!("agent-missing-material");
    let wanted = json!("refused");
    smoke_eq!(
        seen(
            !validates(&invalid),
            json!({ "material": "agent-missing-material" }),
            &wanted,
        ),
        wanted,
        "source_validation"
    );
    let candidate = candidate(source);
    let wanted = json!(true);
    smoke_eq!(
        seen(validates(&candidate), json!(false), &wanted),
        wanted,
        "candidate_validates"
    );
    session.source(&candidate.to_string());
    let args = json!({ "expected_revision": next, "revision": next });
    let reloaded = control(session, "reload", args, "reloads")?;
    smoke_eq!(
        reloaded["observation"]["scene"]["revision"].clone(),
        json!(next),
        "reload_revision"
    );
    let rendering = &reloaded["result"]["rendering"];
    smoke_eq!(
        json!([
            rendering["haze_stats"]["active"],
            rendering["reflections_stats"]["active"]
        ]),
        json!([true, false]),
        "reloaded_haze_without_reflections"
    );
    let lights = &rendering["local_light_stats"];
    let particles = &rendering["particle_stats"];
    let wanted = json!({ "shaded": ">= 1", "drawn": "> 0" });
    smoke_eq!(
        seen(
            number(&lights["shaded"]) >= 1.0 && number(&particles["drawn"]) > 0.0,
            json!({ "local_lights": lights, "particles": particles }),
            &wanted,
        ),
        wanted,
        "lights_particles_drawn"
    );
    let drawn = settled(session, rendering.clone())?;
    smoke_eq!(effects(&drawn), json!([true, false, true]), "saved_effects");
    Ok(())
}

fn settled(session: &mut dyn SmokeSession, mut rendering: Value) -> Asked {
    for _ in 0..SETTLE_RENDERS {
        if rendering["meshes_stats"]["missing_assets"] == 0 {
            break;
        }
        let rendered = control(session, "render", json!({}), "renders_reloaded_scene")?;
        rendering = rendered["result"]["rendering"].clone();
    }
    smoke_eq!(
        rendering["meshes_stats"]["missing_assets"].clone(),
        json!(0),
        "reloaded_scene_settles"
    );
    Ok(rendering)
}

fn saved(session: &mut dyn SmokeSession, look: &Value, night: &Value) -> Result<(), SmokeFailure> {
    let body = json!({ "entity": PLAYER, "component": "Transform" });
    let player = accepted(session, "inspect", body, "inspects_renamed_player")?;
    smoke_eq!(
        player["result"]["entity"]["name"].clone(),
        json!("Agent verified player"),
        "reload_renames_player"
    );
    let saved = inspected(session, "Environment", "inspects_saved_environment")?;
    smoke_eq!(
        json!([
            number(&saved["ambient_occlusion"]["intensity"]),
            saved["bloom"],
            saved["shadows"],
            saved["haze"],
            saved["reflections"]["enabled"],
            saved["highlights"]["enabled"],
            saved["highlights"]["capped"],
        ]),
        json!([
            0.0,
            look["bloom"],
            look["shadows"],
            look["haze"],
            false,
            true,
            true
        ]),
        "saved_environment"
    );
    let wanted = json!("dusk");
    smoke_eq!(
        seen(
            light(&saved) != light(night) && light(&saved) != light(look),
            light(&saved),
            &wanted,
        ),
        wanted,
        "saved_light"
    );
    let lamp = component(session, "agent-lamp", "light", "inspects_lamp")?;
    smoke_eq!(
        json!([
            number(&lamp["intensity"]),
            number(&lamp["radius"]),
            number(&lamp["length"]),
            lamp["region"],
        ]),
        json!([20.0, 0.25, 1.5, "interior"]),
        "lamp_light"
    );
    let emitter = component(session, "agent-lamp", "particles_component", "inspects_emitter")?;
    smoke_eq!(
        json!(number(&emitter["rate"])),
        json!(20.0),
        "lamp_particles"
    );
    let glass = component(session, "agent-glass-box", "Mesh3d", "inspects_glass")?;
    let opacity = number(&glass["material"]["opacity"]);
    let wanted = json!(0.4);
    smoke_eq!(
        seen((opacity - 0.4).abs() < 1e-6, json!(opacity), &wanted),
        wanted,
        "glass_opacity"
    );
    Ok(())
}

fn records(reply: &Value) -> Vec<Value> {
    reply["result"]["records"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn profiled(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let costs = accepted(
        session,
        "profile",
        json!({ "frames": 4, "detail": "full" }),
        "profiles",
    )?;
    let result = &costs["result"];
    let wanted = json!({ "bricks": "measured", "artifact": "a file" });
    smoke_eq!(
        seen(
            result["bricks"]
                .as_array()
                .is_some_and(|bricks| !bricks.is_empty())
                && number(&result["artifact"]["bytes"]) > 0.0,
            json!({ "bricks": result["bricks"], "artifact": result["artifact"] }),
            &wanted,
        ),
        wanted,
        "profile_measures_bricks"
    );
    let wanted = json!(">= 0");
    smoke_eq!(
        seen(
            number(&result["measuring_ms"]) >= 0.0,
            result["measuring_ms"].clone(),
            &wanted,
        ),
        wanted,
        "profile_reports_measuring_cost"
    );
    let unattributed = result["bricks"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|brick| brick["brick"] == UNATTRIBUTED)
        .cloned()
        .unwrap_or_default();
    let share = |field: &str| number(&unattributed[field]) / number(&result[field]);
    let wanted = json!({ "unattributed_share": "<= 0.05" });
    smoke_eq!(
        seen(
            share("cpu_ms") <= 0.05 && (result["gpu_ms"].is_null() || share("gpu_ms") <= 0.05),
            json!({ "row": unattributed, "cpu_ms": result["cpu_ms"], "gpu_ms": result["gpu_ms"] }),
            &wanted,
        ),
        wanted,
        "profile_rows_sum_to_frame"
    );
    Ok(())
}

fn explained(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let assets = query(session, "assets", json!({}), "lists_assets")?;
    let wanted = json!(">= 4");
    smoke_eq!(
        seen(
            number(&assets["result"]["meshes"]) >= 4.0,
            assets["result"]["meshes"].clone(),
            &wanted,
        ),
        wanted,
        "assets_counted"
    );
    profiled(session)?;
    let slow = accepted(
        session,
        "explain",
        json!({ "about": "frame", "top": 2 }),
        "explains_frame",
    )?;
    let wanted = json!({ "decided": "cost" });
    smoke_eq!(
        seen(
            records(&slow)
                .iter()
                .any(|record| record["decided"] == "cost"),
            json!(records(&slow)),
            &wanted,
        ),
        wanted,
        "frame_explained"
    );
    let body = json!({ "about": "entity", "entity": PLAYER });
    let why = accepted(session, "explain", body, "explains_entity")?;
    let wanted = json!({ "brick": "core" });
    smoke_eq!(
        seen(
            records(&why).iter().any(|record| record["brick"] == "core"),
            json!(records(&why)),
            &wanted,
        ),
        wanted,
        "entity_explained"
    );
    let body = json!({ "stats": ["tick"], "entities": [{ "entity": PLAYER, "component": "Transform" }], "frames": 2 });
    let watched = accepted(session, "watch", body, "watches")?;
    let wanted = json!({ "records": ">= 1" });
    smoke_eq!(
        seen(
            number(&watched["result"]["records"]) >= 1.0,
            watched["result"].clone(),
            &wanted,
        ),
        wanted,
        "watch_streams"
    );
    let raycast = accepted(
        session,
        "describe",
        json!({ "name": "raycast" }),
        "describes_raycast",
    )?;
    let raycast = &raycast["result"];
    smoke_eq!(
        json!([raycast["kind"], raycast["verb"], raycast["brick"]]),
        json!(["operation", "query", COLLISION]),
        "operation_described"
    );
    let resumed = control(session, "resume", json!({}), "resumes")?;
    smoke_eq!(
        resumed["result"]["paused"].clone(),
        json!(false),
        "resume_pause"
    );
    let step = named("step", json!({ "ticks": 1 }));
    refused_with(
        session,
        "control",
        step,
        "pause_required",
        "step_needs_pause",
    )?;
    control(session, "pause", json!({}), "pauses_again")?;
    Ok(())
}
