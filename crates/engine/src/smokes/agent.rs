use engine_core::smoke::{SmokeResult, SmokeSession};
use engine_scene::SceneDocument;
use serde_json::{Value, json};

use super::requests::{
    Asked, accepted, component, control, differing, environment, inspected, named, names, number,
    object, pixels, query, refused, refused_with, serve, sorted_keys, status, tick,
};

const GARDEN: &str = include_str!("../../../../web/scripts/fixtures/garden.scene.json");
pub(crate) const PLAYER: &str = "player";

pub(crate) fn garden() -> Asked {
    let parsed = serde_json::from_str::<Value>(GARDEN);
    engine_core::smoke_check!(parsed.is_ok(), "garden_parses", json!(false), json!(true));
    Ok(parsed.unwrap_or_default())
}

fn effects(rendering: &Value) -> Value {
    json!([
        rendering["shadows"]["active"],
        rendering["ambient_occlusion"]["active"],
        rendering["bloom"]["active"],
    ])
}

pub(crate) fn agent(session: &mut dyn SmokeSession) -> SmokeResult {
    let source = garden()?;
    let revision = source["revision"].as_u64().unwrap_or_default();
    serve(session, &source, &[])?;
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
    engine_core::smoke_eq!(
        status["result"]["paused"].clone(),
        json!(true),
        "starts_paused"
    );
    engine_core::smoke_eq!(
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
    engine_core::smoke_check!(
        transform["result"]["descriptor"]["schema"].is_object(),
        "description",
        transform["result"].clone(),
        json!({ "descriptor": { "schema": "object" } })
    );
    let scene = accepted(
        session,
        "describe",
        json!({ "name": "SceneDocument" }),
        "describes_scenes",
    )?;
    let definitions = names(&scene["result"]["schema"]["definitions"]);
    engine_core::smoke_check!(
        definitions.iter().any(|name| name == "SceneEnvironment"),
        "scene_definitions_listed",
        json!(definitions),
        json!("SceneEnvironment")
    );
    let body = json!({ "name": "SceneDocument", "definition": "SceneEnvironment" });
    let defined = accepted(session, "describe", body, "describes_definition")?;
    engine_core::smoke_check!(
        defined["result"]["schema"]["properties"]["terrain"].is_object(),
        "definition_described",
        defined["result"]["schema"]["properties"].clone(),
        json!({ "terrain": "object" })
    );
    let body = json!({ "entity": PLAYER, "component": "Transform" });
    let before = accepted(session, "inspect", body, "inspection")?;
    let invalid = refused(
        session,
        "inspect",
        json!({ "limit": 101 }),
        "page_limit_refused",
    )?;
    engine_core::smoke_eq!(
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
    engine_core::smoke_check!(
        number(&still["entities"]) > 0.0
            && number(&still["components"]["Transform"]["count"]) > 0.0
            && number(&still["assets"]["meshes"]["count"]) > 0.0,
        "digest",
        json!({ "entities": still["entities"], "assets": still["assets"] }),
        json!("entities, transforms and meshes counted")
    );
    control(session, "input", json!({ "forward": 1 }), "input")?;
    let stepped = control(session, "step", json!({ "ticks": 30 }), "steps")?;
    engine_core::smoke_eq!(
        json!(tick(&stepped).wrapping_sub(tick(before))),
        json!(30),
        "precise_step"
    );
    let body = json!({ "entity": PLAYER, "component": "Transform" });
    let after = accepted(session, "inspect", body, "inspects_player")?;
    let (from, to) = (translation(before), translation(&after));
    engine_core::smoke_check!(
        number(&to[2]) < number(&from[2]),
        "input_walks_player",
        to.clone(),
        json!({ "z_below": from[2] })
    );
    let walked = query(session, "digest", json!({}), "digests_walk")?["result"].clone();
    engine_core::smoke_check!(
        walked["hash"] != still["hash"]
            && walked["components"]["Transform"]["hash"]
                != still["components"]["Transform"]["hash"],
        "walk_changes_digest",
        walked["hash"].clone(),
        json!({ "differs_from": still["hash"] })
    );
    engine_core::smoke_eq!(
        walked["assets"].clone(),
        still["assets"].clone(),
        "walk_changes_no_asset"
    );
    let again = query(session, "digest", json!({}), "digests_again")?;
    engine_core::smoke_eq!(
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
    engine_core::smoke_eq!(
        json!(tick(&bad)),
        json!(tick(&after)),
        "invalid_step_keeps_tick"
    );
    Ok(after)
}

fn captured(session: &mut dyn SmokeSession, after: &Value) -> Asked {
    let capture = accepted(session, "capture", json!({}), "capture")?;
    engine_core::smoke_eq!(
        json!(tick(&capture)),
        json!(tick(after)),
        "capture_keeps_tick"
    );
    let artifact = &capture["result"]["artifact"];
    engine_core::smoke_check!(
        number(&artifact["bytes"]) > 1000.0,
        "capture_holds_image",
        artifact.clone(),
        json!({ "bytes": "> 1000" })
    );
    engine_core::smoke_eq!(
        artifact["tick"].clone(),
        json!(tick(after).to_string()),
        "capture_names_tick"
    );
    pixels(&capture)?;
    engine_core::smoke_eq!(
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
    stats_described(schema, schema, &capture["result"]["rendering"], "rendering")?;
    Ok(capture)
}

fn stats_described(root: &Value, schema: &Value, stats: &Value, path: &str) -> SmokeResult {
    let properties = &schema["properties"];
    let (got, wanted) = (sorted_keys(stats), sorted_keys(properties));
    engine_core::smoke_check!(
        got == wanted,
        "frame_stats_described",
        json!({ "path": path, "fields": got }),
        json!({ "path": path, "fields": wanted })
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

fn looked(session: &mut dyn SmokeSession, after: &Value, day: &Value) -> Asked {
    let listing = accepted(session, "describe", json!({}), "lists_types")?;
    let types: Vec<String> = listing["result"]["types"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|described| described["name"].as_str().map(str::to_owned))
        .collect();
    for name in [
        "Atmosphere",
        "Shadows",
        "AmbientOcclusion",
        "Bloom",
        "Environment",
    ] {
        engine_core::smoke_check!(
            types.iter().any(|listed| listed == name),
            "environment_types_listed",
            json!(types),
            json!(name)
        );
    }
    let build = status(session, "status_names_build")?["result"].clone();
    engine_core::smoke_check!(
        names(&build["build"]["features"])
            .iter()
            .any(|feature| feature == "dev")
            && build["build"]["version"].is_string()
            && build.get("adapter").is_some(),
        "status_names_build_and_adapter",
        json!({ "build": build["build"], "adapter": build["adapter"] }),
        json!({ "features": ["dev"], "version": "string", "adapter": "present" })
    );
    let look = inspected(session, "Environment", "environment_inspection")?;
    let atmosphere = inspected(session, "Atmosphere", "inspects_atmosphere")?;
    engine_core::smoke_eq!(
        atmosphere,
        look["atmosphere"].clone(),
        "atmosphere_inside_environment"
    );
    let described = accepted(
        session,
        "describe",
        json!({ "name": "Environment" }),
        "describes_environment",
    )?;
    engine_core::smoke_eq!(
        json!(sorted_keys(&look)),
        json!(sorted_keys(
            &described["result"]["descriptor"]["schema"]["properties"]
        )),
        "environment_sections_described"
    );
    let mesh = component(session, PLAYER, "Mesh3d", "inspects_mesh")?;
    engine_core::smoke_check!(
        mesh["material"]["surface"]["roughness"].is_number(),
        "mesh_material_inspected",
        mesh["material"].clone(),
        json!({ "surface": { "roughness": "number" } })
    );
    let interpolation = component(
        session,
        PLAYER,
        "InterpolateTransform",
        "inspects_interpolation",
    )?;
    engine_core::smoke_eq!(
        interpolation["enabled"].clone(),
        json!(true),
        "player_interpolates"
    );
    let jobs = inspected(session, "Jobs", "inspects_jobs")?;
    let tables = inspected(session, "StaticTables", "inspects_tables")?;
    engine_core::smoke_check!(
        jobs["cache_hits"].is_number() && tables["failed_jobs"].is_number(),
        "jobs_and_tables_inspected",
        json!({ "jobs": jobs, "tables": tables }),
        json!({ "cache_hits": "number", "failed_jobs": "number" })
    );
    let body = named("environment", json!({ "shadows": { "tile_texels": 100 } }));
    let invalid = refused_with(
        session,
        "control",
        body,
        "invalid_environment",
        "environment_validated",
    )?;
    engine_core::smoke_eq!(
        json!([invalid["error"]["path"], tick(&invalid)]),
        json!(["$.args.shadows.tile_texels", tick(after)]),
        "invalid_environment_located"
    );
    engine_core::smoke_eq!(
        inspected(session, "Environment", "inspects_environment_again")?,
        look.clone(),
        "invalid_environment_changes_nothing"
    );
    let rendering = &day["result"]["rendering"];
    engine_core::smoke_eq!(
        json!([
            rendering["haze"]["active"],
            rendering["reflections"]["active"]
        ]),
        json!([true, false]),
        "day_haze_without_reflections"
    );
    Ok(look)
}

fn darkened(session: &mut dyn SmokeSession, after: &Value, day: &Value, look: &Value) -> Asked {
    let args = json!({
        "atmosphere": { "preset": "night" },
        "bloom": { "intensity": 0 },
        "haze": { "reach": 0 },
        "reflections": {},
        "highlights": { "capped": false },
        "sky_occlusion": { "strength": 0.5, "floor": 0.1 },
    });
    let night = environment(session, args, "environment_control")?;
    let result = &night["result"];
    let mut changed = names(&result["environment_changed"]);
    changed.sort();
    engine_core::smoke_eq!(
        json!([result["temporary"], changed]),
        json!([
            true,
            [
                "atmosphere",
                "bloom",
                "haze",
                "highlights",
                "reflections",
                "sky_occlusion"
            ]
        ]),
        "night_changes_sections"
    );
    let rendering = &result["rendering"];
    engine_core::smoke_check!(
        number(&rendering["shadows"]["sky_maps"]) <= 26.0
            && number(&rendering["shadows"]["sky_drawn"]) <= 1.0,
        "sky_shadows_bounded",
        rendering["shadows"].clone(),
        json!({ "sky_maps": "<= 26", "sky_drawn": "<= 1" })
    );
    engine_core::smoke_eq!(
        effects(rendering),
        json!([true, true, false]),
        "night_effects"
    );
    engine_core::smoke_eq!(
        json!([
            rendering["haze"]["active"],
            rendering["reflections"]["active"],
            tick(&night)
        ]),
        json!([false, true, tick(after)]),
        "night_reflects_without_haze"
    );
    let dark = inspected(session, "Environment", "inspects_night")?;
    engine_core::smoke_check!(
        dark["atmosphere"] != look["atmosphere"] && dark["shadows"] == look["shadows"],
        "night_inspected",
        dark.clone(),
        json!({ "atmosphere": "changed", "shadows": look["shadows"] })
    );
    let capture = accepted(session, "capture", json!({}), "captures_night")?;
    engine_core::smoke_check!(
        capture["result"]["artifact"]["sha256"] != day["result"]["artifact"]["sha256"],
        "night_capture_differs",
        capture["result"]["artifact"].clone(),
        json!({ "sha256_differs_from": day["result"]["artifact"]["sha256"] })
    );
    if let (Some(light), Some(darker)) = (pixels(day)?, pixels(&capture)?) {
        let total = light.rgba.len() / 4;
        let changed = differing(&light, &darker, 8)?;
        engine_core::smoke_check!(
            changed > 1000 && changed <= total,
            "night_pixels_differ",
            json!({ "changed": changed, "total": total }),
            json!({ "changed": "> 1000" })
        );
    }
    Ok(dark)
}

fn candidate(source: &Value) -> Value {
    let mut candidate = source.clone();
    candidate["revision"] = json!(number(&source["revision"]) as u64 + 1);
    candidate["environment"] = json!({
        "atmosphere": { "preset": "dusk" },
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
        .and_then(|scene| scene.validate())
        .is_ok()
}

fn reloaded(session: &mut dyn SmokeSession, source: &Value, revision: u64) -> SmokeResult {
    let next = (revision + 1).to_string();
    let stale = named(
        "reload",
        json!({ "expected_revision": "0", "revision": next }),
    );
    let rejected = refused(session, "control", stale, "stale_reload_refused")?;
    engine_core::smoke_eq!(
        rejected["observation"]["scene"]["revision"].clone(),
        json!(revision.to_string()),
        "failed_reload"
    );
    let mut invalid = source.clone();
    invalid["revision"] = json!(revision + 1);
    invalid["entities"][0]["material"] = json!("agent-missing-material");
    engine_core::smoke_check!(
        !validates(&invalid),
        "source_validation",
        json!({ "material": "agent-missing-material" }),
        json!("refused")
    );
    let candidate = candidate(source);
    engine_core::smoke_check!(
        validates(&candidate),
        "candidate_validates",
        json!(false),
        json!(true)
    );
    session.source(&candidate.to_string());
    let args = json!({ "expected_revision": revision.to_string(), "revision": next });
    let reloaded = control(session, "reload", args, "reloads")?;
    engine_core::smoke_eq!(
        reloaded["observation"]["scene"]["revision"].clone(),
        json!(next),
        "reload_revision"
    );
    let rendering = &reloaded["result"]["rendering"];
    engine_core::smoke_eq!(
        json!([
            rendering["haze"]["active"],
            rendering["reflections"]["active"]
        ]),
        json!([true, false]),
        "reloaded_haze_without_reflections"
    );
    engine_core::smoke_check!(
        number(&rendering["local_lights"]["shaded"]) >= 1.0
            && number(&rendering["particles"]["drawn"]) > 0.0,
        "lights_particles_drawn",
        json!({ "local_lights": rendering["local_lights"], "particles": rendering["particles"] }),
        json!({ "shaded": ">= 1", "drawn": "> 0" })
    );
    engine_core::smoke_eq!(
        effects(rendering),
        json!([true, false, true]),
        "saved_effects"
    );
    Ok(())
}

fn saved(session: &mut dyn SmokeSession, look: &Value, night: &Value) -> SmokeResult {
    let body = json!({ "entity": PLAYER, "component": "Transform" });
    let player = accepted(session, "inspect", body, "inspects_renamed_player")?;
    engine_core::smoke_eq!(
        player["result"]["entity"]["name"].clone(),
        json!("Agent verified player"),
        "reload_renames_player"
    );
    let saved = inspected(session, "Environment", "inspects_saved_environment")?;
    engine_core::smoke_eq!(
        json!([
            number(&saved["ambient_occlusion"]["intensity"]),
            saved["bloom"],
            saved["shadows"],
            saved["haze"],
            number(&saved["reflections"]["intensity"]),
            saved["highlights"]["enabled"],
            saved["highlights"]["capped"],
        ]),
        json!([
            0.0,
            look["bloom"],
            look["shadows"],
            look["haze"],
            0.0,
            true,
            true
        ]),
        "saved_environment"
    );
    engine_core::smoke_check!(
        saved["atmosphere"] != night["atmosphere"] && saved["atmosphere"] != look["atmosphere"],
        "saved_atmosphere",
        saved["atmosphere"].clone(),
        json!("dusk")
    );
    let lamp = component(session, "agent-lamp", "LocalLight", "inspects_lamp")?;
    engine_core::smoke_eq!(
        json!([
            number(&lamp["intensity"]),
            number(&lamp["radius"]),
            number(&lamp["length"]),
            lamp["region"],
        ]),
        json!([20.0, 0.25, 1.5, "interior"]),
        "lamp_light"
    );
    let emitter = component(session, "agent-lamp", "ParticleEmitter", "inspects_emitter")?;
    engine_core::smoke_eq!(
        json!(number(&emitter["rate"])),
        json!(20.0),
        "lamp_particles"
    );
    let glass = component(session, "agent-glass-box", "Mesh3d", "inspects_glass")?;
    let opacity = number(&glass["material"]["opacity"]);
    engine_core::smoke_check!(
        (opacity - 0.4).abs() < 1e-6,
        "glass_opacity",
        json!(opacity),
        json!(0.4)
    );
    Ok(())
}

fn records(reply: &Value) -> Vec<Value> {
    reply["result"]["records"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn explained(session: &mut dyn SmokeSession) -> SmokeResult {
    let assets = query(session, "assets", json!({}), "lists_assets")?;
    engine_core::smoke_check!(
        number(&assets["result"]["meshes"]) >= 4.0,
        "assets_counted",
        assets["result"]["meshes"].clone(),
        json!(">= 4")
    );
    let costs = accepted(session, "profile", json!({ "frames": 4 }), "profiles")?;
    let result = &costs["result"];
    engine_core::smoke_check!(
        result["bricks"]
            .as_array()
            .is_some_and(|bricks| !bricks.is_empty())
            && number(&result["artifact"]["bytes"]) > 0.0,
        "profile_measures_bricks",
        json!({ "bricks": result["bricks"], "artifact": result["artifact"] }),
        json!({ "bricks": "measured", "artifact": "a file" })
    );
    let slow = accepted(
        session,
        "explain",
        json!({ "about": "frame", "top": 2 }),
        "explains_frame",
    )?;
    engine_core::smoke_check!(
        records(&slow)
            .iter()
            .any(|record| record["decided"] == "cost"),
        "frame_explained",
        json!(records(&slow)),
        json!({ "decided": "cost" })
    );
    let body = json!({ "about": "entity", "entity": PLAYER });
    let why = accepted(session, "explain", body, "explains_entity")?;
    engine_core::smoke_check!(
        records(&why).iter().any(|record| record["brick"] == "core"),
        "entity_explained",
        json!(records(&why)),
        json!({ "brick": "core" })
    );
    let body = json!({ "stats": ["tick"], "entities": [{ "entity": PLAYER, "component": "Transform" }], "frames": 2 });
    let watched = accepted(session, "watch", body, "watches")?;
    engine_core::smoke_check!(
        number(&watched["result"]["records"]) >= 1.0,
        "watch_streams",
        watched["result"].clone(),
        json!({ "records": ">= 1" })
    );
    let raycast = accepted(
        session,
        "describe",
        json!({ "name": "raycast" }),
        "describes_raycast",
    )?;
    let raycast = &raycast["result"];
    engine_core::smoke_eq!(
        json!([raycast["kind"], raycast["verb"], raycast["brick"]]),
        json!(["operation", "query", "collision"]),
        "operation_described"
    );
    let resumed = control(session, "resume", json!({}), "resumes")?;
    engine_core::smoke_eq!(
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
