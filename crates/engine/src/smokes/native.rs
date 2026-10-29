use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use serde_json::{Value, json};

use super::agent::{PLAYER, garden};
use super::requests::{
    Asked, accepted, control, differing, environment, frames, inspected, named, names, number,
    pixels, query, refused_with, serve, status, tick,
};

const CRYSTAL: &str = "crystal-6";

pub(crate) fn native(session: &mut dyn SmokeSession) -> SmokeResult {
    let source = garden()?;
    let revision = source["revision"].as_u64().unwrap_or_default().to_string();
    serve(session, &source, &[])?;
    let taa = json!({ "temporal_antialiasing": { "enabled": true } });
    environment(session, taa, "turns_taa_on")?;
    let status = described(session)?;
    resolved(session)?;
    lit(session)?;
    measured(session, &status)?;
    explained(session)?;
    digested(session, &revision)?;
    resumed(session)
}

fn described(session: &mut dyn SmokeSession) -> Asked {
    let status = status(session, "status_answers")?;
    let result = &status["result"];
    let adapter = &result["adapter"];
    engine_core::smoke_check!(
        result["paused"] == true
            && adapter["name"]
                .as_str()
                .is_some_and(|name| !name.is_empty())
            && adapter["backend"]
                .as_str()
                .is_some_and(|backend| !backend.is_empty()),
        "adapter_named",
        json!({ "paused": result["paused"], "adapter": adapter }),
        json!({ "paused": true, "adapter": { "name": "named", "backend": "named" } })
    );
    engine_core::smoke_check!(
        names(&result["build"]["features"])
            .iter()
            .any(|feature| feature == "dev"),
        "build_features",
        result["build"].clone(),
        json!({ "features": ["dev"] })
    );
    let listing = accepted(session, "describe", json!({}), "lists")?;
    engine_core::smoke_eq!(
        json!(names(&listing["result"]["verbs"])),
        json!([
            "describe", "inspect", "query", "control", "capture", "profile", "explain", "watch",
            "check"
        ]),
        "nine_verbs"
    );
    let culling = listing["result"]["bricks"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|brick| {
            brick["name"] == "culling"
                && brick["explains"]
                    .as_array()
                    .is_some_and(|explains| !explains.is_empty())
        });
    engine_core::smoke_check!(
        culling,
        "bricks_listed_with_explainers",
        listing["result"]["bricks"].clone(),
        json!({ "name": "culling", "explains": "listed" })
    );
    let antialiasing = inspected(session, "TemporalAntialiasing", "inspects_taa")?;
    engine_core::smoke_eq!(
        antialiasing["enabled"].clone(),
        json!(true),
        "taa_inspected"
    );
    Ok(status)
}

fn sized(rendering: &Value) -> [f64; 4] {
    let resolution = &rendering["resolution"];
    [
        "display_width",
        "display_height",
        "render_width",
        "render_height",
    ]
    .map(|field| number(&resolution[field]))
}

fn resolved(session: &mut dyn SmokeSession) -> SmokeResult {
    let ordinary = frames(session, 4, "frames_run")?["result"]["rendering"].clone();
    engine_core::smoke_eq!(
        ordinary["temporal_antialiasing"]["active"].clone(),
        json!(true),
        "ordinary_frames_resolve_temporally"
    );
    let [width, height, render_width, render_height] = sized(&ordinary);
    engine_core::smoke_check!(
        width > 0.0
            && (render_width, render_height) == (width, height)
            && number(&ordinary["resolution"]["dropped_pixels"]) == 0.0,
        "full_resolution",
        ordinary["resolution"].clone(),
        json!({ "render": [width, height], "dropped_pixels": 0 })
    );
    engine_core::smoke_eq!(
        ordinary["dynamic_resolution"]["active"].clone(),
        json!(false),
        "offscreen_frames_keep_size"
    );
    let half = json!({ "resolution": { "scale": 0.5 } });
    environment(session, half, "scales_resolution")?;
    let scaled = frames(session, 4, "scaled_frames_run")?["result"]["rendering"].clone();
    let [width, height, render_width, render_height] = sized(&scaled);
    engine_core::smoke_check!(
        (render_width * 2.0 - width).abs() <= 1.0 && (render_height * 2.0 - height).abs() <= 1.0,
        "resolution_scaled",
        scaled["resolution"].clone(),
        json!({ "render": [width / 2.0, height / 2.0] })
    );
    let full = json!({ "resolution": { "scale": 1 } });
    environment(session, full, "restores_resolution")?;
    Ok(())
}

fn lit(session: &mut dyn SmokeSession) -> SmokeResult {
    let day = accepted(session, "capture", json!({}), "captures")?;
    let artifact = &day["result"]["artifact"];
    engine_core::smoke_check!(
        artifact["mime_type"] == "image/png" && number(&artifact["bytes"]) > 1000.0,
        "capture_holds_image",
        artifact.clone(),
        json!({ "mime_type": "image/png", "bytes": "> 1000" })
    );
    let night = json!({ "atmosphere": { "preset": "night" } });
    environment(session, night, "night_falls")?;
    let dark = accepted(session, "capture", json!({}), "captures_night")?;
    if let (Some(light), Some(darker)) = (pixels(&day)?, pixels(&dark)?) {
        let total = light.rgba.len() / 4;
        let changed = differing(&light, &darker, 8)?;
        engine_core::smoke_check!(
            changed > total / 10,
            "night_differs",
            json!({ "changed": changed, "total": total }),
            json!({ "changed": total / 10 })
        );
    }
    let weather = named("game", json!({ "name": "weather", "args": {} }));
    refused_with(
        session,
        "control",
        weather,
        "unsupported_command",
        "unknown_game_control",
    )?;
    Ok(())
}

fn measured(session: &mut dyn SmokeSession, status: &Value) -> SmokeResult {
    let ticked = tick(&super::requests::status(session, "status_before_step")?);
    let stepped = control(session, "step", json!({ "ticks": 12 }), "steps")?;
    engine_core::smoke_eq!(
        json!(tick(&stepped).wrapping_sub(ticked)),
        json!(12),
        "precise_step"
    );
    let args = json!({ "mode": "render", "warmup": 2, "samples": 8 });
    let benchmark = control(session, "benchmark", args, "benchmarks")?;
    let benchmark = &benchmark["result"]["benchmark"];
    engine_core::smoke_check!(
        number(&benchmark["frame_interval_ms"]["median"]) > 0.0
            && number(&benchmark["cpu_ms"]["total"]["median"]) > 0.0,
        "benchmark_measures",
        json!({ "frame_interval_ms": benchmark["frame_interval_ms"], "cpu_ms": benchmark["cpu_ms"]["total"] }),
        json!({ "median": "> 0" })
    );
    engine_core::smoke_eq!(
        benchmark["environment"]["adapter"]["name"].clone(),
        status["result"]["adapter"]["name"].clone(),
        "benchmark_names_adapter"
    );
    let costs = accepted(session, "profile", json!({ "frames": 8 }), "profiles")?;
    let costs = &costs["result"];
    engine_core::smoke_check!(
        costs["bricks"]
            .as_array()
            .is_some_and(|bricks| !bricks.is_empty())
            && number(&costs["cpu_ms"]) > 0.0
            && number(&costs["frames"]) == 8.0
            && number(&costs["artifact"]["bytes"]) > 0.0,
        "profile_measures_bricks",
        json!({ "frames": costs["frames"], "cpu_ms": costs["cpu_ms"], "bricks": costs["bricks"] }),
        json!({ "frames": 8, "cpu_ms": "> 0", "bricks": "measured" })
    );
    Ok(())
}

fn records(reply: &Value) -> Vec<Value> {
    reply["result"]["records"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn read_back(
    session: &mut dyn SmokeSession,
    about: &Value,
    check: &'static str,
) -> Result<Vec<Value>, SmokeFailure> {
    Ok(records(&accepted(
        session,
        "explain",
        about.clone(),
        check,
    )?))
}

fn look(session: &mut dyn SmokeSession, position: [f64; 3], target: [f64; 3]) -> SmokeResult {
    let args = json!({ "position": position, "target": target, "vertical_fov_radians": 0.8, "near": 0.1, "far": 150 });
    control(session, "camera", args, "camera_moves")?;
    Ok(())
}

fn explained(session: &mut dyn SmokeSession) -> SmokeResult {
    let body = json!({ "about": "frame", "top": 3 });
    let slow = accepted(session, "explain", body, "explains_frame")?;
    engine_core::smoke_check!(
        records(&slow)
            .iter()
            .any(|record| record["decided"] == "cost" && record["cpu_ms"].is_number()),
        "frame_costs_explained",
        json!(records(&slow)),
        json!({ "decided": "cost", "cpu_ms": "number" })
    );
    let about = json!({ "about": "entity", "entity": CRYSTAL });
    look(session, [0.0, 20.0, 60.0], [0.0, 20.0, 120.0])?;
    let culled = read_back(session, &about, "explains_culled")?;
    let core = culled.iter().find(|record| record["brick"] == "core");
    engine_core::smoke_check!(
        core.is_some_and(|record| record["decided"] == "culled")
            && culled
                .iter()
                .any(|record| record["brick"] == "culling" && record["decided"] == "culled"),
        "culled_entity_explained",
        json!(culled),
        json!({ "core": "culled", "culling": "culled" })
    );
    look(session, [17.0, 21.0, 26.0], [0.0, 0.0, 0.0])?;
    let drawn = read_back(session, &about, "explains_drawn")?;
    engine_core::smoke_check!(
        drawn
            .iter()
            .any(|record| record["brick"] == "level_of_detail" && record["level"].is_u64()),
        "drawn_entity_explained",
        json!(drawn),
        json!({ "brick": "level_of_detail", "level": "integer" })
    );
    let scene = accepted(
        session,
        "explain",
        json!({ "about": "scene" }),
        "explains_scene",
    )?;
    engine_core::smoke_eq!(json!(records(&scene)), json!([]), "scene_explained");
    let body = json!({ "stats": ["tick"], "entities": [{ "entity": PLAYER, "component": "Transform" }], "frames": 3 });
    let watched = accepted(session, "watch", body, "watches")?;
    engine_core::smoke_check!(
        number(&watched["result"]["records"]) >= 1.0,
        "watch_streams",
        watched["result"].clone(),
        json!({ "records": ">= 1" })
    );
    Ok(())
}

fn digest_after(session: &mut dyn SmokeSession, revision: &str, ticks: u32) -> Asked {
    let args = json!({ "expected_revision": revision, "revision": revision });
    control(session, "reload", args, "reloads")?;
    control(session, "input", json!({ "forward": 1 }), "walks")?;
    control(session, "step", json!({ "ticks": ticks }), "steps_walking")?;
    control(session, "input", json!({}), "stops")?;
    Ok(query(session, "digest", json!({}), "digests")?["result"].clone())
}

fn digested(session: &mut dyn SmokeSession, revision: &str) -> SmokeResult {
    let digest = digest_after(session, revision, 12)?;
    engine_core::smoke_check!(
        number(&digest["entities"]) > 0.0
            && number(&digest["components"]["Transform"]["count"]) > 0.0
            && number(&digest["assets"]["meshes"]["count"]) > 0.0,
        "digest_counts",
        json!({ "entities": digest["entities"], "assets": digest["assets"] }),
        json!("entities, transforms and meshes counted")
    );
    let again = digest_after(session, revision, 12)?;
    engine_core::smoke_eq!(
        again["components"].clone(),
        digest["components"].clone(),
        "same_ticks_same_digest"
    );
    let farther = digest_after(session, revision, 24)?;
    engine_core::smoke_check!(
        farther["components"] != digest["components"],
        "walking_changes_digest",
        farther["components"].clone(),
        json!({ "differs_from": digest["components"] })
    );
    Ok(())
}

fn resumed(session: &mut dyn SmokeSession) -> SmokeResult {
    let resumed_at = tick(&status(session, "status_before_resume")?);
    control(session, "resume", json!({}), "resumes")?;
    let mut ticked = resumed_at;
    for _ in 0..3000 {
        if !session.frame() {
            break;
        }
        ticked = tick(&status(session, "status_while_resumed")?);
        if ticked > resumed_at {
            break;
        }
    }
    engine_core::smoke_check!(
        ticked > resumed_at,
        "ticks_run_while_resumed",
        json!(ticked),
        json!({ "above": resumed_at })
    );
    let paused = control(session, "pause", json!({}), "pauses")?;
    engine_core::smoke_eq!(
        json!([paused["result"]["paused"], tick(&paused) > resumed_at]),
        json!([true, true]),
        "pause_keeps_ticks"
    );
    Ok(())
}
