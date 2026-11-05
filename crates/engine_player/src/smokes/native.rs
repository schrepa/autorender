use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_standard_smokes::requests::{
    Asked, accepted, control, environment, frames, inspected, named, names, number, query,
    refused_with, seen, serve, status, tick,
};
use serde_json::{Value, json};

use super::agent::{MESHES, PLAYER, garden};
use super::pixels::{differing, pixels};

const CRYSTAL: &str = "crystal-6";

pub(crate) fn native(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let source = garden()?;
    let revision = source["revision"].as_u64().unwrap_or_default().to_string();
    serve(session, &source)?;
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
    let wanted = json!({ "paused": true, "adapter": { "name": "named", "backend": "named" } });
    smoke_eq!(
        seen(
            result["paused"] == true
                && adapter["name"]
                    .as_str()
                    .is_some_and(|name| !name.is_empty())
                && adapter["backend"]
                    .as_str()
                    .is_some_and(|backend| !backend.is_empty()),
            json!({ "paused": result["paused"], "adapter": adapter }),
            &wanted,
        ),
        wanted,
        "adapter_named"
    );
    let wanted = json!({ "features": ["dev"] });
    smoke_eq!(
        seen(
            names(&result["build"]["features"])
                .iter()
                .any(|feature| feature == "dev"),
            result["build"].clone(),
            &wanted,
        ),
        wanted,
        "build_features"
    );
    let listing = accepted(session, "describe", json!({}), "lists")?;
    smoke_eq!(
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
            brick["name"] == MESHES
                && brick["explains"]
                    .as_array()
                    .is_some_and(|explains| !explains.is_empty())
        });
    let wanted = json!({ "name": MESHES, "explains": "listed" });
    smoke_eq!(
        seen(culling, listing["result"]["bricks"].clone(), &wanted),
        wanted,
        "bricks_listed_with_explainers"
    );
    let antialiasing = inspected(session, "temporal_antialiasing", "inspects_taa")?;
    smoke_eq!(
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

fn resolved(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let ordinary = frames(session, 4, "frames_run")?["result"]["rendering"].clone();
    smoke_eq!(
        ordinary["temporal_antialiasing_stats"]["active"].clone(),
        json!(true),
        "ordinary_frames_resolve_temporally"
    );
    let [width, height, render_width, render_height] = sized(&ordinary);
    let wanted = json!({ "render": [width, height], "dropped_pixels": 0 });
    smoke_eq!(
        seen(
            width > 0.0
                && (render_width, render_height) == (width, height)
                && number(&ordinary["resolution"]["dropped_pixels"]) == 0.0,
            ordinary["resolution"].clone(),
            &wanted,
        ),
        wanted,
        "full_resolution"
    );
    smoke_eq!(
        ordinary["dynamic_resolution"]["active"].clone(),
        json!(false),
        "offscreen_frames_keep_size"
    );
    let half = json!({ "resolution": { "scale": 0.5 } });
    environment(session, half, "scales_resolution")?;
    let scaled = frames(session, 4, "scaled_frames_run")?["result"]["rendering"].clone();
    let [width, height, render_width, render_height] = sized(&scaled);
    let wanted = json!({ "render": [width / 2.0, height / 2.0] });
    smoke_eq!(
        seen(
            (render_width * 2.0 - width).abs() <= 1.0
                && (render_height * 2.0 - height).abs() <= 1.0,
            scaled["resolution"].clone(),
            &wanted,
        ),
        wanted,
        "resolution_scaled"
    );
    let full = json!({ "resolution": { "scale": 1 } });
    environment(session, full, "restores_resolution")?;
    Ok(())
}

fn lit(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let day = accepted(session, "capture", json!({}), "captures")?;
    let artifact = &day["result"]["artifact"];
    let wanted = json!({ "mime_type": "image/png", "bytes": "> 1000" });
    smoke_eq!(
        seen(
            artifact["mime_type"] == "image/png" && number(&artifact["bytes"]) > 1000.0,
            artifact.clone(),
            &wanted,
        ),
        wanted,
        "capture_holds_image"
    );
    let night = json!({
        "time_of_day": { "hours": 23.0 },
        "surface_light": { "rim": 0.8, "emission": 1.0 },
        "grading": { "exposure": 1.35 },
    });
    environment(session, night, "night_falls")?;
    let dark = accepted(session, "capture", json!({}), "captures_night")?;
    if let (Some(light), Some(darker)) = (pixels(&day)?, pixels(&dark)?) {
        let total = light.rgba.len() / 4;
        let changed = differing(&light, &darker, 8)?;
        let wanted = json!({ "changed": total / 10 });
        smoke_eq!(
            seen(
                changed > total / 10,
                json!({ "changed": changed, "total": total }),
                &wanted,
            ),
            wanted,
            "night_differs"
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

fn measured(session: &mut dyn SmokeSession, status: &Value) -> Result<(), SmokeFailure> {
    let ticked = tick(&engine_standard_smokes::requests::status(
        session,
        "status_before_step",
    )?);
    let stepped = control(session, "step", json!({ "ticks": 12 }), "steps")?;
    smoke_eq!(
        json!(tick(&stepped).wrapping_sub(ticked)),
        json!(12),
        "precise_step"
    );
    let args = json!({ "mode": "render", "warmup": 2, "samples": 8 });
    let benchmark = control(session, "benchmark", args, "benchmarks")?;
    let benchmark = &benchmark["result"]["benchmark"];
    let wanted = json!({ "median": "> 0" });
    smoke_eq!(
        seen(
            number(&benchmark["frame_interval_ms"]["median"]) > 0.0
                && number(&benchmark["cpu_ms"]["total"]["median"]) > 0.0,
            json!({ "frame_interval_ms": benchmark["frame_interval_ms"], "cpu_ms": benchmark["cpu_ms"]["total"] }),
            &wanted,
        ),
        wanted,
        "benchmark_measures"
    );
    smoke_eq!(
        benchmark["environment"]["adapter"]["name"].clone(),
        status["result"]["adapter"]["name"].clone(),
        "benchmark_names_adapter"
    );
    let costs = accepted(session, "profile", json!({ "frames": 8 }), "profiles")?;
    let costs = &costs["result"];
    let wanted = json!({ "frames": 8, "cpu_ms": "> 0", "bricks": "measured" });
    smoke_eq!(
        seen(
            costs["bricks"]
                .as_array()
                .is_some_and(|bricks| !bricks.is_empty())
                && number(&costs["cpu_ms"]) > 0.0
                && number(&costs["frames"]) == 8.0
                && number(&costs["artifact"]["bytes"]) > 0.0,
            json!({ "frames": costs["frames"], "cpu_ms": costs["cpu_ms"], "bricks": costs["bricks"] }),
            &wanted,
        ),
        wanted,
        "profile_measures_bricks"
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

fn look(
    session: &mut dyn SmokeSession,
    position: [f64; 3],
    target: [f64; 3],
) -> Result<(), SmokeFailure> {
    let args = json!({ "position": position, "target": target, "vertical_fov_radians": 0.8, "near": 0.1, "far": 150 });
    control(session, "camera", args, "camera_moves")?;
    Ok(())
}

fn explained(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let body = json!({ "about": "frame", "top": 3 });
    let slow = accepted(session, "explain", body, "explains_frame")?;
    let wanted = json!({ "decided": "cost", "cpu_ms": "number" });
    smoke_eq!(
        seen(
            records(&slow)
                .iter()
                .any(|record| record["decided"] == "cost" && record["cpu_ms"].is_number()),
            json!(records(&slow)),
            &wanted,
        ),
        wanted,
        "frame_costs_explained"
    );
    let about = json!({ "about": "entity", "entity": CRYSTAL });
    look(session, [0.0, 20.0, 60.0], [0.0, 20.0, 120.0])?;
    let culled = read_back(session, &about, "explains_culled")?;
    let core = culled.iter().find(|record| record["brick"] == "core");
    let wanted = json!({ "core": "culled", MESHES: "culled" });
    smoke_eq!(
        seen(
            core.is_some_and(|record| record["decided"] == "culled")
                && culled
                    .iter()
                    .any(|record| record["brick"] == MESHES && record["decided"] == "culled"),
            json!(culled),
            &wanted,
        ),
        wanted,
        "culled_entity_explained"
    );
    look(session, [17.0, 21.0, 26.0], [0.0, 0.0, 0.0])?;
    let drawn = read_back(session, &about, "explains_drawn")?;
    let wanted = json!({ "brick": MESHES, "level": "integer" });
    smoke_eq!(
        seen(
            drawn
                .iter()
                .any(|record| record["brick"] == MESHES && record["level"].is_u64()),
            json!(drawn),
            &wanted,
        ),
        wanted,
        "drawn_entity_explained"
    );
    let scene = accepted(
        session,
        "explain",
        json!({ "about": "scene" }),
        "explains_scene",
    )?;
    smoke_eq!(json!(records(&scene)), json!([]), "scene_explained");
    let body = json!({ "stats": ["tick"], "entities": [{ "entity": PLAYER, "component": "Transform" }], "frames": 3 });
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

fn digested(session: &mut dyn SmokeSession, revision: &str) -> Result<(), SmokeFailure> {
    let digest = digest_after(session, revision, 12)?;
    let wanted = json!("entities, transforms and meshes counted");
    smoke_eq!(
        seen(
            number(&digest["entities"]) > 0.0
                && number(&digest["components"]["Transform"]["count"]) > 0.0
                && number(&digest["assets"]["meshes"]["count"]) > 0.0,
            json!({ "entities": digest["entities"], "assets": digest["assets"] }),
            &wanted,
        ),
        wanted,
        "digest_counts"
    );
    let again = digest_after(session, revision, 12)?;
    smoke_eq!(
        again["components"].clone(),
        digest["components"].clone(),
        "same_ticks_same_digest"
    );
    let farther = digest_after(session, revision, 24)?;
    let wanted = json!({ "differs_from": digest["components"] });
    smoke_eq!(
        seen(
            farther["components"] != digest["components"],
            farther["components"].clone(),
            &wanted,
        ),
        wanted,
        "walking_changes_digest"
    );
    Ok(())
}

fn resumed(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
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
    let wanted = json!({ "above": resumed_at });
    smoke_eq!(
        seen(ticked > resumed_at, json!(ticked), &wanted),
        wanted,
        "ticks_run_while_resumed"
    );
    let paused = control(session, "pause", json!({}), "pauses")?;
    smoke_eq!(
        json!([paused["result"]["paused"], tick(&paused) > resumed_at]),
        json!([true, true]),
        "pause_keeps_ticks"
    );
    Ok(())
}
