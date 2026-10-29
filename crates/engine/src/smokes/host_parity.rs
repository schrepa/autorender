use engine_core::smoke::{SmokeResult, SmokeSession};
use engine_scene::SceneDocument;
use serde_json::{Value, json};

use super::requests::{
    FRAME_MS, accepted, control, frames, named, number, query, refused_with, reply, serve, status,
};
use crate::host::SceneLoad;

pub(crate) fn boxed(id: &str) -> Value {
    json!({
        "id": id, "revision": 1,
        "camera": { "position": [0, 2, 4], "target": [0, 0, 0], "fov_degrees": 60, "near": 0.1, "far": 100 },
        "materials": [{ "id": "plain", "color": [0.8, 0.3, 0.2, 1] }],
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": [{ "id": "box", "mesh": "box", "material": "plain", "transform": { "translation": [0, 0, 0] } }],
    })
}

type Case = (&'static str, &'static str, Value, bool);

fn answered(session: &mut dyn SmokeSession, cases: Vec<Case>) -> SmokeResult {
    for (check, verb, body, expect_ok) in cases {
        let reply = reply(session, verb, body);
        engine_core::smoke_check!(
            reply["ok"] == expect_ok,
            check,
            json!({ "ok": reply["ok"], "error": reply["error"] }),
            json!({ "ok": expect_ok })
        );
    }
    Ok(())
}

fn camera() -> Value {
    json!({ "position": [0, 30, 60], "target": [0, 0, 0], "vertical_fov_radians": 1, "near": 0.1, "far": 2000 })
}

fn accepted_cases() -> Vec<Case> {
    vec![
        ("status_answers", "query", named("status", json!({})), true),
        ("describe_lists", "describe", json!({}), true),
        (
            "describe_frame_stats",
            "describe",
            json!({ "name": "FrameStats" }),
            true,
        ),
        (
            "inspect_environment",
            "inspect",
            json!({ "component": "Environment" }),
            true,
        ),
        ("assets_listed", "query", named("assets", json!({})), true),
        (
            "input_taken",
            "control",
            named("input", json!({ "forward": 1, "yaw": 0.25 })),
            true,
        ),
        (
            "step_taken",
            "control",
            named("step", json!({ "ticks": 3 })),
            true,
        ),
        ("render_taken", "control", named("render", json!({})), true),
        (
            "present_taken",
            "control",
            named("present", json!({ "alpha": 0.5 })),
            true,
        ),
        ("capture_taken", "capture", json!({}), true),
        (
            "capture_presented",
            "capture",
            json!({ "alpha": 0.5 }),
            true,
        ),
        (
            "frames_captured",
            "control",
            named(
                "frames",
                json!({ "count": 3, "interval_ms": FRAME_MS, "capture": true }),
            ),
            true,
        ),
        (
            "benchmark_renders",
            "control",
            named(
                "benchmark",
                json!({ "mode": "render", "warmup": 1, "samples": 4 }),
            ),
            true,
        ),
        (
            "benchmark_steps",
            "control",
            named(
                "benchmark",
                json!({ "mode": "step", "warmup": 0, "samples": 2 }),
            ),
            true,
        ),
        (
            "culling_taken",
            "control",
            named("culling", json!({ "enabled": true })),
            true,
        ),
        (
            "environment_taken",
            "control",
            named("environment", json!({ "bloom": { "intensity": 0.5 } })),
            true,
        ),
        ("camera_taken", "control", named("camera", camera()), true),
        (
            "terrain_height_answers",
            "query",
            named("terrain_height", json!({ "points": [[0, 0]] })),
            true,
        ),
        (
            "distribution_answers",
            "query",
            named("distribution", json!({ "values": [4, 1, 3, 2] })),
            true,
        ),
        ("inspect_pages", "inspect", json!({ "limit": 3 }), true),
        ("profile_measures", "profile", json!({ "frames": 2 }), true),
        (
            "explain_frame",
            "explain",
            json!({ "about": "frame" }),
            true,
        ),
        (
            "explain_scene",
            "explain",
            json!({ "about": "scene" }),
            true,
        ),
        (
            "watch_streams",
            "watch",
            json!({ "stats": ["frame", "tick"], "entities": [{ "entity": "box", "component": "Transform" }], "frames": 2 }),
            true,
        ),
    ]
}

fn refused_cases(padding: String) -> Vec<Case> {
    let ray = json!({ "origin": [0, 50, 0], "direction": [0, -1, 0], "max_distance": 100 });
    vec![
        (
            "unknown_op_refused",
            "control",
            named("weather", json!({})),
            false,
        ),
        (
            "unknown_field_refused",
            "control",
            named("render", json!({ "sky": 1 })),
            false,
        ),
        (
            "unknown_verb_refused",
            "weather",
            named("status", json!({})),
            false,
        ),
        (
            "too_large_refused",
            "query",
            named("status", json!({ "padding": padding })),
            false,
        ),
        (
            "raycast_without_collision_refused",
            "query",
            named("raycast", ray),
            false,
        ),
        (
            "invalid_step_refused",
            "control",
            named("step", json!({ "ticks": 0 })),
            false,
        ),
        (
            "invalid_input_refused",
            "control",
            named("input", json!({ "forward": 2 })),
            false,
        ),
        (
            "invalid_benchmark_refused",
            "control",
            named("benchmark", json!({ "samples": 0 })),
            false,
        ),
        (
            "invalid_frames_refused",
            "control",
            named("frames", json!({ "count": 0, "interval_ms": FRAME_MS })),
            false,
        ),
        (
            "unknown_game_control_refused",
            "control",
            named("game", json!({ "name": "weather", "args": {} })),
            false,
        ),
        (
            "reload_to_missing_revision_refused",
            "control",
            named(
                "reload",
                json!({ "expected_revision": "1", "revision": "2" }),
            ),
            false,
        ),
        (
            "deep_profile_refused",
            "profile",
            json!({ "deep": { "duration_ms": 100 } }),
            false,
        ),
    ]
}

fn running_cases() -> Vec<Case> {
    vec![
        ("resume_taken", "control", named("resume", json!({})), true),
        (
            "step_while_running_refused",
            "control",
            named("step", json!({ "ticks": 1 })),
            false,
        ),
        ("capture_while_running_refused", "capture", json!({}), false),
        (
            "frames_while_running_refused",
            "control",
            named("frames", json!({ "count": 1, "interval_ms": FRAME_MS })),
            false,
        ),
        ("pause_taken", "control", named("pause", json!({})), true),
    ]
}

pub(crate) fn host_parity(session: &mut dyn SmokeSession) -> SmokeResult {
    serve(
        session,
        &boxed("host-parity-commands"),
        &["collision", "physics"],
    )?;
    let limits = status(session, "status_answers")?;
    let bytes = number(&limits["result"]["limits"]["request_bytes"]) as usize;
    answered(session, accepted_cases())?;
    answered(session, refused_cases("x".repeat(bytes)))?;
    answered(session, running_cases())
}

pub(crate) fn host_faults(session: &mut dyn SmokeSession) -> SmokeResult {
    serve(
        session,
        &boxed("host-parity-faults"),
        &["collision", "physics"],
    )?;
    let bloom = json!({ "bloom": { "intensity": 0.5 } });
    control(session, "environment", bloom, "bloom_before_the_fault")?;
    control(
        session,
        "debug_fault",
        json!({ "feature": "post" }),
        "debug_fault_taken",
    )?;
    frames(session, 30, "frames_after_the_fault")?;
    let faulted = status(session, "status_after_the_fault")?;
    engine_core::smoke_eq!(
        faulted["result"]["rendering"]["disabled_features"].clone(),
        json!(["post"]),
        "fault_disables_post"
    );
    answered(
        session,
        vec![
            (
                "fault_in_no_lazy_feature_refused",
                "control",
                named("debug_fault", json!({ "feature": "water" })),
                false,
            ),
            (
                "missing_asset_of_no_entity_refused",
                "control",
                named("debug_missing_asset", json!({ "entity": "nothing" })),
                false,
            ),
        ],
    )?;
    let loss = named("debug_device_loss", json!({}));
    refused_with(
        session,
        "control",
        loss,
        "render_failed",
        "device_loss_fails_render",
    )?;
    let render = named("render", json!({}));
    refused_with(
        session,
        "control",
        render,
        "render_failed",
        "render_after_the_loss_fails",
    )?;
    status(session, "status_after_the_loss")?;
    Ok(())
}

const ALPHA_PIXELS: [u8; 16] = [
    201, 77, 13, 3, 10, 250, 128, 128, 255, 255, 255, 0, 33, 66, 99, 254,
];
const ALPHA_URI: &str = "host-parity/alpha.png";

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut low, mut high) = (1_u32, 0_u32);
    for &byte in bytes {
        low = (low + u32::from(byte)) % 65_521;
        high = (high + low) % 65_521;
    }
    (high << 16) | low
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn png(width: u32, pixels: &[u8]) -> Vec<u8> {
    let height = pixels.len() as u32 / 4 / width;
    let mut rows = Vec::with_capacity(pixels.len() + height as usize);
    for row in pixels.chunks(width as usize * 4) {
        rows.push(0);
        rows.extend_from_slice(row);
    }
    let stored = rows.len() as u16;
    let mut deflated = vec![0x78, 0x01, 0x01];
    deflated.extend(stored.to_le_bytes());
    deflated.extend((!stored).to_le_bytes());
    deflated.extend(&rows);
    deflated.extend(adler32(&rows).to_be_bytes());
    let mut header = Vec::with_capacity(13);
    header.extend(width.to_be_bytes());
    header.extend(height.to_be_bytes());
    header.extend([8, 6, 0, 0, 0]);
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &deflated);
    chunk(&mut png, b"IEND", &[]);
    png
}

fn loaded(source: &str, image: &[u8]) -> crate::Result<engine_scene::SceneFiles> {
    let mut load = SceneLoad::new(source)?;
    loop {
        let reads = load.take_reads();
        if reads.is_empty() {
            break;
        }
        for read in reads {
            if let Some(bytes) = load.read(&read.uri, image.to_vec())? {
                load.decode(&read.uri, &bytes)?;
            }
        }
    }
    load.finish()
}

fn textured(session: &mut dyn SmokeSession) -> SmokeResult {
    let image = png(2, &ALPHA_PIXELS);
    let decoded = crate::host::decode_image(ALPHA_URI, &image);
    engine_core::smoke_eq!(
        json!(
            decoded
                .as_ref()
                .map(|image| (image.width, image.height, image.pixels.clone()))
                .ok()
        ),
        json!([2, 2, ALPHA_PIXELS]),
        "png_decodes_exactly"
    );
    let source = json!({
        "id": "host-parity-texture", "revision": 1,
        "camera": { "position": [0, 2, 4], "target": [0, 0, 0], "fov_degrees": 60, "near": 0.1, "far": 100 },
        "textures": [{ "id": "alpha", "uri": ALPHA_URI }],
        "materials": [{ "id": "alpha", "color": [1, 1, 1, 1], "texture": "alpha" }],
        "meshes": [{ "id": "panel", "source": { "kind": "cube" } }],
        "entities": [{ "id": "panel", "mesh": "panel", "material": "alpha", "transform": { "translation": [2, 0, 0] } }],
    })
    .to_string();
    let files = loaded(&source, &image);
    engine_core::smoke_check!(
        files.is_ok(),
        "scene_files_load",
        json!(
            files
                .as_ref()
                .err()
                .map(|error| json!({ "code": error.code, "path": error.path }))
        ),
        json!(null)
    );
    let world = session.app().world_mut();
    let spawned = SceneDocument::from_json(&source)
        .and_then(|scene| scene.instantiate_with(world, &files.unwrap_or_default()));
    engine_core::smoke_check!(
        spawned.is_ok(),
        "textured_scene_instantiates",
        json!(
            spawned
                .as_ref()
                .err()
                .map(|error| json!({ "code": error.code, "path": error.path }))
        ),
        json!(null)
    );
    let digest = query(session, "digest", json!({}), "digests")?;
    engine_core::smoke_eq!(
        digest["result"]["assets"]["textures"]["count"].clone(),
        json!(1),
        "png_texture_resident"
    );
    Ok(())
}

pub(crate) fn host_images(session: &mut dyn SmokeSession) -> SmokeResult {
    serve(session, &boxed("host-parity"), &[])?;
    let body = named("debug_missing_asset", json!({ "entity": "box" }));
    let removed = accepted(session, "control", body, "missing_asset_removed")?;
    let result = &removed["result"];
    engine_core::smoke_eq!(
        json!([
            result["temporary"],
            result["entities"],
            result["rendering"]["missing_assets"]
        ]),
        json!([true, 1, 1]),
        "missing_asset_counted"
    );
    let after = status(session, "status_after_removal")?;
    engine_core::smoke_eq!(
        after["result"]["rendering"]["missing_assets"].clone(),
        json!(1),
        "status_counts_missing_asset"
    );
    let listed = query(session, "assets", json!({}), "lists_assets")?;
    let listed = &listed["result"];
    let entry = &listed["missing"][0];
    engine_core::smoke_eq!(
        json!([
            listed["missing_total"],
            entry["entity"]["id"],
            entry["entity"]["scene"],
            entry["draws"],
            entry["missing"],
            entry["position"],
        ]),
        json!([1, "box", "host-parity", "mesh", "mesh", [0.0, 0.0, 0.0]]),
        "missing_mesh_listed"
    );
    let unknown = named("debug_missing_asset", json!({ "entity": "nothing" }));
    let refused = refused_with(
        session,
        "control",
        unknown,
        "unknown_entity",
        "unknown_entity_refused",
    )?;
    engine_core::smoke_eq!(
        refused["error"]["path"].clone(),
        json!("$.args.entity"),
        "unknown_entity_located"
    );
    textured(session)
}
