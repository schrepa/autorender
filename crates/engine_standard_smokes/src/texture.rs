use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::{FRAME_MS, seen};
use super::scene_kit::{ask, capture, component, control, load, query, reload, render, residency};

const TEXTURES: [(&str, u32, u32); 2] = [("bark", 16, 16), ("leaf", 8, 8)];

fn pixels(width: u32, height: u32, seed: u32, inverted: bool) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let shade = 60 + (x * 11 + y * 7 + seed * 29) % 140;
            let light = ((x >> 1) + (y >> 1)) % 2 == 0;
            let rgb = if light {
                [shade + 40, shade + 20, shade]
            } else {
                [shade, shade + 30, shade / 2]
            };
            for channel in rgb {
                let channel = u8::try_from(channel).unwrap_or(u8::MAX);
                pixels.push(if inverted { 255 - channel } else { channel });
            }
            pixels.push(255);
        }
    }
    pixels
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let byte = |index: usize| u32::from(chunk.get(index).copied().unwrap_or(0));
        let bits = (byte(0) << 16) | (byte(1) << 8) | byte(2);
        for index in 0..4 {
            if index <= chunk.len() {
                encoded.push(char::from(
                    TABLE[((bits >> (18 - 6 * index)) & 63) as usize],
                ));
            } else {
                encoded.push('=');
            }
        }
    }
    encoded
}

fn fixture(inverted: bool) -> Value {
    let textures: Vec<Value> = TEXTURES
        .iter()
        .enumerate()
        .map(|(seed, &(id, width, height))| {
            let rgba8 = base64(&pixels(width, height, seed as u32, inverted));
            json!({"id": id, "width": width, "height": height, "rgba8": rgba8})
        })
        .collect();
    json!({
        "id": "texture-fixture", "revision": 1,
        "environment": {"sky_occlusion": {"enabled": false}},
        "camera": {"position": [0, 6, 10], "target": [0, 1, 0], "fov_degrees": 60, "near": 0.1, "far": 300},
        "textures": textures,
        "materials": [
            {"id": "ground", "color": [0.5, 0.55, 0.45, 1]},
            {"id": "bark", "color": [1, 1, 1, 1], "texture": "bark"},
            {"id": "leaf", "color": [1, 1, 1, 1], "texture": "leaf"},
        ],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}, {"id": "gem", "source": {"kind": "octahedron"}}],
        "entities": [
            {"id": "ground", "mesh": "box", "material": "ground", "transform": {"translation": [0, -0.5, 0], "scale": [20, 1, 20]}},
            {"id": "player", "mesh": "box", "material": "ground", "transform": {"translation": [0, 0.5, 4]}},
            {"id": "trunk", "mesh": "box", "material": "bark", "transform": {"translation": [-1.5, 1.5, 0], "scale": [1, 3, 1]}},
            {"id": "crown", "mesh": "gem", "material": "leaf", "transform": {"translation": [-1.5, 3.8, 0], "scale": [2.4, 2, 2.4]}},
            {"id": "stump", "mesh": "box", "material": "bark", "transform": {"translation": [2, 0.4, -1], "scale": [1.4, 0.8, 1.4]}},
        ],
    })
}

fn texture_bytes() -> u64 {
    TEXTURES
        .iter()
        .map(|&(_, width, height)| u64::from(width * height * 4))
        .sum()
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let source = fixture(false);
    load(session, &source)?;
    ask(session, "control", json!({"name": "pause"}))?;
    let rendering = render(session)?;
    settled(session, rendering)?;
    let initial = residency(session)?;
    smoke_eq!(
        json!([
            initial["textures"],
            initial["texture_cpu"],
            initial["gpu_textures"],
            initial["texture_gpu"]
        ]),
        json!([
            TEXTURES.len(),
            texture_bytes(),
            TEXTURES.len(),
            texture_bytes()
        ]),
        "authored_texture_readiness"
    );
    textured_entity(session)?;
    for _ in 0..3 {
        let warmed = render(session)?;
        smoke_eq!(
            json!(warmed["texture_upload_bytes"]),
            json!(0),
            "warm_upload_reuse"
        );
    }
    control(session, "input", json!({"forward": 1, "keys": ["Space"]}))?;
    control(session, "step", json!({"ticks": 6}))?;
    let input = ask(session, "inspect", json!({"component": "Input"}))?["value"].clone();
    smoke_eq!(
        json!({"forward": input["move_forward"], "held": input["held"]}),
        json!({"forward": 1.0, "held": [0]}),
        "input_reaches_engine"
    );
    control(session, "input", json!({"forward": 0}))?;
    let player = component(session, "player", "Transform")?;
    let mut invalid = source.clone();
    invalid["textures"][0]["rgba8"] = json!("!invalid-base64!");
    let refused = reload(session, &invalid).err().unwrap_or_default();
    let wanted = json!({"path": "$.textures[0]"});
    smoke_eq!(
        seen(
            refused["path"]
                .as_str()
                .is_some_and(|path| path.starts_with("$.textures[0]")),
            json!(refused),
            &wanted,
        ),
        wanted,
        "invalid_reload_path"
    );
    smoke_eq!(
        component(session, "player", "Transform")?,
        player,
        "invalid_reload_preservation"
    );
    smoke_eq!(
        residency(session)?,
        initial.clone(),
        "invalid_reload_keeps_assets"
    );
    replacement(session, &source, &initial)
}

fn textured_entity(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let page = ask(
        session,
        "inspect",
        json!({"component": "Mesh3d", "limit": 100}),
    )?;
    let textured = page["entities"].as_array().and_then(|rows| {
        rows.iter()
            .find(|row| {
                row["components"]["Mesh3d"]["textures"]
                    .as_array()
                    .is_some_and(|textures| textures.iter().any(|texture| texture["alive"] == true))
            })
            .and_then(|row| row["id"].as_str().map(str::to_owned))
    });
    let wanted = json!("a textured entity");
    smoke_eq!(
        seen(textured.is_some(), json!(page["entities"]), &wanted),
        wanted,
        "textured_material_used"
    );
    let id = textured.unwrap_or_default();
    let view = component(session, &id, "Mesh3d")?;
    let texture = &view["textures"][0];
    let authored = TEXTURES
        .iter()
        .any(|&(_, width, height)| texture["width"] == width && texture["height"] == height);
    let wanted = json!({"alive": true, "gpu_resident": true});
    smoke_eq!(
        seen(
            texture["alive"] == true && texture["gpu_resident"] == true && authored,
            json!(texture),
            &wanted,
        ),
        wanted,
        "per_texture_gpu_residency"
    );
    Ok(())
}

fn replacement(
    session: &mut dyn SmokeSession,
    source: &Value,
    initial: &Value,
) -> Result<(), SmokeFailure> {
    reloaded(session, source)?;
    let baseline = capture(session)?;
    let changed = fixture(true);
    let uploaded = reloaded(session, &changed)?;
    smoke_eq!(
        json!(uploaded),
        json!(texture_bytes()),
        "replacement_uploads_once"
    );
    smoke_eq!(
        residency(session)?,
        initial.clone(),
        "replacement_keeps_residency"
    );
    let changed_capture = capture(session)?;
    let wanted = json!({"not": baseline});
    smoke_eq!(
        seen(changed_capture != baseline, json!(changed_capture), &wanted),
        wanted,
        "changed_pixels_render"
    );
    for _ in 0..3 {
        reloaded(session, &changed)?;
        smoke_eq!(
            residency(session)?,
            initial.clone(),
            "repeated_reload_residency"
        );
        let warmed = render(session)?;
        smoke_eq!(
            json!(warmed["texture_upload_bytes"]),
            json!(0),
            "repeated_reload_reuses_uploads"
        );
    }
    smoke_eq!(
        capture(session)?,
        changed_capture,
        "equivalent_replacement_image"
    );
    let mut untextured = changed.clone();
    untextured["textures"] = json!([]);
    if let Some(materials) = untextured["materials"].as_array_mut() {
        for material in materials {
            if let Some(fields) = material.as_object_mut() {
                fields.remove("texture");
            }
        }
    }
    reloaded(session, &untextured)?;
    let retired = query(session, "assets", json!({}))?;
    let cpu = initial["cpu"]
        .as_u64()
        .unwrap_or(0)
        .saturating_sub(texture_bytes());
    smoke_eq!(
        json!({
            "textures": retired["textures"], "texture_cpu": retired["texture_cpu_bytes"],
            "rows": retired["texture_assets"], "gpu": retired["rendering"]["resident_textures"],
            "gpu_bytes": retired["rendering"]["resident_texture_bytes"], "cpu": retired["resident_cpu_bytes"],
        }),
        json!({"textures": 0, "texture_cpu": 0, "rows": [], "gpu": 0, "gpu_bytes": 0, "cpu": cpu}),
        "texture_retirement"
    );
    capture(session)?;
    Ok(())
}

fn reloaded(session: &mut dyn SmokeSession, document: &Value) -> Result<u64, SmokeFailure> {
    let reloaded = reload(session, document);
    let wanted = json!(null);
    smoke_eq!(
        seen(reloaded.is_ok(), json!(reloaded.as_ref().err()), &wanted),
        wanted,
        "reload_accepted"
    );
    let rendering = reloaded.unwrap_or_default()["rendering"].clone();
    settled(session, rendering)
}

fn settled(session: &mut dyn SmokeSession, mut rendering: Value) -> Result<u64, SmokeFailure> {
    let uploads = |rendering: &Value| rendering["texture_upload_bytes"].as_u64().unwrap_or(0);
    let mut uploaded = uploads(&rendering);
    for _ in 0..600 {
        if rendering["meshes_stats"]["missing_assets"] == 0 {
            return Ok(uploaded);
        }
        session.idle(FRAME_MS);
        rendering = render(session)?;
        uploaded += uploads(&rendering);
    }
    smoke_eq!(
        rendering["meshes_stats"]["missing_assets"].clone(),
        json!(0),
        "scene_draws_every_entity"
    );
    Ok(uploaded)
}
