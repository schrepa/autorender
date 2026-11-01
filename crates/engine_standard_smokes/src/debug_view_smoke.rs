use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::{grid, seen};
use super::scene_kit::{ask, control, load, refuse};

const SIDE: usize = 24;
const VIEWS: [&str; 7] = [
    "kinds", "levels", "cards", "shadows", "casters", "overdraw", "normals",
];
const KINDS: [&str; 9] = [
    "mesh",
    "placed",
    "card",
    "placed_card",
    "terrain",
    "cover",
    "water",
    "falls",
    "sky",
];
const SLOTTED: [&str; 2] = ["mesh", "card"];
const QUIET_FRAMES: u32 = 13;

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    load(session, &grid(SIDE, false))?;
    ask(session, "control", json!({"name": "pause"}))?;
    environment(session, json!({"reflection_probes": {"enabled": false}}))?;
    let listing = ask(session, "describe", json!({}))?;
    let listed = listing["types"]
        .as_array()
        .is_some_and(|types| types.iter().any(|kind| kind["name"] == "debug_view"));
    let wanted = json!("debug_view");
    smoke_eq!(
        seen(listed, json!(listing["types"]), &wanted),
        wanted,
        "setting_discoverable"
    );
    let look = ask(session, "inspect", json!({"component": "Environment"}))?;
    smoke_eq!(
        json!(look["value"]["debug_view"]),
        json!({"view": "off", "shade": true}),
        "setting_starts_off"
    );
    let shaded = capture(session)?;
    let wanted = json!("no debug_view");
    smoke_eq!(
        seen(
            !passes(&shaded.0).contains(&"debug_view".to_owned()),
            json!(shaded.0["render_passes"]),
            &wanted,
        ),
        wanted,
        "shaded_has_no_view_pass"
    );
    let before = exact(session)?;
    let shown = views(session, &shaded)?;
    let off = environment(session, json!({"debug_view": {"view": "off"}}))?;
    smoke_eq!(
        json!(off["environment_changed"]),
        json!(["debug_view"]),
        "view_turns_off"
    );
    smoke_eq!(json!(shown_view(session)?), json!("off"), "view_reads_off");
    let again = capture(session)?;
    let wanted = json!({"created": 0});
    smoke_eq!(
        seen(
            !passes(&again.0).contains(&"debug_view".to_owned())
                && again.0["pipelines_created"] == 0,
            json!({"passes": again.0["render_passes"], "created": again.0["pipelines_created"]}),
            &wanted,
        ),
        wanted,
        "off_creates_nothing"
    );
    smoke_eq!(exact(session)?, before.clone(), "off_is_exact");
    let unknown = refuse(
        session,
        "control",
        json!({"name": "environment", "args": {"debug_view": {"view": "everything"}}}),
    )?;
    let wanted = json!({"path": "$.args.debug_view"});
    smoke_eq!(
        seen(
            unknown["path"]
                .as_str()
                .is_some_and(|path| path.starts_with("$.args.debug_view")),
            json!(unknown),
            &wanted,
        ),
        wanted,
        "unknown_view_refused"
    );
    probes(session, &again.0)?;
    smoke_eq!(exact(session)?, before, "probe_keeps_image");
    let wanted = json!(VIEWS);
    smoke_eq!(
        seen(shown.len() == VIEWS.len(), json!(shown), &wanted),
        wanted,
        "views_draw"
    );
    Ok(())
}

fn views(
    session: &mut dyn SmokeSession,
    shaded: &(Value, Value),
) -> Result<Vec<Value>, SmokeFailure> {
    let mut shown = Vec::new();
    for view in VIEWS {
        let changed = environment(session, json!({"debug_view": {"view": view}}))?;
        smoke_eq!(
            json!(changed["environment_changed"]),
            json!(["debug_view"]),
            "view_changes_setting"
        );
        smoke_eq!(json!(shown_view(session)?), json!(view), "view_reads_back");
        let (rendering, image) = capture(session)?;
        let wanted = json!("debug_view");
        smoke_eq!(
            seen(
                passes(&rendering).contains(&"debug_view".to_owned()),
                json!({"view": view, "passes": rendering["render_passes"]}),
                &wanted,
            ),
            wanted,
            "view_pass_runs"
        );
        let wanted = json!({"draw_calls": 1});
        smoke_eq!(
            seen(
                rendering["draw_calls"].as_u64().unwrap_or(0) > 0,
                json!({"view": view, "draw_calls": rendering["draw_calls"]}),
                &wanted,
            ),
            wanted,
            "view_draws"
        );
        let wanted = json!({"not": shaded.1});
        smoke_eq!(
            seen(
                image != shaded.1,
                json!({"view": view, "image": image}),
                &wanted,
            ),
            wanted,
            "view_changes_image"
        );
        shown.push(json!({"view": view, "rendering": rendering, "image": image}));
    }
    let image = |view: &str| {
        shown
            .iter()
            .find(|frame| frame["view"] == view)
            .map(|frame| frame["image"].clone())
            .unwrap_or_default()
    };
    let wanted = json!({"not": image("kinds")});
    smoke_eq!(
        seen(
            image("levels") != image("kinds"),
            json!(image("levels")),
            &wanted,
        ),
        wanted,
        "levels_differ_from_kinds"
    );
    let overdraw = shown
        .iter()
        .find(|frame| frame["view"] == "overdraw")
        .map(|frame| frame["rendering"]["pipelines"].as_u64().unwrap_or(0))
        .unwrap_or(0);
    let pipelines = shaded.0["pipelines"].as_u64().unwrap_or(0);
    let wanted = json!({"more_than": pipelines});
    smoke_eq!(
        seen(overdraw > pipelines, json!(overdraw), &wanted),
        wanted,
        "views_create_pipelines_when_shown"
    );
    Ok(shown)
}

fn probes(session: &mut dyn SmokeSession, rendering: &Value) -> Result<(), SmokeFailure> {
    let width = rendering["resolution"]["display_width"]
        .as_u64()
        .unwrap_or(0);
    let height = rendering["resolution"]["display_height"]
        .as_u64()
        .unwrap_or(0);
    let mut kinds = Vec::new();
    for fraction in [0.3, 0.4, 0.5, 0.6, 0.7] {
        let x = (width as f64 * fraction).floor() as u64;
        let y = (height as f64 * 0.55).floor() as u64;
        let probed = control(session, "probe", json!({"x": x, "y": y, "alpha": 0}))?;
        smoke_eq!(json!(probed["pixel"]), json!([x, y]), "probe_names_pixel");
        let kind = probed["kind"].as_str().unwrap_or_default().to_owned();
        let wanted = json!(KINDS);
        smoke_eq!(
            seen(KINDS.contains(&kind.as_str()), json!(probed), &wanted),
            wanted,
            "probe_kind"
        );
        let wanted = json!({"shaded": [0, 0, 0]});
        smoke_eq!(
            seen(
                probed["shaded"]
                    .as_array()
                    .is_some_and(|shaded| shaded.len() == 3),
                json!(probed),
                &wanted,
            ),
            wanted,
            "probe_shaded"
        );
        if kind != "sky" {
            let wanted = json!("surface");
            smoke_eq!(
                seen(surface(&probed), json!(probed), &wanted),
                wanted,
                "probe_surface"
            );
        }
        if SLOTTED.contains(&kind.as_str()) {
            let wanted = json!({"object": 0, "entity": {}});
            smoke_eq!(
                seen(
                    probed["object"].is_number() && probed["entity"].is_object(),
                    json!(probed),
                    &wanted,
                ),
                wanted,
                "probe_object"
            );
        }
        kinds.push(kind);
    }
    let wanted = json!(SLOTTED);
    smoke_eq!(
        seen(
            kinds.iter().any(|kind| SLOTTED.contains(&kind.as_str())),
            json!(kinds),
            &wanted,
        ),
        wanted,
        "probe_finds_entity"
    );
    let sky = control(
        session,
        "probe",
        json!({"x": width / 2, "y": 2, "alpha": 0}),
    )?;
    let wanted = json!(KINDS);
    smoke_eq!(
        seen(
            sky["kind"]
                .as_str()
                .is_some_and(|kind| KINDS.contains(&kind)),
            json!(sky),
            &wanted,
        ),
        wanted,
        "probe_top"
    );
    let outside = refuse(
        session,
        "control",
        json!({"name": "probe", "args": {"x": width, "y": 0}}),
    )?;
    smoke_eq!(
        json!({"code": outside["code"], "path": outside["path"]}),
        json!({"code": "out_of_range", "path": "$.args.x"}),
        "probe_refuses_outside"
    );
    Ok(())
}

fn surface(probed: &Value) -> bool {
    let depth = probed["depth"].as_f64().unwrap_or(0.0);
    let world = probed["world"].as_array().is_some_and(|world| {
        world
            .iter()
            .all(|value| value.as_f64().is_some_and(f64::is_finite))
    });
    let normal = probed["normal"].as_array().map_or(0.0, |normal| {
        normal
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0).powi(2))
            .sum::<f64>()
            .sqrt()
    });
    depth > 0.0
        && depth <= 1.0
        && world
        && probed["fragments"].as_u64().unwrap_or(0) >= 1
        && (normal - 1.0).abs() < 0.01
}

fn settle(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let mut compiling = true;
    for _ in 0..600 {
        let rendering = present(session)?;
        compiling = rendering["pipelines_compiling"] == true;
        if !compiling {
            break;
        }
    }
    let wanted = json!(false);
    smoke_eq!(
        seen(!compiling, json!(compiling), &wanted),
        wanted,
        "pipelines_compile"
    );
    let (mut previous, mut stable, mut rendering) = (Value::Null, 0, Value::Null);
    let mut calm = 0;
    for frame in 1..=600_u64 {
        if frame > 8 && stable >= 5 && calm >= QUIET_FRAMES {
            return Ok(rendering);
        }
        rendering = present(session)?;
        let detail = &rendering["meshes_stats"]["level_of_detail"];
        let counts = json!([
            detail["lod0_instances"],
            detail["lod1_instances"],
            detail["lod2_instances"],
            detail["impostor_instances"]
        ]);
        let current = rendering["read_back_age"]
            .as_u64()
            .is_some_and(|age| age < frame);
        stable = if current && counts == previous {
            stable + 1
        } else {
            0
        };
        previous = if current { counts } else { Value::Null };
        calm = if quiet(&rendering["shadow_stats"]) {
            calm + 1
        } else {
            0
        };
    }
    let wanted = json!({"stable": 5});
    smoke_eq!(
        seen(stable >= 5, json!(rendering), &wanted),
        wanted,
        "read_back_settles"
    );
    let wanted = json!({"quiet_frames": QUIET_FRAMES});
    smoke_eq!(
        seen(
            calm >= QUIET_FRAMES,
            json!(rendering["shadow_stats"]),
            &wanted
        ),
        wanted,
        "shadows_settle"
    );
    Ok(rendering)
}

fn quiet(shadows: &Value) -> bool {
    shadows["tile_cut"]["cut"].as_u64() == Some(0)
        && shadows["moving_cut"]["kept"].as_u64() == Some(0)
        && ["tiles_drawn", "sky_drawn"]
            .iter()
            .all(|field| shadows[*field].as_u64() == Some(0))
}

fn present(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    Ok(control(session, "present", json!({"alpha": 0}))?["rendering"].clone())
}

fn capture(session: &mut dyn SmokeSession) -> Result<(Value, Value), SmokeFailure> {
    settle(session)?;
    let captured = ask(session, "capture", json!({"alpha": 0}))?;
    let image = captured["artifact"]["sha256"].clone();
    let wanted = json!("sha256");
    smoke_eq!(
        seen(image.is_string(), json!(captured["artifact"]), &wanted),
        wanted,
        "capture_keeps_image"
    );
    Ok((captured["rendering"].clone(), image))
}

fn exact(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let mut previous = Value::Null;
    for _ in 0..20 {
        let captured = ask(session, "capture", json!({"alpha": 0}))?;
        let image = captured["artifact"]["sha256"].clone();
        if image.is_string() && image == previous {
            return Ok(image);
        }
        previous = image;
    }
    let wanted = json!("two equal captures");
    smoke_eq!(
        seen(false, json!(previous), &wanted),
        wanted,
        "exact_captures_agree"
    );
    Ok(previous)
}

fn passes(rendering: &Value) -> Vec<String> {
    rendering["render_passes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|pass| pass.as_str().map(str::to_owned))
        .collect()
}

fn shown_view(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    Ok(ask(session, "inspect", json!({"component": "debug_view"}))?["value"]["view"].clone())
}

fn environment(session: &mut dyn SmokeSession, sections: Value) -> Result<Value, SmokeFailure> {
    control(session, "environment", sections)
}
