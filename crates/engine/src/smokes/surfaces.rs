use engine_core::smoke::{SmokeResult, SmokeSession};
use engine_core::{smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::scene_kit::{SceneFixture, ask, component, control, near_all, refuse, render};

fn fixture() -> Value {
    json!({
        "id": "surfaces-fixture", "revision": 1,
        "camera": {"position": [0, 9, 16], "target": [0, 0, 0], "fov_degrees": 55, "near": 0.1, "far": 500},
        "materials": [
            {"id": "ground", "color": [0.6, 0.62, 0.58, 1]},
            {"id": "plain", "color": [0.8, 0.3, 0.2, 1]},
        ],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "entities": [
            {"id": "floor", "mesh": "box", "material": "ground", "transform": {"translation": [0, -0.5, 0], "scale": [30, 1, 30]}},
            {"id": "behind", "mesh": "box", "material": "plain", "transform": {"translation": [0, 1, -2], "scale": [2, 2, 2]}},
            {"id": "tinted", "mesh": "box", "material": "plain", "tint": [0.2, 0.9, 0.3, 1], "transform": {"translation": [5, 1, 0]}},
            {"id": "probe", "reflection_probe": {"extent": [12, 6, 12]}, "transform": {"translation": [0, 3, 0]}},
        ],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let (_scene, _) = SceneFixture::load(session, &fixture())?;
    ask(session, "control", json!({"name": "pause"}))?;
    let rendering = render(session)?;
    smoke_check!(
        rendering["frame"].is_u64() && rendering["tick"].is_u64(),
        "stats_name_frame_and_tick",
        json!({"frame": rendering["frame"], "tick": rendering["tick"]}),
        json!({"frame": 0, "tick": 0})
    );
    let tint = component(session, "tinted", "Tint")?;
    smoke_check!(
        near_all(&tint, &[0.2, 0.9, 0.3, 1.0], 1e-6),
        "tint_inspects",
        json!(tint),
        json!([0.2, 0.9, 0.3, 1])
    );
    let probe = component(session, "probe", "ReflectionProbe")?;
    smoke_check!(
        near_all(&probe["extent"], &[12.0, 6.0, 12.0], 1e-6),
        "probe_inspects",
        json!(probe),
        json!({"extent": [12, 6, 12]})
    );
    let camera = ask(session, "inspect", json!({"component": "Camera"}))?["value"].clone();
    smoke_check!(camera["position"].is_array(), "camera_inspects", json!(camera), json!({"position": []}));
    environment(session)?;
    let benchmark = control(session, "benchmark", json!({"mode": "render", "warmup": 1, "samples": 3}))?;
    smoke_eq!(json!(benchmark["benchmark"]["samples"]), json!(3), "benchmark_samples");
    errors(session)
}

fn environment(session: &mut dyn SmokeSession) -> SmokeResult {
    let merged = control(
        session,
        "environment",
        json!({"merge": true, "particles": {"max_particles": 1}}),
    )?;
    smoke_eq!(
        json!({
            "changed": merged["environment_changed"],
            "before": merged["before"]["particles"]["enabled"],
            "after": merged["after"]["particles"]["enabled"],
            "budget": merged["after"]["particles"]["max_particles"],
        }),
        json!({"changed": ["particles"], "before": true, "after": true, "budget": 1}),
        "merging_keeps_the_omitted_field"
    );
    let off = control(session, "environment", json!({"merge": true, "particles": {"enabled": false}}))?;
    smoke_eq!(json!(off["after"]["particles"]["max_particles"]), json!(1), "merging_keeps_the_budget");
    control(session, "environment", json!({"particles": {"enabled": true}}))?;
    Ok(())
}

fn errors(session: &mut dyn SmokeSession) -> SmokeResult {
    let unknown = refuse(
        session,
        "control",
        json!({"name": "environment", "args": {"particles": {"max_particle": 1}}}),
    )?;
    smoke_eq!(
        json!({"code": unknown["code"], "path": unknown["path"]}),
        json!({"code": "unknown_field", "path": "$.args.particles.max_particle"}),
        "unknown_field"
    );
    let mistyped = refuse(
        session,
        "control",
        json!({"name": "camera", "args": {"position": [0, 1, "x"], "target": [0, 0, 0], "vertical_fov_radians": 1, "near": 0.1, "far": 100}}),
    )?;
    smoke_eq!(json!(mistyped["code"]), json!("invalid_type"), "mistyped");
    let game = refuse(session, "control", json!({"name": "game", "args": {"name": "workload"}}))?;
    smoke_eq!(json!(game["code"]), json!("unsupported_command"), "no_game_controls");
    let catalog = ask(session, "describe", json!({"name": "ErrorCodes"}))?["codes"].clone();
    for code in [&unknown["code"], &mistyped["code"], &game["code"]] {
        let described = catalog
            .as_array()
            .is_some_and(|codes| codes.contains(code));
        smoke_check!(described, "catalog_describes_codes", json!(code), json!("described"));
    }
    Ok(())
}
