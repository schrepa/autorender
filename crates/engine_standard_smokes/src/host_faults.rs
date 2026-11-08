use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::{control, frames, named, names, refused_with, reply, seen, serve, status};

fn boxed(id: &str) -> Value {
    json!({
        "id": id, "revision": 1,
        "materials": [{ "id": "plain", "color": [0.8, 0.3, 0.2, 1] }],
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": [{ "id": "box", "drawn": {"mesh": "box", "material": "plain"}, "transform": { "translation": [0, 0, 0] } },
            { "id": "camera", "transform": { "translation": [0, 2, 4], "rotation_degrees": [-26.5651, 0, 0] }, "node": { "clock": "real", "mobility": "moving" }, "camera": { "near": 0.1, "far": 100, "projection": { "Perspective": { "vertical_fov_radians": std::f64::consts::FRAC_PI_3 } } }, "view": {} }],
    })
}

fn unsimulated(id: &str) -> Value {
    let mut document = boxed(id);
    document["environment"] =
        json!({ "collision": { "enabled": false }, "physics": { "enabled": false } });
    document
}

type Case = (&'static str, &'static str, Value, bool);

fn answered(session: &mut dyn SmokeSession, cases: Vec<Case>) -> Result<(), SmokeFailure> {
    for (check, verb, body, expect_ok) in cases {
        let reply = reply(session, verb, body);
        let wanted = json!({ "ok": expect_ok });
        smoke_eq!(
            seen(
                reply["ok"] == expect_ok,
                json!({ "ok": reply["ok"], "error": reply["error"] }),
                &wanted,
            ),
            wanted,
            check
        );
    }
    Ok(())
}

pub(crate) fn host_faults(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    serve(session, &unsimulated("host-parity-faults"))?;
    let bloom = json!({ "bloom": { "intensity": 0.5 } });
    control(session, "environment", bloom, "bloom_before_the_fault")?;
    control(
        session,
        "debug_fault",
        json!({ "feature": "bloom" }),
        "debug_fault_taken",
    )?;
    let shown = frames(session, 30, "frames_after_the_fault")?;
    let passes = names(&shown["result"]["rendering"]["render_passes"]);
    let wanted = json!("composite");
    smoke_eq!(
        seen(
            passes.iter().any(|pass| pass == "composite"),
            json!(passes),
            &wanted,
        ),
        wanted,
        "frames_composite_after_the_fault"
    );
    let faulted = status(session, "status_after_the_fault")?;
    smoke_eq!(
        faulted["result"]["rendering"]["disabled_features"].clone(),
        json!(["bloom"]),
        "fault_disables_bloom"
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
