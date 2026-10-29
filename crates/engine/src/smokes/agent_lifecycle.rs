use engine_core::smoke::{SmokeResult, SmokeSession};
use serde_json::json;

use super::host_parity::boxed;
use super::requests::{control, environment, inspected, names, serve, status, tick};

pub(crate) fn agent_lifecycle(session: &mut dyn SmokeSession) -> SmokeResult {
    serve(session, &boxed("agent-lifecycle"), &[])?;
    let first = status(session, "status_answers")?;
    let bloom = inspected(session, "Bloom", "inspects_bloom")?;
    let brighter = json!({ "bloom": { "intensity": 2.5 } });
    environment(session, brighter, "environment_taken")?;
    control(session, "step", json!({ "ticks": 5 }), "steps")?;
    let changed = inspected(session, "Bloom", "inspects_changed_bloom")?;
    engine_core::smoke_check!(
        changed != bloom,
        "temporary_change_applies",
        changed.clone(),
        json!({ "differs_from": bloom })
    );
    session.restart(&[]);
    control(session, "pause", json!({}), "pauses_replaced_runtime")?;
    let replaced = status(session, "replaced_runtime_answers")?;
    engine_core::smoke_eq!(
        json!([tick(&replaced), replaced["observation"]["scene"]]),
        json!([tick(&first), first["observation"]["scene"]]),
        "runtime_replacement"
    );
    engine_core::smoke_eq!(
        inspected(session, "Bloom", "inspects_replaced_bloom")?,
        bloom,
        "replacement_drops_temporary_changes"
    );
    let standard: Vec<&str> = crate::bricks::standard().names().collect();
    engine_core::smoke_eq!(
        json!(names(&replaced["result"]["features"])),
        json!(standard),
        "replacement_adds_standard_bricks"
    );
    session.restart(&["physics"]);
    control(
        session,
        "pause",
        json!({}),
        "pauses_runtime_without_physics",
    )?;
    let without = status(session, "runtime_without_physics_answers")?;
    let kept: Vec<&str> = standard
        .iter()
        .copied()
        .filter(|&name| name != "physics")
        .collect();
    engine_core::smoke_eq!(
        json!(names(&without["result"]["features"])),
        json!(kept),
        "replacement_leaves_bricks_out"
    );
    Ok(())
}
