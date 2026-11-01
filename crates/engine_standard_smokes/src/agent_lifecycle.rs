use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::json;

use super::host_parity::boxed;
use super::requests::{control, inspected, names, seen, serve, status, tick};

const PHYSICS: &str = "engine_physics";

pub(crate) fn agent_lifecycle(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    serve(session, &boxed("agent-lifecycle"))?;
    let first = status(session, "status_answers")?;
    let bloom = inspected(session, "bloom", "inspects_bloom")?;
    let brighter = json!({ "bloom": { "intensity": 2.5 } });
    control(session, "environment", brighter, "environment_taken")?;
    control(session, "step", json!({ "ticks": 5 }), "steps")?;
    let changed = inspected(session, "bloom", "inspects_changed_bloom")?;
    let wanted = json!({ "differs_from": bloom });
    smoke_eq!(
        seen(changed != bloom, changed.clone(), &wanted),
        wanted,
        "temporary_change_applies"
    );
    session.without(&[]);
    control(session, "pause", json!({}), "pauses_replaced_runtime")?;
    let replaced = status(session, "replaced_runtime_answers")?;
    smoke_eq!(
        json!([tick(&replaced), replaced["observation"]["scene"]]),
        json!([tick(&first), first["observation"]["scene"]]),
        "runtime_replacement"
    );
    smoke_eq!(
        inspected(session, "bloom", "inspects_replaced_bloom")?,
        bloom,
        "replacement_drops_temporary_changes"
    );
    let standard = names(&first["result"]["features"]);
    smoke_eq!(
        json!(names(&replaced["result"]["features"])),
        json!(standard),
        "replacement_adds_standard_bricks"
    );
    session.without(&[PHYSICS]);
    control(
        session,
        "pause",
        json!({}),
        "pauses_runtime_without_physics",
    )?;
    let without = status(session, "runtime_without_physics_answers")?;
    let kept: Vec<&String> = standard.iter().filter(|name| *name != PHYSICS).collect();
    smoke_eq!(
        json!(names(&without["result"]["features"])),
        json!(kept),
        "replacement_leaves_bricks_out"
    );
    Ok(())
}
