use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::{component, control, grid, refused_with, serve};

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let mut document = grid(2, false);
    let node = json!({ "active": false, "visible": true, "clock": "real", "mobility": "moving" });
    let mover = json!({ "by": "game", "pace": "frame" });
    document["entities"][1]["node"] = node.clone();
    document["entities"][1]["mover"] = mover.clone();
    serve(session, &document)?;
    control(session, "step", json!({ "ticks": 1 }), "steps")?;
    smoke_eq!(
        component(session, "piece-0", "node", "inspects_node")?,
        node,
        "scene_node_inspected"
    );
    smoke_eq!(
        component(session, "piece-0", "mover", "inspects_mover")?,
        mover,
        "scene_mover_inspected"
    );
    smoke_eq!(
        component(session, "cap-0", "NodeWord", "inspects_word")?,
        json!({ "active": false, "visible": true, "clock": "real", "mobility": "moving" }),
        "child_inherits_scene_node"
    );
    let set = json!({ "active": true, "visible": true, "clock": null, "mobility": null });
    smoke_eq!(
        authored(session, "piece-0", "node", json!({ "active": true }))?,
        set,
        "control_sets_node"
    );
    smoke_eq!(
        authored(session, "piece-1", "node", json!({ "clock": "real" }))?,
        json!({ "active": true, "visible": true, "clock": "real", "mobility": null }),
        "control_sets_clock"
    );
    smoke_eq!(
        authored(session, "piece-0", "mover", Value::Null)?,
        Value::Null,
        "control_removes_mover"
    );
    control(session, "step", json!({ "ticks": 1 }), "steps_again")?;
    smoke_eq!(
        component(session, "cap-0", "NodeWord", "inspects_word_again")?,
        json!({ "active": true, "visible": true, "clock": "game", "mobility": "static" }),
        "child_follows_control"
    );
    smoke_eq!(
        component(session, "piece-1", "NodeWord", "inspects_clock")?["clock"].clone(),
        json!("real"),
        "clock_follows_control"
    );
    refused_with(
        session,
        "inspect",
        json!({ "entity": "piece-0", "component": "mover" }),
        "missing_component",
        "removed_mover_absent",
    )?;
    let unnamed = json!({ "entity": "piece-0", "key": "mover", "value": { "pace": "tick" } });
    refused_with(
        session,
        "control",
        json!({ "name": "component", "args": unnamed }),
        "missing_field",
        "mover_names_its_mover",
    )?;
    Ok(())
}

fn authored(
    session: &mut dyn SmokeSession,
    entity: &str,
    key: &str,
    value: Value,
) -> Result<Value, SmokeFailure> {
    let args = json!({ "entity": entity, "key": key, "value": value });
    Ok(control(session, "component", args, "authors_component")?["result"]["value"].clone())
}
