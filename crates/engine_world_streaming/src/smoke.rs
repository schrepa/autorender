use engine_core::smoke::{Checked, ask, camera, control, environment, query, refuse, render};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

const SCENE: &str = "streaming-fixture";

fn fixture(streamed: &[Value]) -> Value {
    json!({
        "id": SCENE, "revision": 1,
        "camera": {"position": [0, 60, 40], "target": [20, 20, 0], "fov_degrees": 60, "near": 0.5, "far": 20000},
        "materials": [{"id": "plain", "color": [0.8, 0.3, 0.2, 1]}],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "prefabs": [{"id": "lamp", "entities": [
            {"id": "post", "mesh": "box", "material": "plain", "transform": {"translation": [0, 1, 0], "scale": [0.2, 2, 0.2]}},
            {"id": "shade", "mesh": "box", "material": "plain", "transform": {"translation": [0, 2.2, 0], "scale": [12, 0.2, 12]}},
        ]}],
        "entities": [],
        "streamed": streamed,
    })
}

fn objects() -> Vec<Value> {
    vec![
        json!({"id": "crate-near", "mesh": "box", "material": "plain", "transform": {"translation": [20, 0.5, 0]}}),
        json!({"id": "crate-mid", "mesh": "box", "material": "plain", "transform": {"translation": [900, 0.5, 0]}}),
        json!({"id": "spinner", "transform": {"translation": [-30, 30, 10]}}),
        json!({"id": "spinner-blade", "parent": "spinner", "mesh": "box", "material": "plain", "transform": {"scale": [3, 0.2, 0.4]}}),
        json!({"id": "lamp-far", "prefab": "lamp", "transform": {"translation": [1500, 0, 300]}}),
        json!({"id": "tower", "mesh": "box", "material": "plain", "transform": {"translation": [5000, 30, 0], "scale": [10, 60, 10]}}),
    ]
}

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    load(session, &fixture(&objects()))?;
    ask(session, "control", json!({"name": "pause"}))?;
    let rendering = look(session, 0.0, 0.0)?;
    let binned = cells(session)?;
    smoke_eq!(binned["items"], json!(5), "binning_items");
    smoke_eq!(
        grids(&binned, |grid| json!([grid["cells"], grid["loaded"]])),
        json!([[3, 2], [1, 1], [1, 1]]),
        "binning_grids"
    );
    smoke_eq!(
        counts(&rendering),
        json!({"cells": 5, "loaded": 4, "pending": 0, "entities": 7}),
        "binning_stats"
    );
    smoke_eq!(
        loaded(session, "crate-mid")?,
        false,
        "out_of_range_unloaded"
    );
    let marked = query(session, "streamed_id", json!({"entity": "spinner"}))?;
    smoke_eq!(
        marked["id"],
        json!("streaming-fixture/spinner"),
        "root_carries_id"
    );
    hysteresis(session)?;
    settings(session)?;
    removals(session)?;
    load(session, &fixture(&objects()[..1]))?;
    render(session)?;
    smoke_eq!(cells(session)?["items"], json!(1), "replaced_items");
    Ok(())
}

fn hysteresis(session: &mut dyn SmokeSession) -> Checked<()> {
    look(session, 850.0, 0.0)?;
    smoke_eq!(loaded(session, "crate-near")?, true, "hysteresis_keeps");
    smoke_eq!(loaded(session, "crate-mid")?, true, "hysteresis_loads");
    let rendering = look(session, 1000.0, 0.0)?;
    smoke_eq!(loaded(session, "crate-near")?, false, "hysteresis_unloads");
    smoke_eq!(
        count(&rendering, "despawned") > 0,
        true,
        "hysteresis_despawns"
    );
    smoke_eq!(
        grids(&cells(session)?, |grid| grid["loaded"].clone()),
        json!([1, 1, 1]),
        "hysteresis_grids"
    );
    Ok(())
}

fn settings(session: &mut dyn SmokeSession) -> Checked<()> {
    streaming(session, json!({"spawns_per_frame": 1}))?;
    let rendering = look(session, 0.0, 0.0)?;
    smoke_eq!(
        [
            count(&rendering, "spawned") > 0,
            stat(&rendering, "pending") == 1
        ],
        [true, true],
        "budget_defers"
    );
    let rendering = render(session)?;
    smoke_eq!(
        [stat(&rendering, "pending"), stat(&rendering, "loaded")],
        [json!(0), json!(4)],
        "budget_catches_up"
    );

    streaming(session, json!({"frozen": true}))?;
    let rendering = look(session, 3000.0, 3000.0)?;
    smoke_eq!(stat(&rendering, "loaded"), json!(4), "frozen_cells_stay");
    let rendering = streaming(session, json!({"frozen": false}))?;
    smoke_eq!(
        stat(&rendering, "loaded"),
        json!(2),
        "unfrozen_small_cells_go"
    );
    let rendering = look(session, 0.0, 0.0)?;
    smoke_eq!(stat(&rendering, "loaded"), json!(4), "returning_loads");
    let rendering = streaming(session, json!({"enabled": false}))?;
    smoke_eq!(
        [stat(&rendering, "loaded"), stat(&rendering, "entities")],
        [json!(0), json!(0)],
        "disabled_unloads"
    );
    let rendering = streaming(session, json!({"enabled": true}))?;
    smoke_eq!(stat(&rendering, "loaded"), json!(4), "enabled_loads");
    let refused = refuse(
        session,
        "control",
        "environment",
        json!({"world_streaming": {"unload": 0.5}}),
    )?;
    smoke_eq!(
        refused["path"],
        json!("$.args.world_streaming.unload"),
        "setting_refused"
    );
    Ok(())
}

fn removals(session: &mut dyn SmokeSession) -> Checked<()> {
    look(session, 0.0, 0.0)?;
    session.step(1);
    look_from(session, [10.0, 60.0, 40.0], [30.0, 20.0, 0.0])?;
    session.step(1);
    let moving = cells(session)?;
    smoke_eq!(
        moving["velocity"][0].as_f64().unwrap_or(0.0) > 100.0,
        true,
        "prefetch_measures_motion"
    );
    smoke_eq!(loaded(session, "crate-mid")?, true, "prefetch_loads_ahead");

    look(session, 0.0, 0.0)?;
    let removal = control(
        session,
        "world_changes",
        json!({"remove": ["streaming-fixture/crate-near"]}),
    )?;
    smoke_eq!(removal["newly_removed"], json!(1), "removal_counts");
    smoke_eq!(loaded(session, "crate-near")?, false, "removed_despawns");
    let removals = query(session, "world_removals", json!({}))?;
    smoke_eq!(
        json!({"removed": removals["removed"], "ids": removals["ids"]}),
        json!({"removed": 1, "ids": ["streaming-fixture/crate-near"]}),
        "removal_inspects"
    );
    look(session, 1000.0, 0.0)?;
    look(session, 0.0, 0.0)?;
    smoke_eq!(loaded(session, "crate-near")?, false, "removed_stays_out");
    control(
        session,
        "world_changes",
        json!({"restore": ["streaming-fixture/crate-near"]}),
    )?;
    smoke_eq!(loaded(session, "crate-near")?, true, "restored_spawns");
    let empty = refuse(session, "control", "world_changes", json!({"remove": [""]}))?;
    smoke_eq!(empty["path"], json!("$.args.remove[0]"), "empty_id_refused");
    Ok(())
}

fn load(session: &mut dyn SmokeSession, document: &Value) -> Checked<()> {
    session.source(&document.to_string());
    control(
        session,
        "reload",
        json!({"expected_revision": "1", "revision": "1"}),
    )?;
    Ok(())
}

fn loaded(session: &mut dyn SmokeSession, id: &str) -> Checked<bool> {
    let reply = session.request("inspect", json!({"entity": id, "component": "Transform"}));
    let missing = reply
        .as_ref()
        .err()
        .map(|refused| refused["error"]["code"].clone())
        .filter(|code| code != "unknown_entity");
    smoke_eq!(missing, None::<Value>, "loaded_inspects");
    Ok(reply.is_ok())
}

fn look(session: &mut dyn SmokeSession, x: f32, z: f32) -> Checked<Value> {
    look_from(session, [x, 60.0, z + 40.0], [x + 20.0, 20.0, z])
}

fn look_from(
    session: &mut dyn SmokeSession,
    position: [f32; 3],
    target: [f32; 3],
) -> Checked<Value> {
    let changed = camera(
        session,
        json!({
            "position": position, "target": target,
            "vertical_fov_radians": std::f32::consts::FRAC_PI_3, "near": 0.5, "far": 20000,
        }),
    )?;
    Ok(changed["rendering"].clone())
}

fn streaming(session: &mut dyn SmokeSession, section: Value) -> Checked<Value> {
    let changed = environment(session, json!({"world_streaming": section}))?;
    Ok(changed["rendering"].clone())
}

fn cells(session: &mut dyn SmokeSession) -> Checked<Value> {
    query(session, "world_cells", json!({}))
}

fn stat(rendering: &Value, field: &str) -> Value {
    rendering["world_streaming_stats"][field].clone()
}

fn count(rendering: &Value, field: &str) -> u64 {
    stat(rendering, field).as_u64().unwrap_or(0)
}

fn counts(rendering: &Value) -> Value {
    let streamed = &rendering["world_streaming_stats"];
    json!({
        "cells": streamed["cells"],
        "loaded": streamed["loaded"],
        "pending": streamed["pending"],
        "entities": streamed["entities"],
    })
}

fn grids(cells: &Value, pick: impl Fn(&Value) -> Value) -> Value {
    cells["grids"]
        .as_array()
        .into_iter()
        .flatten()
        .map(pick)
        .collect()
}
