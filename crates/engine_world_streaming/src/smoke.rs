use engine_core::smoke::{Checked, camera, control, environment, query, refuse, render};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

fn fixture(objects: &[Value]) -> Value {
    json!({
        "id": "streaming-fixture", "revision": 1,
        "camera": {"position": [0, 60, 40], "target": [20, 20, 0], "fov_degrees": 60, "near": 0.5, "far": 20000},
        "materials": [{"id": "plain", "color": [0.8, 0.3, 0.2, 1]}],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "entities": [],
        "environment": {"streamed": {
            "objects": objects,
            "prefabs": [{"id": "lamp", "entities": [
                {"id": "post", "mesh": "box", "material": "plain", "transform": {"translation": [0, 1, 0], "scale": [0.2, 2, 0.2]}},
                {"id": "shade", "mesh": "box", "material": "plain", "transform": {"translation": [0, 2.2, 0], "scale": [12, 0.2, 12]}},
            ]}],
        }},
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

const NEAR_CELL: (u64, [i64; 2]) = (0, [0, 0]);

const MID_CELL: (u64, [i64; 2]) = (0, [3, 0]);

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    load(session, &fixture(&objects()))?;
    let rendering = look(session, 0.0, 0.0)?;
    let binned = cells(session)?;
    smoke_eq!(binned["items"], json!(5), "binning_items");
    smoke_eq!(
        grids(&binned, |grid| json!([grid["cells"], grid["loaded"]])),
        json!([[3, 2], [1, 1], [1, 1]]),
        "binning_grids"
    );
    smoke_eq!(counts(&rendering), json!([5, 4, 0, 7]), "binning_stats");
    smoke_eq!(cell_loaded(&binned, MID_CELL), false, "out_of_range_unloaded");
    hysteresis(session)?;
    settings(session)?;
    removals(session)?;
    load(session, &fixture(&objects()[..1]))?;
    look(session, 0.0, 0.0)?;
    smoke_eq!(cells(session)?["items"], json!(1), "replaced_items");
    Ok(())
}

fn hysteresis(session: &mut dyn SmokeSession) -> Checked<()> {
    look(session, 850.0, 0.0)?;
    let held = cells(session)?;
    smoke_eq!(cell_loaded(&held, NEAR_CELL), true, "hysteresis_keeps");
    smoke_eq!(cell_loaded(&held, MID_CELL), true, "hysteresis_loads");
    let rendering = look(session, 1000.0, 0.0)?;
    let left = cells(session)?;
    smoke_eq!(cell_loaded(&left, NEAR_CELL), false, "hysteresis_unloads");
    smoke_eq!(stat(&rendering, "despawned") > 0, true, "hysteresis_despawns");
    smoke_eq!(
        grids(&left, |grid| grid["loaded"].clone()),
        json!([1, 1, 1]),
        "hysteresis_grids"
    );
    Ok(())
}

fn settings(session: &mut dyn SmokeSession) -> Checked<()> {
    streaming(session, json!({"spawns_per_frame": 1}))?;
    let rendering = look(session, 0.0, 0.0)?;
    smoke_eq!(
        [stat(&rendering, "spawned") > 0, stat(&rendering, "pending") == 1],
        [true, true],
        "budget_defers"
    );
    let rendering = advance(session)?;
    smoke_eq!(
        [stat(&rendering, "pending"), stat(&rendering, "loaded")],
        [0, 4],
        "budget_catches_up"
    );

    streaming(session, json!({"frozen": true}))?;
    let rendering = look(session, 3000.0, 3000.0)?;
    smoke_eq!(stat(&rendering, "loaded"), 4, "frozen_cells_stay");
    let rendering = streaming(session, json!({"frozen": false}))?;
    smoke_eq!(stat(&rendering, "loaded"), 2, "unfrozen_small_cells_go");
    let rendering = look(session, 0.0, 0.0)?;
    smoke_eq!(stat(&rendering, "loaded"), 4, "returning_loads");
    let rendering = streaming(session, json!({"enabled": false}))?;
    smoke_eq!(
        [stat(&rendering, "loaded"), stat(&rendering, "entities")],
        [0, 0],
        "disabled_unloads"
    );
    let rendering = streaming(session, json!({"enabled": true}))?;
    smoke_eq!(stat(&rendering, "loaded"), 4, "enabled_loads");
    let refused = refuse(
        session,
        "control",
        "environment",
        json!({"world_streaming": {"unload": 0.5}}),
    )?;
    smoke_eq!(refused["code"], json!("invalid_value"), "setting_refused");
    Ok(())
}

fn removals(session: &mut dyn SmokeSession) -> Checked<()> {
    look(session, 0.0, 0.0)?;
    camera(session, view([10.0, 60.0, 40.0], [30.0, 20.0, 0.0]))?;
    advance(session)?;
    let moving = cells(session)?;
    smoke_eq!(
        moving["velocity"][0].as_f64().unwrap_or(0.0) > 100.0,
        true,
        "prefetch_measures_motion"
    );
    smoke_eq!(cell_loaded(&moving, MID_CELL), true, "prefetch_loads_ahead");

    let before = stat(&look(session, 0.0, 0.0)?, "entities");
    let removal = control(
        session,
        "world_changes",
        json!({"remove": ["crate-near"]}),
    )?;
    smoke_eq!(removal["newly_removed"], json!(1), "removal_counts");
    smoke_eq!(
        stat(&advance(session)?, "entities"),
        before.saturating_sub(1),
        "removed_despawns"
    );
    smoke_eq!(
        query(session, "world_removals", json!({}))?,
        json!({"removed": 1, "ids": ["crate-near"]}),
        "removal_inspects"
    );
    look(session, 1000.0, 0.0)?;
    smoke_eq!(
        stat(&look(session, 0.0, 0.0)?, "entities"),
        before.saturating_sub(1),
        "removed_stays_out"
    );
    control(
        session,
        "world_changes",
        json!({"restore": ["crate-near"]}),
    )?;
    smoke_eq!(
        stat(&advance(session)?, "entities"),
        before,
        "restored_spawns"
    );
    let empty = refuse(
        session,
        "control",
        "world_changes",
        json!({"remove": [""]}),
    )?;
    smoke_eq!(empty["code"], json!("out_of_range"), "empty_id_refused");
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

fn view(position: [f32; 3], target: [f32; 3]) -> Value {
    json!({
        "position": position, "target": target,
        "vertical_fov_radians": std::f32::consts::FRAC_PI_3, "near": 0.5, "far": 20000,
    })
}

fn look(session: &mut dyn SmokeSession, x: f32, z: f32) -> Checked<Value> {
    camera(session, view([x, 60.0, z + 40.0], [x + 20.0, 20.0, z]))?;
    advance(session)
}

fn advance(session: &mut dyn SmokeSession) -> Checked<Value> {
    session.step(1);
    render(session)
}

fn streaming(session: &mut dyn SmokeSession, section: Value) -> Checked<Value> {
    environment(session, json!({"world_streaming": section}))?;
    advance(session)
}

fn cells(session: &mut dyn SmokeSession) -> Checked<Value> {
    query(session, "world_cells", json!({}))
}

fn stat(rendering: &Value, field: &str) -> u64 {
    rendering["world_streaming_stats"][field]
        .as_u64()
        .unwrap_or(u64::MAX)
}

fn counts(rendering: &Value) -> Value {
    json!(["cells", "loaded", "pending", "entities"].map(|field| stat(rendering, field)))
}

fn grids(cells: &Value, pick: impl Fn(&Value) -> Value) -> Value {
    cells["grids"]
        .as_array()
        .into_iter()
        .flatten()
        .map(pick)
        .collect()
}

fn cell_loaded(cells: &Value, (grid, key): (u64, [i64; 2])) -> bool {
    cells["nearest"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|cell| {
            cell["grid"] == grid && cell["key"] == json!(key) && cell["loaded"] == true
        })
}
