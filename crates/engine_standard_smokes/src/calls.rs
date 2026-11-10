use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, control, query, settled};
use serde_json::{Value, json};

use super::requests::seen;

const SPAWNED: usize = 100;
const TICKS: usize = 4;
const GATE: &str = "gate";
const BUTTON: &str = "go";

fn fixture() -> Value {
    json!({
        "id": "calls-by-name", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": GATE, "transform": {"translation": [0.0, 0.0, 0.0]},
             "collider": {"shape": {"kind": "cuboid", "half_extents": [1.0, 1.0, 1.0]}, "sensor": true}},
            {"id": "menu", "ui": {"root": {"kind": "panel", "children": [
                {"id": BUTTON, "kind": "button", "text": "Go", "width": {"px": 120.0}, "height": {"px": 40.0},
                 "action": {"emit": BUTTON}},
            ]}}},
            {"id": "camera", "transform": {"translation": [0.0, 30.0, 60.0], "rotation_degrees": [-26.0, 0.0, 0.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.1, "far": 1000.0, "projection": {"Perspective": {"vertical_fov_radians": 1.0}}},
             "view": {}},
        ],
    })
}

fn far(index: usize, tick: usize) -> [f64; 3] {
    let (row, column) = ((index / 10) as f64, (index % 10) as f64);
    [20.0 + column * 4.0, 0.25 * tick as f64, row * 4.0]
}

fn spawned() -> Value {
    let entities: Vec<Value> = (0..SPAWNED)
        .map(|index| {
            json!({
                "node": {"mobility": "moving"},
                "transform": {"translation": far(index, 0)},
                "collider": {"shape": {"kind": "cuboid", "half_extents": [0.25, 0.25, 0.25]}},
            })
        })
        .collect();
    json!({ "entities": entities })
}

fn cursor(session: &mut dyn SmokeSession, name: &str) -> Checked<u64> {
    let page = query(session, "events", json!({"name": name}))?;
    Ok(page["next"].as_u64().unwrap_or(0))
}

fn rows_after(session: &mut dyn SmokeSession, name: &str, after: u64) -> Checked<Vec<Value>> {
    let page = query(session, "events", json!({"name": name, "after": after}))?;
    Ok(page["rows"].as_array().cloned().unwrap_or_default())
}

fn all_done(reply: &Value, count: usize, check: &'static str) -> Checked<()> {
    let wanted = json!({"done": count, "refused": 0});
    let got = json!({"done": reply["done"], "refused": reply["refused"]});
    smoke_eq!(
        seen(got == wanted, reply["rows"].clone(), &wanted),
        wanted,
        check
    );
    Ok(())
}

fn ids_of(reply: &Value) -> Vec<String> {
    reply["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| row["done"]["entity"].as_str().map(str::to_owned))
        .collect()
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&fixture().to_string());
    settled(session)?;
    let collisions = cursor(session, "collision")?;
    let presses = cursor(session, "ui_pressed")?;
    let spawn = control(session, "spawn", spawned())?;
    all_done(&spawn, SPAWNED, "spawn_lands_every_row")?;
    let ids = ids_of(&spawn);
    smoke_eq!(ids.len(), SPAWNED, "spawn_names_every_entity");
    moved(session, &ids)?;
    collided(session, collisions, &ids[0])?;
    pressed(session, presses)?;
    changed(session, &ids[0])?;
    removed(session, &ids)
}

fn moved(session: &mut dyn SmokeSession, ids: &[String]) -> Checked<()> {
    for tick in 1..=TICKS {
        let moves: Vec<Value> = ids
            .iter()
            .enumerate()
            .map(|(index, id)| {
                let translation = if index == 0 && tick >= 2 {
                    [0.0, 0.0, 0.0]
                } else {
                    far(index, tick)
                };
                json!({"entity": id, "translation": translation})
            })
            .collect();
        let written = control(session, "teleport", json!({ "moves": moves }))?;
        all_done(&written, ids.len(), "teleport_writes_every_row")?;
        session.step(1);
    }
    Ok(())
}

fn collided(session: &mut dyn SmokeSession, after: u64, entered: &str) -> Checked<()> {
    let rows = rows_after(session, "collision", after)?;
    let found = rows.iter().any(|row| {
        row["kind"] == "enter"
            && ((row["first"] == GATE && row["second"] == entered)
                || (row["first"] == entered && row["second"] == GATE))
    });
    let wanted = json!({"kind": "enter", "pair": [GATE, entered]});
    smoke_eq!(
        seen(found, json!(rows), &wanted),
        wanted,
        "collision_raised_as_event"
    );
    Ok(())
}

fn pressed(session: &mut dyn SmokeSession, after: u64) -> Checked<()> {
    control(session, "ui_press", json!({"id": BUTTON}))?;
    session.step(1);
    let rows = rows_after(session, "ui_pressed", after)?;
    let found = rows.iter().any(|row| row["button"] == BUTTON);
    let wanted = json!({"button": BUTTON});
    smoke_eq!(
        seen(found, json!(rows), &wanted),
        wanted,
        "ui_press_raised_as_event"
    );
    Ok(())
}

fn changed(session: &mut dyn SmokeSession, id: &str) -> Checked<()> {
    let change = control(
        session,
        "change",
        json!({"changes": [
            {"entity": id, "component": "collider", "data": {"shape": {"kind": "sphere", "radius": 0.5}}},
        ]}),
    )?;
    all_done(&change, 1, "change_lands")?;
    let refused = control(
        session,
        "change",
        json!({"changes": [{"entity": "nobody", "component": "collider", "data": null}]}),
    )?;
    let wanted = json!({"done": 0, "refused": 1, "reason": "unknown_entity"});
    let got = json!({
        "done": refused["done"],
        "refused": refused["refused"],
        "reason": refused["rows"][0]["refused"]["reason"],
    });
    smoke_eq!(got, wanted, "change_refuses_per_row");
    Ok(())
}

fn removed(session: &mut dyn SmokeSession, ids: &[String]) -> Checked<()> {
    let removal = control(session, "remove", json!({ "entities": ids }))?;
    all_done(&removal, ids.len(), "remove_lands_every_row")?;
    let again = control(session, "remove", json!({ "entities": &ids[..1] }))?;
    let wanted = json!({"done": 0, "refused": 1});
    let got = json!({"done": again["done"], "refused": again["refused"]});
    smoke_eq!(got, wanted, "removed_entities_are_gone");
    Ok(())
}
